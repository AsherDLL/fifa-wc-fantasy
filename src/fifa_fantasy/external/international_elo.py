"""International country strength derived from martj42/international_results.

Why:
- The hand-maintained `data/static/fifa_rankings.csv` is a point-in-time
  snapshot. International form moves fast (recent qualifiers swing it).
- martj42's CSV has every international match since 1872; we update an
  Elo rating per country through the whole history and produce a current
  rating + recent-form features.

Output columns (`data/external/country_elo.csv`):
    country_name        canonical English name (from martj42)
    elo                 current Elo rating (anchored to 1500 mean)
    matches             matches ever played
    last10_form         (W*3 + D*1) / 30 over last 10 matches, [0..1]
    last24m_goals_for   average goals scored in last 24 months
    last24m_goals_ag    average goals conceded in last 24 months
    last_match_date     ISO date of most recent match

The downstream features pipeline maps `country_name` -> FIFA squad name
or abbreviation via the small lookup in `mapping.py`. Countries with no
match in the last 24 months fall back to current Elo only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import pandas as pd

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DEFAULT_CACHE = Path("data/external/cache/international_results.csv")
DEFAULT_OUTPUT = Path("data/external/country_elo.csv")

# Tournament weights for K-factor (per FIFA's own Elo-style adjustment).
# Friendly games shouldn't move ratings as much as a World Cup.
K_FRIENDLY = 10
K_QUALIFIER = 25
K_TOURNAMENT = 40
K_WORLD_CUP = 60

HOME_ADVANTAGE = 60  # Elo points added to home side's expected score
BASE_ELO = 1500


def fetch_results(cache_path: Path = DEFAULT_CACHE, refresh: bool = False) -> pd.DataFrame:
    """Download (or load cached) martj42 results CSV."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not cache_path.exists():
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            r = client.get(RESULTS_URL)
            r.raise_for_status()
        cache_path.write_bytes(r.content)
    df = pd.read_csv(cache_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _k_factor(tournament: str) -> int:
    t = (tournament or "").lower()
    if "fifa world cup" in t and "qualif" not in t:
        return K_WORLD_CUP
    if "qualif" in t:
        return K_QUALIFIER
    if "friendly" in t:
        return K_FRIENDLY
    return K_TOURNAMENT


def compute_elo(results: pd.DataFrame) -> pd.DataFrame:
    """Roll a per-country Elo through the full match history.

    Returns the same DataFrame with `home_elo_before`, `away_elo_before`,
    `home_elo_after`, `away_elo_after` columns appended.
    """
    elos: dict[str, float] = {}
    rows = results.itertuples(index=False)
    home_before, away_before, home_after, away_after = [], [], [], []
    for r in rows:
        home, away = str(r.home_team), str(r.away_team)
        e_h = elos.get(home, BASE_ELO)
        e_a = elos.get(away, BASE_ELO)
        # Effective Elo gap with home advantage (neutral fixtures get none).
        adv = 0 if bool(r.neutral) else HOME_ADVANTAGE
        exp_h = 1.0 / (1.0 + 10 ** (((e_a) - (e_h + adv)) / 400))
        exp_a = 1.0 - exp_h
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            score_h = score_a = 0.5  # no result -> draw, doesn't move much
        elif r.home_score > r.away_score:
            score_h, score_a = 1.0, 0.0
        elif r.home_score < r.away_score:
            score_h, score_a = 0.0, 1.0
        else:
            score_h, score_a = 0.5, 0.5
        k = _k_factor(getattr(r, "tournament", ""))
        # Larger goal margins move Elo more.
        margin = 1
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            margin = max(1, abs(int(r.home_score) - int(r.away_score)))
        gd_mult = (margin + 1) ** 0.5 / (2 ** 0.5)  # 1, 1.22, 1.41, 1.58, ...
        new_e_h = e_h + k * gd_mult * (score_h - exp_h)
        new_e_a = e_a + k * gd_mult * (score_a - exp_a)
        home_before.append(e_h); away_before.append(e_a)
        home_after.append(new_e_h); away_after.append(new_e_a)
        elos[home] = new_e_h
        elos[away] = new_e_a
    out = results.copy()
    out["home_elo_before"] = home_before
    out["away_elo_before"] = away_before
    out["home_elo_after"] = home_after
    out["away_elo_after"] = away_after
    return out


def summarize(results_with_elo: pd.DataFrame) -> pd.DataFrame:
    """Per-country snapshot: current Elo + recent form features."""
    df = results_with_elo
    # Take the most recent post-match Elo per country (whether home or away).
    home_last = (df[["date", "home_team", "home_elo_after"]]
                 .rename(columns={"home_team": "country_name", "home_elo_after": "elo"}))
    away_last = (df[["date", "away_team", "away_elo_after"]]
                 .rename(columns={"away_team": "country_name", "away_elo_after": "elo"}))
    long = pd.concat([home_last, away_last], ignore_index=True).sort_values("date")
    snap = long.groupby("country_name").tail(1).set_index("country_name")[["date", "elo"]]
    snap = snap.rename(columns={"date": "last_match_date"})

    # Match-count and last-10 form using full history.
    match_count = pd.concat([df["home_team"], df["away_team"]], ignore_index=True).value_counts()
    snap["matches"] = match_count

    cutoff_24m = df["date"].max() - timedelta(days=730)
    recent = df[df["date"] >= cutoff_24m].copy()

    def form_for(country: str) -> tuple[float, float, float]:
        sub = (recent[(recent["home_team"] == country) | (recent["away_team"] == country)]
               .tail(10))
        if sub.empty:
            return 0.0, 0.0, 0.0
        pts = 0; gf = 0.0; ga = 0.0; n = 0
        for r in sub.itertuples(index=False):
            if r.home_team == country:
                my, opp = r.home_score, r.away_score
            else:
                my, opp = r.away_score, r.home_score
            if pd.isna(my) or pd.isna(opp):
                continue
            gf += float(my); ga += float(opp); n += 1
            if my > opp: pts += 3
            elif my == opp: pts += 1
        return pts / 30.0, gf / max(n, 1), ga / max(n, 1)

    form_rows = [form_for(c) for c in snap.index]
    snap["last10_form"] = [r[0] for r in form_rows]
    snap["last24m_goals_for"] = [r[1] for r in form_rows]
    snap["last24m_goals_ag"] = [r[2] for r in form_rows]
    snap = snap.reset_index()
    return snap


def refresh(refresh_cache: bool = True,
            output_path: Path = DEFAULT_OUTPUT,
            cache_path: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """End-to-end: fetch -> roll Elo -> summarize -> persist CSV."""
    results = fetch_results(cache_path=cache_path, refresh=refresh_cache)
    with_elo = compute_elo(results)
    snap = summarize(with_elo)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snap.to_csv(output_path, index=False)
    return snap


def load(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[
            "country_name", "elo", "matches", "last10_form",
            "last24m_goals_for", "last24m_goals_ag", "last_match_date",
        ])
    return pd.read_csv(path)
