"""football-data.co.uk per-league CSVs.

Why:
- The GBM is trained on FPL EPL data joined with our own features. Adding
  bookmaker odds and per-club season-form from a second source is a cheap
  way to enrich training without scraping match reports.
- football-data.co.uk publishes one CSV per (league, season) at
  https://www.football-data.co.uk/mmz4281/<season>/<code>.csv where
  `season` is e.g. `2425` for 2024-25 and `code` is the league
  (E0=Premier League, E1=Championship, SP1=La Liga, D1=Bundesliga, etc.).

Output:
- Raw CSVs cached under `data/external/football_data/<season>_<code>.csv`.
- A normalized parquet at `data/external/fd_matches.parquet` with columns:
      league, season, date, home, away, fthg, ftag, ftr,
      home_odds_pinnacle, draw_odds_pinnacle, away_odds_pinnacle (where present)
- A per-club season Elo at `data/external/club_elo.csv` (built from the
  league CSVs the same way as international_elo.compute_elo).
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from .international_elo import BASE_ELO, HOME_ADVANTAGE

DEFAULT_CACHE_DIR = Path("data/external/football_data")
DEFAULT_PARQUET = Path("data/external/fd_matches.parquet")

# Standard league codes used by the site. Trim/expand to taste.
LEAGUES = ("E0", "SP1", "D1", "I1", "F1")  # Premier, La Liga, Bundes, Serie A, Ligue 1


def _season_url(season: str, code: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"


def fetch_season(season: str, code: str,
                 cache_dir: Path = DEFAULT_CACHE_DIR,
                 refresh: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{season}_{code}.csv"
    if refresh or not out.exists():
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            r = client.get(_season_url(season, code))
            if r.status_code != 200:
                return out  # missing season, leave path absent
            out.write_bytes(r.content)
    return out


def _normalize(csv_path: Path, season: str, league: str) -> pd.DataFrame:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    # The site sometimes ships a stray byte; use Python engine to be resilient.
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    cols_lower = {c.lower(): c for c in df.columns}
    needed = ["date", "hometeam", "awayteam", "fthg", "ftag", "ftr"]
    for n in needed:
        if n not in cols_lower:
            return pd.DataFrame()
    out = pd.DataFrame({
        "league": league,
        "season": season,
        "date": pd.to_datetime(df[cols_lower["date"]], dayfirst=True, errors="coerce"),
        "home": df[cols_lower["hometeam"]].astype(str),
        "away": df[cols_lower["awayteam"]].astype(str),
        "fthg": pd.to_numeric(df[cols_lower["fthg"]], errors="coerce"),
        "ftag": pd.to_numeric(df[cols_lower["ftag"]], errors="coerce"),
        "ftr":  df[cols_lower["ftr"]].astype(str),
    })
    # Pinnacle (or Bet365 fallback) closing odds.
    for src in ("psh", "psd", "psa", "b365h", "b365d", "b365a"):
        if src in cols_lower:
            out[src] = pd.to_numeric(df[cols_lower[src]], errors="coerce")
    return out.dropna(subset=["date", "home", "away"])


def refresh_all(seasons: tuple[str, ...] = ("2223", "2324", "2425"),
                leagues: tuple[str, ...] = LEAGUES,
                cache_dir: Path = DEFAULT_CACHE_DIR,
                refresh: bool = False) -> pd.DataFrame:
    frames = []
    for season in seasons:
        for code in leagues:
            path = fetch_season(season, code, cache_dir, refresh)
            sub = _normalize(path, season, code)
            if not sub.empty:
                frames.append(sub)
    if not frames:
        return pd.DataFrame()
    all_matches = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    DEFAULT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    all_matches.to_parquet(DEFAULT_PARQUET, index=False)
    return all_matches


def compute_club_elo_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Per-(club, date) snapshot of Elo BEFORE each match the club played.

    Returns a long DataFrame: club, date, elo_before. Useful for joining
    historical Elo without lookahead bias - for a training match on date D,
    take the latest row where date < D.
    """
    elos: dict[str, float] = {}
    K = 20
    rows = []
    for r in matches.itertuples(index=False):
        h, a = str(r.home), str(r.away)
        e_h = elos.get(h, BASE_ELO); e_a = elos.get(a, BASE_ELO)
        rows.append({"club": h, "date": r.date, "elo_before": e_h})
        rows.append({"club": a, "date": r.date, "elo_before": e_a})
        if pd.isna(r.fthg) or pd.isna(r.ftag):
            continue
        exp_h = 1.0 / (1.0 + 10 ** (((e_a) - (e_h + HOME_ADVANTAGE)) / 400))
        if r.fthg > r.ftag:
            s_h, s_a = 1.0, 0.0
        elif r.fthg < r.ftag:
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5
        margin = max(1, abs(int(r.fthg) - int(r.ftag)))
        gd_mult = (margin + 1) ** 0.5 / (2 ** 0.5)
        elos[h] = e_h + K * gd_mult * (s_h - exp_h)
        elos[a] = e_a + K * gd_mult * (s_a - (1.0 - exp_h))
    return pd.DataFrame(rows).sort_values(["club", "date"]).reset_index(drop=True)


def load_matches(path: Path = DEFAULT_PARQUET) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
