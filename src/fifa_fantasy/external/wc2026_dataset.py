"""WC 2026 community dataset (mominullptr/FIFA-World-Cup-2026-Dataset).

Why:
- Our collector only sees what the fantasy API exposes: no real match xG,
  no actual lineups/minutes, no referee histories, no per-match corners.
- This CC0-1.0 dataset (sources: FIFA.com, Sofascore, FBref; updated after
  each match) fills those gaps for two consumers:
    * walk-forward feature configs E/F in scripts/wc_forward_validation.py
      (real team xG form, real player start rates);
    * scripts/match_predictions.py (round-8 winner/score/corners/cards/
      scorer predictions).

Only the seven CSVs we consume are fetched. Raw copies land in the
gitignored cache; canonical copies are committed under data/external/wc2026/
(CC0 permits redistribution; see docs/data-provenance.md).

Trust and fallbacks: validate() cross-checks every completed match score
against our own fixtures parquet and records a freshness status. When the
dataset lags (missing lineups/xG for recent rounds), the feature builders
emit rows only for covered rounds (downstream merges leave NaN, which
LightGBM handles), and match_predictions.py falls back per component.

Caveats: external `kickoff_time_utc` is actually local kickoff time, so
the match spine joins on team pair + nearest calendar date, never on time.
Goals and xG include extra time for AET matches (small upward bias,
accepted). External stage_id does not map to fantasy rounds; round ids
always come from our fixtures parquet via the spine.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from fifa_fantasy.external.team_news.matcher import PlayerNameMatcher

BASE_URL = ("https://raw.githubusercontent.com/mominullptr/"
            "FIFA-World-Cup-2026-Dataset/main/")
FILES = ("matches", "teams", "referees", "match_lineups",
         "match_team_stats", "squads_and_players", "player_stats")
DEFAULT_CACHE_DIR = Path("data/external/cache/wc2026_dataset")
DEFAULT_OUT_DIR = Path("data/external/wc2026")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_VALIDATION_OUT = Path("data/evaluation/wc2026_dataset_validation.json")

FORM_WINDOW = 3  # same trailing window as the rest of the pipeline


def _squash(s: str) -> str:
    """Lowercase, accent-strip, drop non-letters ("O'Reilly" == "Oreilly")."""
    import re
    import unicodedata
    d = unicodedata.normalize("NFD", s)
    return re.sub(r"[^a-z]", "", "".join(
        c for c in d if unicodedata.category(c) != "Mn").lower())


def match_player_ids(players: pd.DataFrame,
                     matcher: PlayerNameMatcher) -> pd.Series:
    """Our player_id per external roster row (player_name/abbr/position).

    Tiers: the shared PlayerNameMatcher with position, then a local
    first+last token match within country (resolves the Hernandez and
    Martinez sibling collisions and apostrophe variants the shared
    last-name tiers cannot).
    """
    ref = matcher.players.copy()
    toks = ref["full_name"].astype(str).str.split()
    ref["first_sq"] = toks.str[0].map(_squash)
    ref["last_sq"] = toks.str[-1].map(_squash)
    ids: list[float] = []
    for r in players.itertuples(index=False):
        res = matcher.match(str(r.player_name), country_abbr=str(r.abbr),
                            position=str(r.position))
        pid: float = res.player_id if res.player_id is not None else float("nan")
        if pd.isna(pid):
            parts = str(r.player_name).split()
            cand = ref[(ref["country_abbr"] == str(r.abbr))
                       & (ref["first_sq"] == _squash(parts[0]))
                       & (ref["last_sq"] == _squash(parts[-1]))]
            if len(cand) == 1:
                pid = int(cand.iloc[0]["player_id"])
        ids.append(pid)
    return pd.Series(ids, index=players.index)


def fetch(cache_dir: Path = DEFAULT_CACHE_DIR,
          refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Download (or load cached) the consumed CSVs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for name in FILES:
            path = cache_dir / f"{name}.csv"
            if refresh or not path.exists():
                r = client.get(f"{BASE_URL}{name}.csv")
                r.raise_for_status()
                path.write_bytes(r.content)
            out[name] = pd.read_csv(path)
    return out


def refresh(refresh_cache: bool = True,
            out_dir: Path = DEFAULT_OUT_DIR,
            cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, pd.DataFrame]:
    """Fetch and persist canonical copies under data/external/wc2026/."""
    frames = fetch(cache_dir=cache_dir, refresh=refresh_cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return frames


def load(name: str, out_dir: Path = DEFAULT_OUT_DIR) -> pd.DataFrame:
    """Read one canonical CSV; empty frame if absent so joins degrade."""
    path = out_dir / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _latest(raw_dir: Path, prefix: str) -> pd.DataFrame:
    files = sorted(raw_dir.glob(f"{prefix}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {prefix}_*.parquet in {raw_dir}")
    return pd.read_parquet(files[-1])


def team_id_map(teams: pd.DataFrame | None = None,
                raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """[team_id, abbr, squad_id, country]; external fifa_code == our abbr."""
    if teams is None:
        teams = load("teams")
    squads = _latest(raw_dir, "squads")
    m = teams.merge(squads, left_on="fifa_code", right_on="abbr", how="left")
    return m.rename(columns={"name": "country"})[
        ["team_id", "abbr", "squad_id", "country"]]


def match_spine(matches: pd.DataFrame | None = None,
                tmap: pd.DataFrame | None = None,
                raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Join external matches to our fixtures on team pair + nearest date.

    Returns one row per external match with our fixture_id/round_id and
    both squad ids attached, plus a `completed` flag.
    """
    if matches is None:
        matches = load("matches")
    if tmap is None:
        tmap = team_id_map(raw_dir=raw_dir)
    fixtures = _latest(raw_dir, "fixtures")
    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(
        fixtures["kickoff"].str[:10], errors="coerce")

    by_id = tmap.set_index("team_id")
    m = matches.copy()
    m["home_abbr"] = m["home_team_id"].map(by_id["abbr"])
    m["away_abbr"] = m["away_team_id"].map(by_id["abbr"])
    m["date"] = pd.to_datetime(m["date"], errors="coerce")

    fx = fixtures[["fixture_id", "round_id", "home_squad_abbr",
                   "away_squad_abbr", "home_squad_id", "away_squad_id",
                   "date", "home_score", "away_score", "status"]]
    j = m.merge(fx, left_on=["home_abbr", "away_abbr"],
                right_on=["home_squad_abbr", "away_squad_abbr"],
                how="left", suffixes=("_ext", "_ours"))
    # A pair can meet twice (group + knockout rematch): keep nearest date.
    j["date_gap"] = (j["date_ext"] - j["date_ours"]).abs().dt.days
    j = (j.sort_values("date_gap")
          .drop_duplicates("match_id", keep="first")
          .sort_values("match_id"))
    j["completed"] = j["status_ext"].eq("Completed")
    return j.reset_index(drop=True)


XG_FORM_COLUMNS = ["squad_id", "country", "round_id",
                   "team_xg_form_real", "team_xga_form_real"]


def team_xg_form(window: int = FORM_WINDOW,
                 raw_dir: Path = DEFAULT_RAW_DIR,
                 out_dir: Path = DEFAULT_OUT_DIR) -> pd.DataFrame:
    """Leak-free trailing real-xG form per (squad, fantasy round).

    For each team and round k, the mean xG for/against over its last
    `window` completed matches in rounds strictly before k. Rows exist for
    every round in our fixtures, including future ones. Empty (typed)
    frame when the dataset has not been fetched, so merges degrade to NaN.
    """
    if load("matches", out_dir=out_dir).empty:
        return pd.DataFrame(columns=XG_FORM_COLUMNS)
    spine = match_spine(raw_dir=raw_dir)
    done = spine[spine["completed"] & spine["home_xg"].notna()]
    long = pd.concat([
        done.rename(columns={"home_squad_id": "squad_id",
                             "home_xg": "xg_for", "away_xg": "xg_against"})[
            ["squad_id", "round_id", "xg_for", "xg_against"]],
        done.rename(columns={"away_squad_id": "squad_id",
                             "away_xg": "xg_for", "home_xg": "xg_against"})[
            ["squad_id", "round_id", "xg_for", "xg_against"]],
    ], ignore_index=True)
    tmap = team_id_map(raw_dir=raw_dir).set_index("squad_id")
    all_rounds = sorted(_latest(raw_dir, "fixtures")["round_id"].unique())
    rows = []
    for squad_id, grp in long.groupby("squad_id"):
        grp = grp.sort_values("round_id")
        for k in all_rounds:
            prior = grp[grp["round_id"] < k].tail(window)
            if prior.empty:
                continue
            rows.append({
                "squad_id": int(squad_id),
                "country": str(tmap.loc[squad_id, "country"]),
                "round_id": int(k),
                "team_xg_form_real": float(prior["xg_for"].mean()),
                "team_xga_form_real": float(prior["xg_against"].mean()),
            })
    return pd.DataFrame(rows)


def _matched_lineups(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Lineup rows with our player_id attached via name matching.

    Full team roster per completed match: players absent from the lineup
    sheet get started=0, minutes=0 (a real signal, not missing data).
    """
    lineups = load("match_lineups")
    players = load("squads_and_players")
    spine = match_spine(raw_dir=raw_dir)
    done = spine[spine["completed"]][["match_id", "round_id"]]

    tmap = team_id_map(raw_dir=raw_dir).set_index("team_id")
    matcher = PlayerNameMatcher.from_latest(raw_dir)
    players = players.copy()
    players["abbr"] = players["team_id"].map(tmap["abbr"])
    players["our_player_id"] = match_player_ids(players, matcher)

    team_matches = pd.concat([
        load("matches").rename(columns={"home_team_id": "team_id"})[
            ["match_id", "team_id"]],
        load("matches").rename(columns={"away_team_id": "team_id"})[
            ["match_id", "team_id"]],
    ]).merge(done, on="match_id")
    roster = team_matches.merge(
        players[["player_id", "team_id", "our_player_id"]], on="team_id")
    out = roster.merge(
        lineups[["match_id", "player_id", "is_starting_xi",
                 "minutes_played"]],
        on=["match_id", "player_id"], how="left")
    out["is_starting_xi"] = out["is_starting_xi"].fillna(0).astype(int)
    out["minutes_played"] = out["minutes_played"].fillna(0).astype(float)
    return out


def player_start_features(window: int = FORM_WINDOW,
                          raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Leak-free trailing start rate and minutes share per (player, round).

    real_start_rate_lag: fraction of the team's last `window` completed
    matches (rounds < k) the player started. minutes_share_lag: mean
    minutes/90 over the same matches. Only players matched to our ids.
    """
    if load("match_lineups").empty:
        return pd.DataFrame(columns=["player_id", "round_id",
                                     "real_start_rate_lag",
                                     "minutes_share_lag"])
    lu = _matched_lineups(raw_dir=raw_dir)
    lu = lu[lu["our_player_id"].notna()]
    all_rounds = sorted(_latest(raw_dir, "fixtures")["round_id"].unique())
    rows = []
    for pid, grp in lu.groupby("our_player_id"):
        grp = grp.sort_values("round_id")
        for k in all_rounds:
            prior = grp[grp["round_id"] < k].tail(window)
            if prior.empty:
                continue
            rows.append({
                "player_id": int(pid),
                "round_id": int(k),
                "real_start_rate_lag": float(prior["is_starting_xi"].mean()),
                "minutes_share_lag": float(
                    (prior["minutes_played"] / 90.0).mean()),
            })
    return pd.DataFrame(rows)


def validate(out_path: Path | None = DEFAULT_VALIDATION_OUT,
             raw_dir: Path = DEFAULT_RAW_DIR) -> dict:
    """Cross-check the dataset against our collector data; write JSON."""
    import json
    from datetime import datetime, timezone

    spine = match_spine(raw_dir=raw_dir)
    done = spine[spine["completed"]]
    ours_done = done[done["status_ours"] == "complete"]
    score_rows = ours_done[ours_done["home_score_ours"].notna()]
    mismatches = score_rows[
        (score_rows["home_score_ext"] != score_rows["home_score_ours"])
        | (score_rows["away_score_ext"] != score_rows["away_score_ours"])]

    fixtures = _latest(raw_dir, "fixtures")
    lineups = load("match_lineups")
    tmap = team_id_map(raw_dir=raw_dir)
    players = load("squads_and_players")
    matcher = PlayerNameMatcher.from_latest(raw_dir)
    match_rates = {}
    for abbr in ("FRA", "ENG", "ESP", "ARG"):
        team_id = int(tmap.loc[tmap["abbr"] == abbr, "team_id"].iloc[0])
        sub = players[players["team_id"] == team_id].copy()
        sub["abbr"] = abbr
        matched = match_player_ids(sub, matcher)
        match_rates[abbr] = round(
            float(matched.notna().mean()) if len(sub) else 0.0, 3)

    ext_completed = int(spine["completed"].sum())
    ours_completed = int((fixtures["status"] == "complete").sum())
    xg_covered = int(done["home_xg"].notna().sum())
    lineup_max = int(lineups["match_id"].max()) if len(lineups) else 0
    ext_max_done = int(done["match_id"].max()) if len(done) else 0

    fresh = (ext_completed >= ours_completed
             and lineup_max >= ext_max_done
             and xg_covered == ext_completed)
    status = "ok" if fresh and len(mismatches) <= 2 else (
        "score_mismatch" if len(mismatches) > 2 else "stale")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {"base_url": BASE_URL,
                   "raw_dir": str(raw_dir),
                   "files": list(FILES)},
        "status": status,
        "scores_checked": int(len(score_rows)),
        "score_mismatches": mismatches[
            ["match_id", "home_abbr", "away_abbr", "home_score_ext",
             "away_score_ext", "home_score_ours", "away_score_ours"]
        ].to_dict("records"),
        "ext_completed": ext_completed,
        "ours_completed": ours_completed,
        "xg_covered": xg_covered,
        "lineups_through_match": lineup_max,
        "unmapped_teams": int(spine["fixture_id"].isna().sum()),
        "player_match_rates": match_rates,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=1))
    if len(mismatches) > 2:
        raise SystemExit(
            f"wc2026_dataset: {len(mismatches)} score mismatches vs our "
            f"fixtures; refusing to use the dataset (see {out_path})")
    return payload
