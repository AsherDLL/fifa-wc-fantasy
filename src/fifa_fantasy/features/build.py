"""Build the per-(player, round) feature table.

Output schema (column groups):
    - Player columns: player_id, full_name, position, country, ...
    - Fixture columns: fixture_id, round_id, stage, is_home, kickoff,
      opponent_squad_id, opponent_name, opponent_abbr, venue_*
    - Squad-strength columns (player's own): squad_total_price,
      squad_avg_price, squad_top_n_avg_price, squad_top_n_rank, squad_size
    - Opponent-strength columns (prefixed `opp_`): opp_squad_total_price,
      opp_squad_avg_price, opp_squad_top_n_avg_price, opp_squad_top_n_rank
    - Derived: strength_diff (own top-n minus opp top-n),
      days_since_prev_match, days_to_next_match
"""

from __future__ import annotations

import pandas as pd


STRENGTH_COLUMNS = [
    "squad_total_price",
    "squad_avg_price",
    "squad_top_n_avg_price",
    "squad_top_n_rank",
    "squad_rank_points",
    "squad_rank_position",
]


def flatten_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Wide-to-long: emit two rows per fixture, one for each side.

    Output columns:
        squad_id, opponent_squad_id, opponent_name, opponent_abbr,
        is_home, fixture_id, round_id, stage, kickoff,
        venue_name, venue_city, status
    """
    # `status` lives on both players and fixtures (per-player availability vs
    # per-match scheduled/finished). Rename fixture's to disambiguate before
    # any downstream merge.
    common_cols = [
        "fixture_id",
        "round_id",
        "stage",
        "kickoff",
        "venue_name",
        "venue_city",
    ]
    fixtures = fixtures.rename(columns={"status": "fixture_status"})
    common_cols = common_cols + ["fixture_status"]

    home = fixtures[
        common_cols
        + ["home_squad_id", "away_squad_id", "away_squad_name", "away_squad_abbr"]
    ].rename(
        columns={
            "home_squad_id": "squad_id",
            "away_squad_id": "opponent_squad_id",
            "away_squad_name": "opponent_name",
            "away_squad_abbr": "opponent_abbr",
        }
    )
    home["is_home"] = True

    away = fixtures[
        common_cols
        + ["away_squad_id", "home_squad_id", "home_squad_name", "home_squad_abbr"]
    ].rename(
        columns={
            "away_squad_id": "squad_id",
            "home_squad_id": "opponent_squad_id",
            "home_squad_name": "opponent_name",
            "home_squad_abbr": "opponent_abbr",
        }
    )
    away["is_home"] = False

    return pd.concat([home, away], ignore_index=True)


def _attach_rest_days(grid: pd.DataFrame) -> pd.DataFrame:
    """Compute days_since_prev_match and days_to_next_match per (squad, round).

    Rest days are a squad-level property (every player on the same squad in
    the same round shares them), so we compute them once on the deduped
    (squad_id, round_id, kickoff) table and merge back.
    """
    schedule = (
        grid[["squad_id", "round_id", "kickoff"]]
        .drop_duplicates()
        .sort_values(["squad_id", "kickoff"])
        .reset_index(drop=True)
    )
    grouped = schedule.groupby("squad_id", sort=False)["kickoff"]
    schedule["days_since_prev_match"] = grouped.diff().dt.total_seconds() / 86400.0
    schedule["days_to_next_match"] = (
        grouped.diff(-1).mul(-1).dt.total_seconds() / 86400.0
    )
    return grid.merge(
        schedule[["squad_id", "round_id", "days_since_prev_match", "days_to_next_match"]],
        on=["squad_id", "round_id"],
        how="left",
    )


def _attach_opponent_strength(
    grid: pd.DataFrame, squad_strength: pd.DataFrame
) -> pd.DataFrame:
    opp = squad_strength[["squad_id"] + STRENGTH_COLUMNS].rename(
        columns={"squad_id": "opponent_squad_id", **{c: f"opp_{c}" for c in STRENGTH_COLUMNS}}
    )
    return grid.merge(opp, on="opponent_squad_id", how="left")


def _attach_country_elo(grid: pd.DataFrame, country_elo: pd.DataFrame) -> pd.DataFrame:
    """Join the international Elo snapshot onto both own and opponent country.

    `country_elo` is whatever `external.international_elo.load()` returned,
    with `country_name` translated to the FIFA Fantasy `country` field via
    `external.mapping.to_fifa_country`. An empty frame leaves all elo
    columns NaN, which the heuristic and GBM both tolerate.
    """
    if country_elo is None or country_elo.empty:
        for c in ("country_elo", "opp_country_elo", "country_elo_diff",
                  "country_last10_form", "opp_country_last10_form"):
            grid[c] = pd.NA
        return grid
    snap = country_elo[["country_name", "elo", "last10_form"]].copy()
    snap = snap.rename(columns={
        "country_name": "country",
        "elo": "country_elo",
        "last10_form": "country_last10_form",
    })
    grid = grid.merge(snap, on="country", how="left")
    opp = snap.rename(columns={
        "country": "opponent_name",
        "country_elo": "opp_country_elo",
        "country_last10_form": "opp_country_last10_form",
    })
    grid = grid.merge(opp, on="opponent_name", how="left")
    grid["country_elo_diff"] = (
        pd.to_numeric(grid["country_elo"], errors="coerce")
        - pd.to_numeric(grid["opp_country_elo"], errors="coerce")
    )
    return grid


def team_gc_history(fixtures: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Per-(squad_id, round_id) trailing mean goals conceded, leak-free.

    For each completed fixture a team's goals conceded is the opponent's
    score. For an upcoming round we want the mean over the previous
    `window` completed rounds, strictly before the round in question, so
    the shift(1) excludes the current round. Returns columns
    [squad_id, round_id, team_gc_form]. Rounds are 1-based; a team's
    first round gets NaN.

    Shared by WC inference (features.build) and WC label extraction
    (training.wc) so `team_gc_form` has one definition everywhere.
    """
    fx = fixtures.copy()
    for col in ("home_score", "away_score"):
        fx[col] = pd.to_numeric(fx.get(col), errors="coerce")
    completed = fx[fx["home_score"].notna() & fx["away_score"].notna()]
    rows = []
    for _, r in completed.iterrows():
        rows.append({"squad_id": int(r["home_squad_id"]), "round_id": int(r["round_id"]),
                     "gc": float(r["away_score"])})
        rows.append({"squad_id": int(r["away_squad_id"]), "round_id": int(r["round_id"]),
                     "gc": float(r["home_score"])})
    if not rows:
        return pd.DataFrame(columns=["squad_id", "round_id", "team_gc_form"])
    gc = pd.DataFrame(rows).sort_values(["squad_id", "round_id"]).reset_index(drop=True)
    g = gc.groupby("squad_id", sort=False)["gc"]
    gc["team_gc_form"] = g.transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    return gc[["squad_id", "round_id", "team_gc_form"]]


def completed_rounds_by_squad(fixtures: pd.DataFrame) -> dict[int, int]:
    """Number of completed rounds per squad_id, from fixture scores.

    The FIFA API truncates a player's `round_points` at the last round the
    player personally recorded anything, silently dropping trailing DNPs.
    The squad's completed-fixture count is the true number of rounds the
    player was available to appear in; the gap between the two is exactly
    the trailing zeros the API swallowed.
    """
    fx = fixtures.copy()
    for col in ("home_score", "away_score"):
        fx[col] = pd.to_numeric(fx.get(col), errors="coerce")
    completed = fx[fx["home_score"].notna() & fx["away_score"].notna()]
    counts: dict[int, int] = {}
    for _, r in completed.iterrows():
        for side in ("home_squad_id", "away_squad_id"):
            sid = int(r[side])
            counts[sid] = counts.get(sid, 0) + 1
    return counts


def pad_round_points(rp, n_completed: int | None) -> list:
    """Reconstruct the full per-round history from a truncated API list.

    Trailing DNP rounds are missing from `round_points` (interior DNPs are
    zero-filled by the API, trailing ones truncated). Padding with zeros up
    to the squad's completed-round count restores them, so a player benched
    for the last two rounds no longer shows the form and participation of
    his last two matches PLAYED. Without this, form_lag is inflated and the
    availability discount never fires for recently-dropped players, the
    exact rotation risks it exists to catch.
    """
    vals = [float(v) for v in list(rp)] if rp is not None else []
    if n_completed is not None and len(vals) < n_completed:
        vals = vals + [0.0] * (n_completed - len(vals))
    return vals


def _attach_padded_round_points(grid: pd.DataFrame,
                                fixtures: pd.DataFrame) -> pd.DataFrame:
    """Add `_rp_padded`: round_points with trailing DNPs restored."""
    if "round_points" not in grid.columns:
        return grid
    counts = completed_rounds_by_squad(fixtures)
    grid["_rp_padded"] = [
        pad_round_points(rp, counts.get(int(sid)) if pd.notna(sid) else None)
        for rp, sid in zip(grid["round_points"], grid["squad_id"])
    ]
    return grid


def _attach_team_gc_form(grid: pd.DataFrame, fixtures: pd.DataFrame,
                         window: int = 3) -> pd.DataFrame:
    """Attach team_gc_form to the upcoming-round grid.

    Each upcoming (squad, round) gets the mean goals conceded over the
    team's LAST `window` completed rounds, computed fresh from the fixture
    scores. The previous implementation broadcast the team's latest stored
    trailing value from `team_gc_history`, but that value is the form AT
    the last completed round (it excludes that round's own concession by
    design), so inference served a window stale by one match relative to
    the training-side lookup.
    """
    fx = fixtures.copy()
    for col in ("home_score", "away_score"):
        fx[col] = pd.to_numeric(fx.get(col), errors="coerce")
    completed = fx[fx["home_score"].notna() & fx["away_score"].notna()]
    if completed.empty:
        grid["team_gc_form"] = pd.NA
        return grid
    rows = []
    for _, r in completed.iterrows():
        rows.append((int(r["home_squad_id"]), int(r["round_id"]), float(r["away_score"])))
        rows.append((int(r["away_squad_id"]), int(r["round_id"]), float(r["home_score"])))
    gc = pd.DataFrame(rows, columns=["squad_id", "round_id", "gc"])
    gc = gc.sort_values(["squad_id", "round_id"])
    current = (gc.groupby("squad_id")["gc"]
               .apply(lambda s: float(s.tail(window).mean()))
               .rename("team_gc_form").reset_index())
    return grid.merge(current, on="squad_id", how="left")


def _attach_form_lag(grid: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Attach `form_lag`: trailing mean of the player's realised points over
    the previous `window` completed rounds.

    The `round_points` list holds one integer per completed round. For an
    upcoming fixture we predict, all of those are strictly in the past, so
    the mean of the last `window` of them is the recency signal for the next
    round. This matches training/features.add_lagged_form (EPL) and
    training/wc.extract_wc_training_rows (WC labels), so the GBM sees one
    feature with one meaning. NaN before a player has any completed round
    (e.g. pre-MD1), which the GBM handles natively.

    Reads `_rp_padded` (trailing DNPs restored) when present, so a benched
    player's missed rounds count as zeros instead of being skipped.
    """
    def last_k_mean(rp) -> float:
        if rp is None:
            return float("nan")
        vals = [v for v in list(rp)]
        if not vals:
            return float("nan")
        w = vals[-window:]
        return float(sum(w)) / len(w)

    src = "_rp_padded" if "_rp_padded" in grid.columns else "round_points"
    if src in grid.columns:
        grid["form_lag"] = grid[src].map(last_k_mean)
    else:
        grid["form_lag"] = pd.NA
    return grid


def _attach_start_rate_lag(grid: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Attach `start_rate_lag`: trailing participation rate over completed rounds.

    The FIFA API does not expose per-round minutes, so participation is
    proxied by round_points > 0 (a player who took the pitch banks at least
    the appearance points). Mean over the last `window` completed rounds.
    Matches the WC training-side definition in training/wc.py. NaN before a
    player has any completed round.

    Reads `_rp_padded` (trailing DNPs restored) when present: a player
    benched for the last two rounds is a 1/3 participation, not the 3/3
    his truncated played-matches list implied.
    """
    def part_rate(rp) -> float:
        if rp is None:
            return float("nan")
        vals = [1.0 if v > 0 else 0.0 for v in list(rp)]
        if not vals:
            return float("nan")
        w = vals[-window:]
        return float(sum(w)) / len(w)

    src = "_rp_padded" if "_rp_padded" in grid.columns else "round_points"
    if src in grid.columns:
        grid["start_rate_lag"] = grid[src].map(part_rate)
    else:
        grid["start_rate_lag"] = pd.NA
    return grid


def _attach_team_news(grid: pd.DataFrame,
                     news_table: pd.DataFrame | None) -> pd.DataFrame:
    """Join the latest scraped predicted-XI signal into the per-row grid.

    Adds two columns:
        predicted_starting_xi : bool | NaN
            True if the latest news has the player as a starter for this
            fixture, False if benched, NaN if no news available.
        xi_confidence : float | NaN
            Per-source confidence in [0, 1].

    NaN-safe: when no news exists or the join misses, downstream models
    fall back to their current behaviour.
    """
    if news_table is None or news_table.empty:
        grid["predicted_starting_xi"] = pd.NA
        grid["xi_confidence"] = pd.NA
        return grid
    # Latest record per (fixture_id, player_id).
    latest = (news_table.sort_values("scraped_at_utc")
              .groupby(["fixture_id", "player_id"]).tail(1).copy())
    latest["predicted_starting_xi"] = latest["status"] == "starting"
    latest = latest.rename(columns={"source_confidence": "xi_confidence"})
    return grid.merge(
        latest[["fixture_id", "player_id", "predicted_starting_xi", "xi_confidence"]],
        on=["fixture_id", "player_id"], how="left",
    )


def build_player_round_features(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    squad_strength: pd.DataFrame,
    country_elo: pd.DataFrame | None = None,
    team_news: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per (player, upcoming round in which their squad plays).

    Inner-joins players to fixtures, so eliminated or rest-day squads have
    no rows for the rounds they're sitting out.

    `country_elo` is the optional output of
    `external.international_elo.load()` after country-name normalization.
    When supplied, the grid gets `country_elo`, `opp_country_elo`,
    `country_elo_diff`, and the last-10 form columns. None / empty is fine.
    """
    fix_long = flatten_fixtures(fixtures)

    # Player's own squad strength.
    own_strength = squad_strength[["squad_id"] + STRENGTH_COLUMNS]
    enriched_players = players.merge(own_strength, on="squad_id", how="left")

    grid = enriched_players.merge(fix_long, on="squad_id", how="inner")
    grid = _attach_opponent_strength(grid, squad_strength)
    grid["strength_diff"] = (
        grid["squad_top_n_avg_price"] - grid["opp_squad_top_n_avg_price"]
    )
    # FIFA ranking signal: NaN-tolerant subtract so missing rankings fall back
    # naturally to the price-based strength_diff in the heuristic.
    grid["rank_diff"] = (
        pd.to_numeric(grid["squad_rank_points"], errors="coerce")
        - pd.to_numeric(grid["opp_squad_rank_points"], errors="coerce")
    )
    grid = _attach_country_elo(grid, country_elo)
    grid = _attach_team_news(grid, team_news)
    grid = _attach_padded_round_points(grid, fixtures)
    grid = _attach_form_lag(grid)
    grid = _attach_start_rate_lag(grid)
    grid = _attach_team_gc_form(grid, fixtures)
    grid = _attach_rest_days(grid)
    if "_rp_padded" in grid.columns:
        grid = grid.drop(columns=["_rp_padded"])
    return grid.reset_index(drop=True)
