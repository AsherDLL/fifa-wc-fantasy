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


def build_player_round_features(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    squad_strength: pd.DataFrame,
    country_elo: pd.DataFrame | None = None,
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
    grid = _attach_rest_days(grid)
    return grid.reset_index(drop=True)
