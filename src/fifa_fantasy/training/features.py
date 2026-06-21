"""Engineer training features compatible with the WC inference table.

At inference (WC), the per-(player, round) feature table has these
LightGBM-relevant columns:

    position, price_millions, is_home,
    strength_diff, rank_diff,
    squad_top_n_avg_price, opp_squad_top_n_avg_price,
    squad_rank_points,     opp_squad_rank_points,
    ownership_fraction

For training (EPL), we synthesize matching columns:

    strength_diff  = own_team_strength - opp_team_strength    (FPL strength 1-5)
    rank_diff      = None (no FIFA ranking for clubs; LightGBM treats as NaN)
    squad_*_price  = team-aggregated FPL prices per gameweek
    opp_*          = same, on opponent
    ownership_fraction = computed from the FPL selected count later if we need it

Target is `total_points` per gameweek.

The intent is not perfect parity, only structural parity: the model
learns shape from features that have the same meaning across data sets.
A row is dropped if the player played 0 minutes that gameweek (no
useful signal for fantasy-points prediction beyond DNP).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fifa_fantasy.external.football_data import (
    compute_club_elo_history, load_matches,
)
from fifa_fantasy.external.mapping import to_fd_club

TOP_N = 11
FEATURE_COLUMNS = [
    "position",
    "price_millions",
    "is_home",
    "strength_diff",
    "squad_top_n_avg_price",
    "opp_squad_top_n_avg_price",
    # rank_diff is included at inference but unavailable for EPL training;
    # we let LightGBM treat it as NaN. Same column name keeps inference
    # code unchanged.
    # team_elo_diff: club-Elo gap (training) or country-Elo gap (inference).
    # Both derived from football-data.co.uk and martj42 respectively; same
    # scale (Elo ~400 per 10:1 odds) so the GBM treats them uniformly.
    "team_elo_diff",
]


def _team_top_n_avg_price(per_team_gw: pd.DataFrame, top_n: int = TOP_N) -> pd.Series:
    return (
        per_team_gw.sort_values("price_millions", ascending=False)
        .groupby(["team_id", "gameweek"], sort=False)
        .head(top_n)
        .groupby(["team_id", "gameweek"], sort=False)["price_millions"]
        .mean()
        .rename("top_n_avg_price")
    )


def build_training_table(player_gameweek: pd.DataFrame) -> pd.DataFrame:
    """Augment the scraped FPL table with feature columns matching inference."""
    df = player_gameweek.copy()
    df = df[df["minutes"] > 0].copy()  # drop DNPs; same as inference filter

    # Per-(team, gameweek) top-N avg price.
    top = _team_top_n_avg_price(df).reset_index()
    own = top.rename(columns={"top_n_avg_price": "squad_top_n_avg_price"})
    df = df.merge(own, on=["team_id", "gameweek"], how="left")

    opp = top.rename(columns={
        "team_id": "opponent_team_id",
        "top_n_avg_price": "opp_squad_top_n_avg_price",
    })
    df = df.merge(opp, on=["opponent_team_id", "gameweek"], how="left")

    df["strength_diff"] = (
        df["squad_top_n_avg_price"] - df["opp_squad_top_n_avg_price"]
    )

    # Ownership: vaastav 2022-23 / 2023-24 dumps lack ownership; the
    # live-API 2024-25 file may have it as a pre-divided fraction. Anything
    # missing gets a neutral 0.10. EPL absolute ownership is in the same
    # ballpark as WC absolute ownership, so this is reasonable.
    if "ownership_fraction" in df.columns:
        df["ownership_fraction"] = pd.to_numeric(
            df["ownership_fraction"], errors="coerce"
        ).fillna(0.10)
    else:
        df["ownership_fraction"] = 0.10

    df = _attach_team_elo_diff(df)

    df["target"] = df["total_points"].astype(int)
    return df.reset_index(drop=True)


def _attach_team_elo_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Add `team_elo_diff` (own club Elo - opponent club Elo) at match time.

    Uses football-data.co.uk match history to roll Elo forward and joins
    each (club, kickoff) row to the latest pre-match snapshot. Missing
    data leaves NaN; LightGBM tolerates that, and on the inference side
    the same column gets populated from country_elo_diff so the GBM sees
    one feature.
    """
    matches = load_matches()
    if matches.empty or "team_name" not in df.columns or "opponent_team_name" not in df.columns:
        df["team_elo_diff"] = pd.NA
        return df
    history = compute_club_elo_history(matches)
    # Map FPL team names to football-data club names.
    df = df.copy()
    df["_fd_club"] = df["team_name"].map(to_fd_club)
    df["_fd_opp"] = df["opponent_team_name"].map(to_fd_club)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
    # Strip tz on both sides for asof merge.
    df["_kickoff_naive"] = df["kickoff_time"].dt.tz_localize(None)

    history = history.sort_values(["club", "date"]).reset_index(drop=True)
    history["date"] = pd.to_datetime(history["date"])

    hist_sorted = (
        history[["club", "date", "elo_before"]]
        .rename(columns={"date": "_kickoff_naive"})
        .sort_values("_kickoff_naive")
        .reset_index(drop=True)
    )

    def lookup(side_col: str, out_col: str) -> pd.Series:
        merge_left = (
            df[["_kickoff_naive", side_col]]
            .rename(columns={side_col: "club"})
            .sort_values("_kickoff_naive")
            .reset_index()
        )
        merged = pd.merge_asof(
            merge_left, hist_sorted,
            on="_kickoff_naive", by="club", direction="backward",
        )
        merged = merged.sort_values("index").set_index("index")
        return merged["elo_before"].rename(out_col)

    df["_own_elo"] = lookup("_fd_club", "_own_elo").reindex(df.index)
    df["_opp_elo"] = lookup("_fd_opp", "_opp_elo").reindex(df.index)
    df["team_elo_diff"] = pd.to_numeric(df["_own_elo"], errors="coerce") - \
                          pd.to_numeric(df["_opp_elo"], errors="coerce")
    df = df.drop(columns=["_fd_club", "_fd_opp", "_kickoff_naive", "_own_elo", "_opp_elo"])
    return df
