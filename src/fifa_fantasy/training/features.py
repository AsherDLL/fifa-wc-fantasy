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

import pandas as pd

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
    df["target"] = df["total_points"].astype(int)
    return df.reset_index(drop=True)
