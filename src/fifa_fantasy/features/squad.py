"""Per-squad strength features.

Two independent signals are computed and stored side by side so the
downstream consumer can pick or blend them:

1. `squad_top_n_avg_price`: mean price of the squad's top-N most expensive
   players in the FIFA Fantasy pool. Encodes the game's own pricing model,
   which roughly tracks club-league quality.
2. `rank_points` and `rank_position`: FIFA Men's World Ranking (loaded
   from `data/static/fifa_rankings.csv`). Tracks recent national-team
   results, which the price proxy does not.

When the rankings file is missing or a country has no entry, the rank
columns are left as NaN and the heuristic falls back to the price-only
signal.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_TOP_N = 11


def squad_strength(
    players: pd.DataFrame,
    squads: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    rankings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per squad with strength proxies.

    Columns added on top of the squad table:
        squad_total_price, squad_avg_price,
        squad_top_n_avg_price, squad_top_n_rank,
        squad_size

    `squad_top_n_avg_price` is the mean price of the squad's `top_n` most
    expensive players — a closer proxy for starting-XI quality than the
    mean across the whole roster. Rank 1 is the strongest squad.
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    by_squad = players.groupby("squad_id", sort=False)
    agg = pd.DataFrame(
        {
            "squad_total_price": by_squad["price_millions"].sum(),
            "squad_avg_price": by_squad["price_millions"].mean(),
            "squad_size": by_squad["player_id"].count().astype("int64"),
        }
    )

    top_n_avg = (
        players.sort_values("price_millions", ascending=False)
        .groupby("squad_id", sort=False)
        .head(top_n)
        .groupby("squad_id", sort=False)["price_millions"]
        .mean()
        .rename("squad_top_n_avg_price")
    )
    agg = agg.join(top_n_avg)

    agg["squad_top_n_rank"] = (
        agg["squad_top_n_avg_price"].rank(ascending=False, method="min").astype("int64")
    )

    out = squads.merge(agg.reset_index(), on="squad_id", how="left")

    if rankings is not None and not rankings.empty:
        out = out.merge(
            rankings.rename(columns={"rank_points": "squad_rank_points",
                                     "rank_position": "squad_rank_position"}),
            left_on="name", right_on="country", how="left",
        ).drop(columns=["country"])
    else:
        out["squad_rank_points"] = pd.NA
        out["squad_rank_position"] = pd.NA

    return out
