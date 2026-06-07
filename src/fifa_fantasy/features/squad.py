"""Per-squad strength features derived from the player pool prices.

Pricing already encodes market expectations of expected points, so squad
strength can be approximated from price aggregates without any match data
— useful before any games have been played.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_TOP_N = 11


def squad_strength(
    players: pd.DataFrame,
    squads: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
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

    return squads.merge(agg.reset_index(), on="squad_id", how="left")
