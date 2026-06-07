"""Apply scouting bonus + aggregate per-(player, round) predictions to per-player.

The scouting bonus rule (Fantasy.md): +2 pts if a player scores **more than**
4 pts in a match AND is owned by **fewer than** 5% of teams. We treat the
heuristic point estimate as the deterministic match outcome — a coarse
approximation that Phase 3b's quantile regression will refine.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from fifa_fantasy.scoring import (
    SCOUTING_BONUS,
    SCOUTING_BONUS_OWNERSHIP_THRESHOLD,
    SCOUTING_BONUS_POINTS_THRESHOLD,
)


def apply_scouting_bonus(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add `scouting_bonus` and `effective_points` columns to a copy.

    Reuses the canonical thresholds from `scoring.py` so the bonus rule is
    encoded in exactly one place.
    """
    out = predictions.copy()
    triggers = (out["predicted_points"] > SCOUTING_BONUS_POINTS_THRESHOLD) & (
        out["ownership_fraction"] < SCOUTING_BONUS_OWNERSHIP_THRESHOLD
    )
    out["scouting_bonus"] = triggers.astype(int) * SCOUTING_BONUS
    out["effective_points"] = out["predicted_points"] + out["scouting_bonus"]
    return out


def aggregate_to_player(
    predictions: pd.DataFrame,
    rounds: Iterable[int],
) -> pd.DataFrame:
    """Sum effective_points across the given rounds; one row per player.

    Keeps the player-level metadata (name, position, country, price, etc.)
    from the underlying rows.
    """
    rounds = list(rounds)
    in_scope = predictions[predictions["round_id"].isin(rounds)]
    sums = (
        in_scope.groupby("player_id", sort=False)
        .agg(
            total_effective_points=("effective_points", "sum"),
            total_predicted_points=("predicted_points", "sum"),
        )
        .reset_index()
    )
    # Take metadata from any row of the player (it's constant per snapshot).
    meta_cols = [
        "player_id",
        "full_name",
        "position",
        "country",
        "country_abbr",
        "squad_id",
        "price_millions",
        "ownership_fraction",
        "status",
        "is_eliminated",
    ]
    meta = in_scope[meta_cols].drop_duplicates("player_id")
    return meta.merge(sums, on="player_id", how="inner")
