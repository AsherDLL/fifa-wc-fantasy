"""Apply scouting bonus + aggregate per-(player, round) predictions to per-player.

The scouting bonus rule (Fantasy.md): +2 pts if a player scores **more than**
4 pts in a match AND is owned by **fewer than** 5% of teams. We treat the
heuristic point estimate as the deterministic match outcome, a coarse
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

# Availability discount. A player who has taken the pitch in only a
# fraction of their team's recent matches is a rotation risk, and a
# benched starter scores near zero regardless of predicted points. We
# discount effective points toward a floor by the player's recent
# participation rate (`start_rate_lag`, in [0, 1]).
#
# The floor is deliberately not zero: the WC participation signal is a
# proxy (round_points > 0), so a nailed starter who blanked twice looks
# like a rotation risk and we must not zero them out. Validated on the
# clean heuristic backend across MD1-R32: discounting lifted realised
# squad points from 275 to 283 and the result was stable for any floor in
# [0.3, 0.7]. See scripts/wc_forward_validation.py context and docs 11g.
AVAILABILITY_FLOOR = 0.5


def availability_factor(start_rate_lag: float,
                        floor: float = AVAILABILITY_FLOOR) -> float:
    """Map a participation rate in [0, 1] to a multiplier in [floor, 1].

    NaN (no completed rounds yet, e.g. pre-MD1) returns 1.0: with no
    history we make no availability claim.
    """
    if start_rate_lag is None or pd.isna(start_rate_lag):
        return 1.0
    return floor + (1.0 - floor) * float(start_rate_lag)


def apply_availability_discount(predictions: pd.DataFrame,
                                floor: float = AVAILABILITY_FLOOR) -> pd.DataFrame:
    """Scale `effective_points` by each player's availability factor.

    Adds `availability_factor` and `rotation_risk` columns for auditing and
    for the report to surface. Requires `effective_points` (run
    apply_scouting_bonus first) and tolerates a missing `start_rate_lag`
    column (treated as no-history, factor 1.0).

    NaN policy is context-aware. Pre-tournament every player is NaN (no
    completed rounds), and NaN correctly means "unknown", factor 1.0. But
    once ANY player in the frame has participation history, a NaN means
    "has never taken the pitch this tournament", which is the strongest
    bench signal available, so it is treated as participation 0.0 (the
    floor). Without this, the solver starts never-played backup keepers:
    the Poisson backend gives every GK of a team identical team-level
    clean-sheet points, the backup is always cheaper, and at factor 1.0
    the backup wins the slot (observed: a 0.9%-owned third keeper picked
    to start over the 13%-owned ever-present number one).
    """
    out = predictions.copy()
    if "effective_points" not in out.columns:
        raise ValueError("run apply_scouting_bonus before apply_availability_discount")
    if "start_rate_lag" not in out.columns:
        out["availability_factor"] = 1.0
        out["rotation_risk"] = False
        return out
    tournament_underway = out["start_rate_lag"].notna().any()
    rate = out["start_rate_lag"]
    if tournament_underway:
        rate = rate.fillna(0.0)
    out["availability_factor"] = rate.map(
        lambda s: availability_factor(s, floor)
    )
    out["rotation_risk"] = out["availability_factor"] < 1.0
    out["effective_points"] = out["effective_points"] * out["availability_factor"]
    return out


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
