"""Heuristic baseline predictor.

Goal: produce a sensible `predicted_points` per (player, round) row so the
optimizer has something to consume before any matches are played. Formula:

    base    = points_per_price_unit[position] * price_millions
    matchup = 1 + alpha * tanh(combined_diff)
    home    = 1 + beta * is_home
    premium = premium_boost * max(0, price_millions - PREMIUM_PRICE_THRESHOLD)

    predicted_points = base * matchup * home + premium

`combined_diff` blends two signals:
    z_price = strength_diff / STRENGTH_DIFF_SCALE        (squad top-11 price gap)
    z_rank  = rank_diff     / RANK_DIFF_SCALE            (FIFA world ranking gap)
    combined_diff = price_weight * z_price + rank_weight * z_rank

If `rank_diff` is missing (no FIFA ranking row for the country), the rank
weight is dropped for that row and the price signal carries the full
matchup adjustment. Default weights give the FIFA ranking the heavier
share since national-team form tracks national-team output better than
club-league price does.

Zeroed for any player whose `status` is not "playing" or whose squad is
eliminated. `premium_boost = 0` (default) preserves the original behaviour;
positive values tilt the optimizer toward premium-priced players.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fifa_fantasy.scoring import Position

# Position-specific multiple of price. Tuned so a 6.0M MID predicts ~3.6 pts
# (close to a typical FPL-style starter floor) and a 10.0M FWD ~6.5 pts.
POINTS_PER_PRICE_UNIT: dict[Position, float] = {
    Position.GK: 0.50,
    Position.DEF: 0.55,
    Position.MID: 0.60,
    Position.FWD: 0.65,
}

# Strength-diff is the difference of top-11 average prices (~ ±3 in practice).
# tanh(x / 2) saturates by |x|=5 so the matchup adjustment caps at ±alpha.
# Strength signals. The price-based `strength_diff` ranges roughly +/-3 across
# the field; the FIFA ranking `rank_diff` is in ranking points (roughly
# +/-700 across the field). Scales normalize both to a comparable z-like range
# before blending.
STRENGTH_DIFF_SCALE = 2.0
RANK_DIFF_SCALE = 250.0

# Blend weights for the two strength signals. Default tilts toward the FIFA
# ranking since it tracks national-team form, where the price proxy tracks
# club-league quality. Both signals normalize to roughly +/-1 at saturation.
PRICE_SIGNAL_WEIGHT = 0.35
RANK_SIGNAL_WEIGHT = 0.65

# Matchup saturation. Bumped to +/-40% so a top-vs-bottom matchup carries real
# weight; the previous 25% was too conservative.
STRENGTH_DIFF_ALPHA = 0.40
HOME_ADVANTAGE_BETA = 0.05  # +5% for the home side

# Premium-tier knob (off by default). Adds `premium_boost * max(0, price - threshold)`
# extra points so the optimizer doesn't always prefer mid-priced players whose
# per-$M return is artificially flat in the linear-in-price base term.
PREMIUM_PRICE_THRESHOLD = 9.0
DEFAULT_PREMIUM_BOOST = 0.0

PLAYING_STATUS = "playing"


def _position_coef(value: object) -> float:
    if isinstance(value, Position):
        return POINTS_PER_PRICE_UNIT[value]
    return POINTS_PER_PRICE_UNIT[Position(value)]


def _combined_matchup_z(features: pd.DataFrame) -> np.ndarray:
    """Blend the price-based and ranking-based strength signals.

    NaN-safe for `rank_diff`: rows without a FIFA ranking get the full
    matchup weight on the price signal so they degrade gracefully.
    """
    z_price = features["strength_diff"].astype(float).to_numpy() / STRENGTH_DIFF_SCALE
    rank_raw = pd.to_numeric(features.get("rank_diff"), errors="coerce") \
        if "rank_diff" in features.columns else pd.Series(
            [pd.NA] * len(features), dtype="Float64"
        )
    z_rank = rank_raw.to_numpy(dtype=float) / RANK_DIFF_SCALE  # NaN preserved
    has_rank = ~pd.isna(rank_raw).to_numpy()

    blended = np.where(
        has_rank,
        PRICE_SIGNAL_WEIGHT * z_price + RANK_SIGNAL_WEIGHT * z_rank,
        z_price,  # fall back to full-weight price when rank is missing
    )
    return blended


def heuristic_predict(
    features: pd.DataFrame,
    premium_boost: float = DEFAULT_PREMIUM_BOOST,
) -> pd.DataFrame:
    """Add a `predicted_points` column to a copy of `features`.

    Players whose `status` != "playing" or whose squad is eliminated are
    predicted at 0. Otherwise the module-docstring formula applies.

    `premium_boost` adds `premium_boost * max(0, price - 9.0)` to the
    prediction. Defaults to 0.0 (preserves prior behaviour); positive
    values tilt the optimizer toward 9M+ players.
    """
    out = features.copy()

    coef = out["position"].map(_position_coef).astype(float)
    price = out["price_millions"].astype(float)
    base = coef * price

    combined_z = _combined_matchup_z(out)
    matchup = 1.0 + STRENGTH_DIFF_ALPHA * np.tanh(combined_z)
    home = 1.0 + HOME_ADVANTAGE_BETA * out["is_home"].astype(int)

    raw = base * matchup * home
    if premium_boost:
        raw = raw + premium_boost * np.maximum(0.0, price - PREMIUM_PRICE_THRESHOLD)

    available = (out["status"] == PLAYING_STATUS) & (~out["is_eliminated"].astype(bool))
    out["predicted_points"] = np.where(available, raw, 0.0)
    return out
