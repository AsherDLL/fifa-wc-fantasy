"""Heuristic baseline predictor — no training, no labels.

Goal: produce a sensible `predicted_points` per (player, round) row so the
optimizer (Phase 4) has something to consume before the LightGBM models in
Phase 3b are trained. The formula is:

    base   = points_per_price_unit[position] * price_millions
    matchup = 1 + alpha * tanh(strength_diff / scale)
    home    = 1 + beta * is_home
    premium = premium_boost * max(0, price_millions - PREMIUM_PRICE_THRESHOLD)

    predicted_points = base * matchup * home + premium

and zeroed for any player whose `status` is not "playing" or whose squad is
eliminated. `premium_boost = 0` (default) reproduces the original heuristic;
positive values add a linear-above-threshold term that tilts the optimizer
toward £10M+ players whose ceiling the linear-in-price base under-weights.
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
STRENGTH_DIFF_SCALE = 2.0
STRENGTH_DIFF_ALPHA = 0.25  # ±25% at saturation
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


def heuristic_predict(
    features: pd.DataFrame,
    premium_boost: float = DEFAULT_PREMIUM_BOOST,
) -> pd.DataFrame:
    """Add a `predicted_points` column to a copy of `features`.

    Players whose `status` != "playing" or whose squad is eliminated are
    predicted at 0. Otherwise the formula in the module docstring applies.

    `premium_boost` adds `premium_boost * max(0, price - 9.0)` to the
    prediction. Defaults to 0.0 (preserves original behaviour). Values
    around 0.3–0.6 tilt the optimizer toward £10M+ players.
    """
    out = features.copy()

    coef = out["position"].map(_position_coef).astype(float)
    price = out["price_millions"].astype(float)
    base = coef * price

    matchup = 1.0 + STRENGTH_DIFF_ALPHA * np.tanh(
        out["strength_diff"].astype(float) / STRENGTH_DIFF_SCALE
    )
    home = 1.0 + HOME_ADVANTAGE_BETA * out["is_home"].astype(int)

    raw = base * matchup * home
    if premium_boost:
        raw = raw + premium_boost * np.maximum(0.0, price - PREMIUM_PRICE_THRESHOLD)

    available = (out["status"] == PLAYING_STATUS) & (~out["is_eliminated"].astype(bool))
    out["predicted_points"] = np.where(available, raw, 0.0)
    return out
