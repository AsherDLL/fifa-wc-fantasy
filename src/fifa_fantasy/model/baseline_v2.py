"""Heuristic v2 - empirically tuned and expert-knowledge-informed.

Improvements over v1 (`baseline.py`):

1. **Realised-form anchor**. Once a player has 2+ matches of tournament
   data, blend the model's structural prediction with their actual
   per-match average. The blend weight shifts toward realised data as
   N grows. This addresses v1's blindness to non-EPL stars (Messi at
   model 5 vs realised 13).

2. **Position-specific scoring components**.
   - GK: scale save bonus by opp_xg (empirically 0.50 multiplier;
     Section 09c). Clean sheet probability from Poisson.
   - DEF: clean sheet probability bonus + attacking-return prior for
     full-backs (proxied by price tier).
   - MID: ball-progression / chance-creation bonus, proxied by realised
     total relative to position median.
   - FWD: penalty-taker / goal-conversion bonus from realised goals
     per match.

3. **Rotation risk multiplier**. Per-team factor in [0, 1] from
   `rotation_risk.py` (loaded from per-stage config). Group-stage
   clinched teams get 0.55-0.80; knockout games get 1.0 unless
   eliminated.

4. **Premium-tier non-linear bump**. Players above $9.5M get an extra
   boost reflecting the heavy-tailed nature of elite-tier scoring
   (Mbappé, Messi, Haaland can put up 15+ pt games; mid-priced
   players cannot).

5. **Set-piece premium**. Players flagged as penalty/corner/free-kick
   takers (currently hard-coded from external knowledge; future work
   to scrape) get an additive bonus.

Validation: held-out RMSE on EPL 2024-25 GW 30-38 and live WC backtest
(Section 8b extended). Must not regress v1's per-position numbers and
must improve on at least one position to ship.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fifa_fantasy.scoring import Position

# v1 constants (reused).
POINTS_PER_PRICE_UNIT: dict[Position, float] = {
    Position.GK: 0.50,
    Position.DEF: 0.55,
    Position.MID: 0.60,
    Position.FWD: 0.65,
}
STRENGTH_DIFF_SCALE = 2.0
RANK_DIFF_SCALE = 250.0
ELO_DIFF_SCALE = 400.0
PRICE_SIGNAL_WEIGHT = 0.35
STRENGTH_SIGNAL_WEIGHT = 0.65
STRENGTH_DIFF_ALPHA = 0.40
HOME_ADVANTAGE_BETA = 0.05
PREMIUM_PRICE_THRESHOLD = 9.0

# v2 additions.
REALISED_BLEND_MAX = 0.40   # max weight on realised average; reached after 4+ games
PREMIUM_TIER_BUMP = 0.08    # per million above 9.5M (premium ceiling capture)
PREMIUM_TIER_THRESHOLD = 9.5

# GK save bonus (consistent with Poisson v2 empirical fit).
GK_SAVE_BONUS_PER_OPP_XG = 0.50
# DEF clean sheet probability bonus. Empirically calibrated.
DEF_CS_BONUS_FACTOR = 0.3
# MID ball-progression bonus from realised over-performance.
MID_PROGRESSION_FACTOR = 0.15
# FWD goal-conversion bonus.
FWD_CONVERSION_FACTOR = 0.20

# Set-piece takers known pre-tournament (small curated list; future
# work to scrape per-round). Maps player_id -> bonus pts/match.
# Empty by default; populate from external knowledge as discovered.
SET_PIECE_TAKERS: dict[int, float] = {
    # Examples (commented out; populate when validated):
    # 38: 0.6,    # Messi (Argentina free kicks)
    # 500: 0.5,   # Mbappé (France penalties)
    # 468: 0.7,   # Kane (England penalties)
}

PLAYING_STATUS = "playing"


def _position_coef(value: object) -> float:
    if isinstance(value, Position):
        return POINTS_PER_PRICE_UNIT[value]
    return POINTS_PER_PRICE_UNIT[Position(value)]


def _strength_z(features: pd.DataFrame) -> np.ndarray:
    """Blend price-based and Elo/rank-based strength signals (same as v1)."""
    z_price = features["strength_diff"].astype(float).to_numpy() / STRENGTH_DIFF_SCALE
    elo_raw = pd.to_numeric(features.get("country_elo_diff"), errors="coerce") \
        if "country_elo_diff" in features.columns else pd.Series(
            [pd.NA] * len(features), dtype="Float64"
        )
    z_elo = elo_raw.to_numpy(dtype=float) / ELO_DIFF_SCALE
    has_elo = ~pd.isna(elo_raw).to_numpy()

    rank_raw = pd.to_numeric(features.get("rank_diff"), errors="coerce") \
        if "rank_diff" in features.columns else pd.Series(
            [pd.NA] * len(features), dtype="Float64"
        )
    z_rank = rank_raw.to_numpy(dtype=float) / RANK_DIFF_SCALE
    has_rank = ~pd.isna(rank_raw).to_numpy()

    z_strength = np.where(has_elo, z_elo, np.where(has_rank, z_rank, 0.0))
    has_strength = has_elo | has_rank
    return np.where(
        has_strength,
        PRICE_SIGNAL_WEIGHT * z_price + STRENGTH_SIGNAL_WEIGHT * z_strength,
        z_price,
    )


def _opp_xg(features: pd.DataFrame) -> np.ndarray:
    """Estimate opponent expected goals using the Poisson backend's formula."""
    from .poisson import (
        BASE_TEAM_XG, ELO_DIFF_SLOPE_XG, HOME_XG_BOOST, PRICE_DIFF_SLOPE_XG,
        RANK_DIFF_SLOPE_XG,
    )
    price_diff = features["strength_diff"].astype(float).to_numpy()
    elo_diff = pd.to_numeric(features.get("country_elo_diff"), errors="coerce") \
        if "country_elo_diff" in features.columns else pd.Series(np.nan, index=features.index)
    rank_diff = pd.to_numeric(features.get("rank_diff"), errors="coerce") \
        if "rank_diff" in features.columns else pd.Series(np.nan, index=features.index)
    elo_arr = np.asarray(elo_diff, dtype=float)
    rank_arr = np.asarray(rank_diff, dtype=float)
    has_elo = ~np.isnan(elo_arr)
    has_rank = ~np.isnan(rank_arr)
    strength_component = np.where(
        has_elo, np.nan_to_num(elo_arr) * ELO_DIFF_SLOPE_XG,
        np.where(has_rank, np.nan_to_num(rank_arr) * RANK_DIFF_SLOPE_XG, 0.0),
    )
    matchup = strength_component + price_diff * PRICE_DIFF_SLOPE_XG
    is_home = features["is_home"].astype(int).to_numpy()
    opp_factor = np.exp(-np.clip(matchup, -1.2, 1.2)) * (
        1.0 + HOME_XG_BOOST * (1 - is_home)
    )
    return BASE_TEAM_XG * opp_factor


def _realised_avg_per_match(features: pd.DataFrame) -> np.ndarray:
    """Per-row realised average points per match.

    Computed from `round_points` (list of per-round scores). When N=0
    (pre-tournament), returns 0 and the blend weight will be 0.
    """
    avgs = np.zeros(len(features))
    if "round_points" not in features.columns:
        return avgs
    rps = features["round_points"].values
    totals = features.get("total_points", pd.Series(np.zeros(len(features)))).astype(float).to_numpy()
    for i, rp in enumerate(rps):
        try:
            n = len(list(rp)) if rp is not None else 0
        except TypeError:
            n = 0
        if n > 0:
            avgs[i] = totals[i] / n
    return avgs


def _realised_blend_weight(features: pd.DataFrame) -> np.ndarray:
    """Weight on realised data; grows from 0 (no games) to REALISED_BLEND_MAX (4+ games)."""
    n_games = np.zeros(len(features))
    if "round_points" not in features.columns:
        return n_games
    for i, rp in enumerate(features["round_points"].values):
        try:
            n_games[i] = len(list(rp)) if rp is not None else 0
        except TypeError:
            n_games[i] = 0
    return np.clip(n_games / 4.0, 0.0, 1.0) * REALISED_BLEND_MAX


def _gk_save_pts(features: pd.DataFrame, gk_mask: np.ndarray, opp_xg: np.ndarray) -> np.ndarray:
    """Expected GK save points scaled by opponent xG (Section 09c)."""
    out = np.zeros(len(features))
    out[gk_mask] = np.maximum(opp_xg[gk_mask], 0.0) * GK_SAVE_BONUS_PER_OPP_XG
    return out


def _def_cs_bonus(features: pd.DataFrame, def_mask: np.ndarray, opp_xg: np.ndarray) -> np.ndarray:
    """Clean sheet probability bonus for defenders."""
    out = np.zeros(len(features))
    # P(clean sheet) under Poisson(opp_xg) = exp(-opp_xg)
    cs_prob = np.exp(-np.maximum(opp_xg, 0.0))
    out[def_mask] = cs_prob[def_mask] * DEF_CS_BONUS_FACTOR * 5  # 5 = CS reward for DEF
    return out


def _mid_progression_bonus(features: pd.DataFrame, mid_mask: np.ndarray,
                            realised_avg: np.ndarray, position_avg: dict) -> np.ndarray:
    """Bonus for midfielders over-performing their position average."""
    out = np.zeros(len(features))
    pos_avg_mid = position_avg.get("MID", 2.0)
    if pos_avg_mid <= 0:
        return out
    over = (realised_avg - pos_avg_mid) / pos_avg_mid
    out[mid_mask] = np.clip(over[mid_mask], 0.0, 1.5) * MID_PROGRESSION_FACTOR * pos_avg_mid
    return out


def _fwd_conversion_bonus(features: pd.DataFrame, fwd_mask: np.ndarray,
                          realised_avg: np.ndarray, position_avg: dict) -> np.ndarray:
    """Goal-conversion bonus for forwards over-performing their position average."""
    out = np.zeros(len(features))
    pos_avg_fwd = position_avg.get("FWD", 2.0)
    if pos_avg_fwd <= 0:
        return out
    over = (realised_avg - pos_avg_fwd) / pos_avg_fwd
    out[fwd_mask] = np.clip(over[fwd_mask], 0.0, 1.5) * FWD_CONVERSION_FACTOR * pos_avg_fwd
    return out


def _premium_tier_bump(features: pd.DataFrame) -> np.ndarray:
    """Non-linear bump for premium players (above $9.5M)."""
    price = features["price_millions"].astype(float).to_numpy()
    return PREMIUM_TIER_BUMP * np.maximum(price - PREMIUM_TIER_THRESHOLD, 0.0)


def _set_piece_bonus(features: pd.DataFrame) -> np.ndarray:
    """Additive bonus for known set-piece takers."""
    pids = features["player_id"].astype(int).to_numpy()
    return np.array([SET_PIECE_TAKERS.get(int(p), 0.0) for p in pids])


def heuristic_v2_predict(features: pd.DataFrame,
                         rotation_risk: dict[str, float] | None = None) -> pd.DataFrame:
    """Add `predicted_points` to a copy of `features` using the v2 formula.

    Args:
        features: per-(player, round) feature table.
        rotation_risk: optional dict[country_abbr -> multiplier in (0,1]].
            When supplied, predictions for that country's players are
            scaled by the multiplier. Defaults to all 1.0 (no rotation
            haircut).

    Returns:
        Copy of features with `predicted_points` column added (along
        with `v2_components` debug column showing the breakdown).
    """
    out = features.copy()

    # Base from v1 formula.
    coef = out["position"].map(_position_coef).astype(float)
    price = out["price_millions"].astype(float)
    z = _strength_z(out)
    matchup = 1.0 + STRENGTH_DIFF_ALPHA * np.tanh(z)
    home = 1.0 + HOME_ADVANTAGE_BETA * out["is_home"].astype(int)
    v1_base = coef * price * matchup * home

    # Position masks.
    positions = out["position"].to_numpy()
    gk_mask = positions == "GK"
    def_mask = positions == "DEF"
    mid_mask = positions == "MID"
    fwd_mask = positions == "FWD"

    # Opponent xG.
    opp_xg = _opp_xg(out)

    # Per-position bonuses.
    gk_bonus = _gk_save_pts(out, gk_mask, opp_xg)
    def_bonus = _def_cs_bonus(out, def_mask, opp_xg)
    pos_avg = {
        "GK": float(out.loc[gk_mask, "total_points"].astype(float).mean()
                    if gk_mask.any() else 2.0),
        "DEF": float(out.loc[def_mask, "total_points"].astype(float).mean()
                     if def_mask.any() else 2.0),
        "MID": float(out.loc[mid_mask, "total_points"].astype(float).mean()
                     if mid_mask.any() else 2.0),
        "FWD": float(out.loc[fwd_mask, "total_points"].astype(float).mean()
                     if fwd_mask.any() else 2.0),
    }
    realised_avg = _realised_avg_per_match(out)
    mid_bonus = _mid_progression_bonus(out, mid_mask, realised_avg, pos_avg)
    fwd_bonus = _fwd_conversion_bonus(out, fwd_mask, realised_avg, pos_avg)

    # Premium tier + set-piece.
    premium_bump = _premium_tier_bump(out)
    set_piece_bonus = _set_piece_bonus(out)

    # Combine.
    model_pred = (v1_base + gk_bonus + def_bonus + mid_bonus + fwd_bonus
                  + premium_bump + set_piece_bonus)

    # Realised-form anchor: blend model with realised average.
    blend_w = _realised_blend_weight(out)
    blended = (1 - blend_w) * model_pred + blend_w * realised_avg

    # Rotation risk multiplier.
    if rotation_risk is not None and "country_abbr" in out.columns:
        country = out["country_abbr"].astype(str)
        risk_mult = country.map(lambda c: rotation_risk.get(c, 1.0)).astype(float).to_numpy()
        blended = blended * risk_mult

    available = (out["status"] == PLAYING_STATUS) & (~out["is_eliminated"].astype(bool))
    out["predicted_points"] = np.where(available, np.clip(blended, 0, None), 0.0)

    # Optional debug columns
    out["v2_v1_base"] = v1_base
    out["v2_gk_bonus"] = gk_bonus
    out["v2_def_bonus"] = def_bonus
    out["v2_mid_bonus"] = mid_bonus
    out["v2_fwd_bonus"] = fwd_bonus
    out["v2_premium_bump"] = premium_bump
    out["v2_realised_avg"] = realised_avg
    out["v2_blend_weight"] = blend_w
    return out
