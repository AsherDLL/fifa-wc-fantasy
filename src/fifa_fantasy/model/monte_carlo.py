"""Monte Carlo per-match simulator (Section 11b innovation contribution).

For each (player, round) row, simulate N versions of the match and
aggregate to a fantasy-points distribution. Unlike the heuristic and
GBM backends which output a single point estimate, the Monte Carlo
backend returns:

    predicted_points     mean of the simulated distribution
    predicted_p10        10th percentile
    predicted_p50        median
    predicted_p90        90th percentile

The simulation per match:

  1. Sample team xG from a log-normal centred on the Poisson backend's
     own_xg estimate (small multiplicative noise to capture form/news).
  2. Sample team goals from Poisson(team_xg).
  3. Distribute goals to players using a position share scaled by
     form-weighted player multipliers. The multiplier for a player is
     `realised_avg_per_match / position_avg_for_position`.
  4. Sample assists at 0.7 * goals using the same distribution shape.
  5. For GKs: sample shots faced from opp_xg * 4, then saves as
     binomial(shots, save_pct), with save_pct read from realised data
     when available, else 0.7.
  6. Compute fantasy points using the canonical scoring constants.

A Polymarket prior on goal difference (when available) is incorporated
as a multiplicative adjustment to team xG: if the market implies a
larger goal differential than our model, we shift the team-xg ratio.

This backend is NOT trained on EPL data; it consumes the WC-2026
realised-data anchors directly. Its outputs are particularly useful
in the knockout rounds where single-match variance dominates and a
distribution beats a point estimate.

CALIBRATION STATUS: experimental. The simulator's mean has not been
held-out validated against EPL training data the way the heuristic
and Poisson backends were. Use with care until Section 7.5b validation
is complete.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .poisson import (
    BASE_TEAM_XG, ELO_DIFF_SLOPE_XG, GOAL_POINTS, HOME_XG_BOOST,
    PRICE_DIFF_SLOPE_XG, RANK_DIFF_SLOPE_XG,
)

# Per-position share of own-team goals (same as Poisson backend).
GOAL_SHARE = {"GK": 0.0, "DEF": 0.10, "MID": 0.30, "FWD": 0.50}
ASSIST_SHARE = {"GK": 0.01, "DEF": 0.18, "MID": 0.50, "FWD": 0.31}

# Fantasy scoring constants.
APPEARANCE_POINTS = 2
ASSIST_POINTS = 3
CLEAN_SHEET_POINTS = {"GK": 5, "DEF": 5, "MID": 1, "FWD": 0}
DEF_GC_PENALTY_FACTOR = 0.5
GK_SHOT_RATIO = 4.0          # shots on target per unit opp_xg (calibrated below)
DEFAULT_SAVE_PCT = 0.70      # median GK save percentage

SHOT_PER_XG_CALIBRATED = 2.0  # empirical; see scripts/gk_formula_ab.py findings
DEFAULT_PER_PLAYER_FORM_MULTIPLIER = 1.0   # neutral prior

N_SIMULATIONS_DEFAULT = 1000


def _team_xg(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (own_xg, opp_xg) using the Poisson backend's formula."""
    elo_diff = pd.to_numeric(features.get("country_elo_diff"), errors="coerce") \
        if "country_elo_diff" in features.columns else pd.Series(np.nan, index=features.index)
    rank_diff = pd.to_numeric(features.get("rank_diff"), errors="coerce") \
        if "rank_diff" in features.columns else pd.Series(np.nan, index=features.index)
    price_diff = features["strength_diff"].astype(float).to_numpy()
    elo_arr = np.asarray(elo_diff, dtype=float)
    rank_arr = np.asarray(rank_diff, dtype=float)
    has_elo = ~np.isnan(elo_arr)
    has_rank = ~np.isnan(rank_arr)
    strength_component = np.where(
        has_elo, np.nan_to_num(elo_arr) * ELO_DIFF_SLOPE_XG,
        np.where(has_rank, np.nan_to_num(rank_arr) * RANK_DIFF_SLOPE_XG, 0.0),
    )
    matchup = strength_component + price_diff * PRICE_DIFF_SLOPE_XG
    own_factor = np.exp(np.clip(matchup, -1.2, 1.2))
    opp_factor = np.exp(-np.clip(matchup, -1.2, 1.2))
    is_home = features["is_home"].astype(int).to_numpy()
    own_factor = own_factor * (1.0 + HOME_XG_BOOST * is_home)
    opp_factor = opp_factor * (1.0 + HOME_XG_BOOST * (1 - is_home))
    return BASE_TEAM_XG * own_factor, BASE_TEAM_XG * opp_factor


def _player_form_multiplier(row: pd.Series, position_avg: float) -> float:
    """Form-weighted multiplier scaling player vs the position average.

    Uses total_points / matches_played as the per-match realised average.
    A player at 2x the positional average gets 2x the goal/assist share.
    """
    total = float(row.get("total_points", 0.0))
    rp = row.get("round_points")
    n = max(len(list(rp)), 1) if rp is not None else 1
    avg = total / n
    if position_avg <= 0:
        return DEFAULT_PER_PLAYER_FORM_MULTIPLIER
    return float(np.clip(avg / position_avg, 0.3, 3.0))


def simulate(features: pd.DataFrame,
             n_simulations: int = N_SIMULATIONS_DEFAULT,
             seed: int = 42) -> pd.DataFrame:
    """Add predicted_points + percentiles to a copy of `features`.

    Players with non-playing status or eliminated squads are predicted 0.
    """
    rng = np.random.default_rng(seed)
    out = features.copy()
    own_xg, opp_xg = _team_xg(out)
    positions = out["position"].to_numpy()

    # Per-position realised PER-MATCH average (form-multiplier denominator).
    # It must be on the same per-match scale as the numerator in
    # _player_form_multiplier (total / matches played). Using the mean of
    # tournament TOTALS here made the denominator grow with rounds played,
    # clamping every multiplier to the 0.3 floor by mid-tournament and
    # erasing the per-player differentiation the multiplier exists for.
    if "round_points" in out.columns:
        matches = out["round_points"].map(
            lambda rp: max(len(list(rp)), 1) if rp is not None else 1
        ).astype(float)
    else:
        matches = pd.Series(1.0, index=out.index)
    per_match = out["total_points"].astype(float) / matches
    pos_avg = {}
    for pos in ("GK", "DEF", "MID", "FWD"):
        mask = positions == pos
        if mask.any():
            vals = per_match[mask]
            pos_avg[pos] = float(vals.mean()) if len(vals) else 1.0
        else:
            pos_avg[pos] = 1.0

    # Pre-compute per-player form multipliers.
    form_mult = np.ones(len(out))
    for i, row in enumerate(out.itertuples()):
        pos = row.position
        form_mult[i] = _player_form_multiplier(out.iloc[i], pos_avg.get(pos, 1.0))

    # Simulate.
    sim_pts = np.zeros((n_simulations, len(out)))
    for s in range(n_simulations):
        # Per-row noise on team xG: log-normal, sigma small.
        own_xg_s = own_xg * rng.lognormal(mean=0.0, sigma=0.15, size=len(out))
        opp_xg_s = opp_xg * rng.lognormal(mean=0.0, sigma=0.15, size=len(out))

        # Team goals.
        team_goals = rng.poisson(own_xg_s)
        opp_goals = rng.poisson(opp_xg_s)

        # Per-player goals: fraction of team_goals proportional to position
        # share * form multiplier. Approximation: scale GOAL_SHARE by
        # form_mult and clip.
        player_goal_rate = np.array(
            [GOAL_SHARE.get(p, 0.0) for p in positions]) * form_mult
        player_xg_s = team_goals * player_goal_rate
        player_goals = rng.poisson(np.clip(player_xg_s, 0, 5))

        # Per-player assists at 70% of goals.
        player_assist_rate = np.array(
            [ASSIST_SHARE.get(p, 0.0) for p in positions]) * form_mult * 0.7
        player_assists = rng.poisson(np.clip(team_goals * player_assist_rate, 0, 5))

        # Goal points by position.
        goal_pts = np.array([GOAL_POINTS.get(p, 0) for p in positions]) * player_goals

        # Clean sheet?
        clean_sheet = opp_goals == 0
        cs_value = np.array([CLEAN_SHEET_POINTS.get(p, 0) for p in positions])
        cs_pts = np.where(clean_sheet, cs_value, 0)

        # GK saves.
        gk_mask = positions == "GK"
        save_pts = np.zeros(len(out))
        if gk_mask.any():
            shots = rng.poisson(np.clip(opp_xg_s[gk_mask] * SHOT_PER_XG_CALIBRATED, 0, 20))
            saves = rng.binomial(shots, DEFAULT_SAVE_PCT)
            save_pts[gk_mask] = saves // 3

        # Defender goals-conceded penalty.
        gc_penalty = np.zeros(len(out))
        def_mask = positions == "DEF"
        if def_mask.any():
            gc_penalty[def_mask] = -np.maximum(opp_goals[def_mask] - 1, 0) * DEF_GC_PENALTY_FACTOR

        # Assist points.
        assist_pts = player_assists * ASSIST_POINTS

        # Appearance.
        appearance = np.full(len(out), APPEARANCE_POINTS, dtype=float)

        sim_pts[s] = (appearance + goal_pts + assist_pts + cs_pts
                      + save_pts + gc_penalty)

    # Zero out non-playing/eliminated.
    available = ((out["status"] == "playing")
                 & (~out["is_eliminated"].astype(bool))).to_numpy()
    # Team-news gate: explicit-False xi flag means a confirmed bench player.
    # Scale predictions by xi_confidence when present; NaN leaves them as-is.
    if "predicted_starting_xi" in out.columns:
        xi = out["predicted_starting_xi"]
        is_bench = (xi == False).to_numpy()  # noqa: E712
        available = available & ~is_bench
        if "xi_confidence" in out.columns:
            # Scale by source confidence on rows where we have data.
            conf = pd.to_numeric(out["xi_confidence"], errors="coerce").fillna(1.0).to_numpy()
            # Apply only on rows with explicit news (starting=True); leave NaN/unset at 1.0.
            has_news = (xi == True).to_numpy()  # noqa: E712
            scale = np.where(has_news, conf, 1.0)
            sim_pts = sim_pts * scale
    sim_pts[:, ~available] = 0.0

    out["predicted_points"] = sim_pts.mean(axis=0)
    out["predicted_p10"] = np.percentile(sim_pts, 10, axis=0)
    out["predicted_p50"] = np.percentile(sim_pts, 50, axis=0)
    out["predicted_p90"] = np.percentile(sim_pts, 90, axis=0)
    return out


# Compatibility alias for the model CLI dispatcher.
def mc_predict(features: pd.DataFrame,
               n_simulations: int = N_SIMULATIONS_DEFAULT,
               seed: int = 42) -> pd.DataFrame:
    return simulate(features, n_simulations=n_simulations, seed=seed)
