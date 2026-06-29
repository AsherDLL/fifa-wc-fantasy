"""Structural Poisson goals predictor.

A third backend, distinct from the heuristic (price-driven) and the GBM
(EPL-trained). The structure follows standard football modelling:

1. Estimate per-team expected goals for the fixture from strength signals
   (FIFA ranking, squad top-11 price). No training, no labels needed.
2. Distribute team xG across player positions using fixed share weights
   for goals and assists. Forwards take the largest share of goals;
   midfielders carry assists; defenders share modest goal contributions.
3. Translate per-player expected goals + expected assists +
   appearance + clean-sheet probability + scouting bonus into Fantasy
   points using the same scoring constants the rest of the project
   already encodes.

This is intentionally interpretable: every number can be traced back to
a single formula. It does not learn from data; it leverages structure.
Use it as a check on the heuristic and the GBM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Pool-wide WC group-stage scoring rate, anchored to the typical 2.5 to
# 2.8 goals per match across recent tournaments. We split symmetrically as
# the base for an evenly-matched fixture; the matchup signal shifts it.
BASE_TEAM_XG = 1.30                 # expected goals for a typical WC team per match
HOME_XG_BOOST = 0.10                # +10% xG for home side
RANK_DIFF_SLOPE_XG = 1.0 / 800.0    # 800 ranking points ~ 1.0x swing
PRICE_DIFF_SLOPE_XG = 1.0 / 6.0     # 6.0M price diff ~ 1.0x swing
# Elo's natural log-odds scale: 400 Elo = 10x odds. A 400-Elo gap should
# move team xG roughly 1.0x (matchup factor e^1 ~ 2.7), matching what the
# FIFA-rank slope produced for top-vs-bottom WC fixtures. Elo data is a
# live, derived signal so it takes precedence over rank_diff when present.
ELO_DIFF_SLOPE_XG = 1.0 / 400.0

# Per-position share of own-team expected goals (rough WC consensus).
GOAL_SHARE = {"GK": 0.0, "DEF": 0.10, "MID": 0.30, "FWD": 0.50}
# Per-position share of own-team expected assists. Sums to ~1 across
# outfield positions; GK is essentially zero.
ASSIST_SHARE = {"GK": 0.01, "DEF": 0.18, "MID": 0.50, "FWD": 0.31}

# Fantasy.md point values for goals scored, by position.
GOAL_POINTS = {"GK": 9, "DEF": 7, "MID": 6, "FWD": 5}
# Clean-sheet rewards by position.
CLEAN_SHEET_POINTS = {"GK": 5, "DEF": 5, "MID": 1, "FWD": 0}
# Appearance assumed full match for a starter (60+ minutes).
APPEARANCE_POINTS = 2
# Common rules.
ASSIST_POINTS = 3
# Scouting bonus thresholds from scoring.py (referenced explicitly here so
# this module stays self-contained).
SCOUTING_BONUS_POINTS_THRESHOLD = 4
SCOUTING_BONUS_OWNERSHIP_THRESHOLD = 0.05
SCOUTING_BONUS = 2

# Goalkeeper save-bonus scaling. v1 used a flat 1.0 constant (the same
# expected save bonus for every goalkeeper regardless of opponent
# strength). That formulation lost information: GKs facing high-xG
# opponents see more shots and accumulate more save-bonus points than
# GKs facing low-xG opponents, all else equal.
#
# v2 (current): scales with opp_xg. The multiplier was first derived
# theoretically as SHOT_PER_XG_RATIO * SAVE_PCT / SAVES_PER_BONUS ≈ 1.13
# but empirically calibrated to 0.50 against EPL 2024-25 GW 30-38
# held-out plus WC 2026 MD1-R32 realised data. The theoretical value
# overestimates because not all opponent xG produces on-target shots and
# the median (not top-tier) GK does not save at the elite 0.85 rate.
# See scripts/gk_formula_ab.py and docs/whitepaper/sections/09c_*.md.
GK_SAVE_BONUS_PER_OPP_XG = 0.50
DEF_GC_PENALTY_FACTOR = 0.5  # average -0.5 pts/match from goals-conceded beyond first


def _team_xg(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (own_team_xG_for, opp_team_xG_for) per row.

    Both are positive scalars representing expected goals scored in this
    fixture. The matchup signal combines the FIFA ranking gap and the
    top-11 price gap into a multiplicative factor; the home side gets a
    small boost.
    """
    price_diff = features["strength_diff"].astype(float).to_numpy()

    # National-team strength: prefer Elo (live, real matches) over FIFA
    # ranking (static snapshot). When neither is present (EPL training),
    # the price component carries the matchup signal.
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
    # Map matchup to a multiplicative factor centred on 1.0. Clipping the
    # exponent keeps the predictions sane at the extremes (e.g. France
    # vs Curacao).
    own_factor = np.exp(np.clip(matchup, -1.2, 1.2))
    opp_factor = np.exp(-np.clip(matchup, -1.2, 1.2))

    is_home = features["is_home"].astype(int).to_numpy()
    own_factor = own_factor * (1.0 + HOME_XG_BOOST * is_home)
    opp_factor = opp_factor * (1.0 + HOME_XG_BOOST * (1 - is_home))

    own_xg = BASE_TEAM_XG * own_factor
    opp_xg = BASE_TEAM_XG * opp_factor
    return own_xg, opp_xg


def _clean_sheet_prob(opp_xg: np.ndarray) -> np.ndarray:
    """Poisson P(opp scores 0) given opp_xg."""
    return np.exp(-opp_xg)


def _per_position_player_xg(position: str, own_xg: np.ndarray) -> np.ndarray:
    share = GOAL_SHARE[position]
    return own_xg * share


def _per_position_player_assists(position: str, own_xg: np.ndarray) -> np.ndarray:
    # Assist count tracks team goals at a slightly lower rate; we use the
    # same own_xg as a proxy because goals and assists scale together.
    share = ASSIST_SHARE[position]
    return own_xg * 0.7 * share  # roughly 70% of goals have an assist


def poisson_predict(features: pd.DataFrame) -> pd.DataFrame:
    """Return `features` with a `predicted_points` column attached.

    Players whose `status` != "playing" or whose squad is eliminated are
    predicted at 0, matching the contract of the other backends.
    """
    out = features.copy()
    own_xg, opp_xg = _team_xg(out)
    cs_prob = _clean_sheet_prob(opp_xg)

    positions = out["position"].to_numpy()
    base = np.full(len(out), float(APPEARANCE_POINTS))

    # Goals contribution.
    goal_pts = np.zeros(len(out))
    for pos in ("GK", "DEF", "MID", "FWD"):
        mask = positions == pos
        if not mask.any():
            continue
        player_xg = _per_position_player_xg(pos, own_xg[mask])
        goal_pts[mask] = player_xg * GOAL_POINTS[pos]

    # Assists.
    assist_pts = np.zeros(len(out))
    for pos in ("GK", "DEF", "MID", "FWD"):
        mask = positions == pos
        if not mask.any():
            continue
        player_xa = _per_position_player_assists(pos, own_xg[mask])
        assist_pts[mask] = player_xa * ASSIST_POINTS

    # Clean sheet bonus.
    cs_pts = np.zeros(len(out))
    for pos, val in CLEAN_SHEET_POINTS.items():
        mask = positions == pos
        if not mask.any() or val == 0:
            continue
        cs_pts[mask] = cs_prob[mask] * val

    # Goalkeeper save bonus + goals-conceded penalty.
    # Save bonus scales with opp_xg (more shots faced -> more save chances).
    # See module-level comment for the calibration history.
    gk_mask = positions == "GK"
    gk_save_pts = np.zeros(len(out))
    gk_save_pts[gk_mask] = np.maximum(opp_xg[gk_mask], 0.0) * GK_SAVE_BONUS_PER_OPP_XG

    def_mask = positions == "DEF"
    gc_penalty = np.zeros(len(out))
    # Use expected goals conceded beyond the first as a -1 per goal hit.
    # E[max(0, k-1)] under Poisson(opp_xg) = opp_xg - 1 + exp(-opp_xg).
    expected_extra_conceded = np.clip(opp_xg - 1.0 + cs_prob, 0.0, None)
    gc_penalty[gk_mask] = -expected_extra_conceded[gk_mask]
    gc_penalty[def_mask] = -expected_extra_conceded[def_mask] * DEF_GC_PENALTY_FACTOR

    raw = base + goal_pts + assist_pts + cs_pts + gk_save_pts + gc_penalty

    # Scouting bonus: +2 pts when predicted > 4 AND ownership < 5%.
    ownership = out["ownership_fraction"].astype(float).to_numpy()
    scouting = np.where(
        (raw > SCOUTING_BONUS_POINTS_THRESHOLD)
        & (ownership < SCOUTING_BONUS_OWNERSHIP_THRESHOLD),
        SCOUTING_BONUS,
        0.0,
    )
    raw = raw + scouting

    available = (
        (out["status"] == "playing")
        & (~out["is_eliminated"].astype(bool))
    )
    out["predicted_points"] = np.where(available, np.clip(raw, 0.0, None), 0.0)
    return out
