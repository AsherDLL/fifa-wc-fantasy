"""Dedicated captain / vice-captain selection algorithm.

The default `solve_lineup` picks the captain as `argmax(predicted_points)`
within the starting XI, with vice as second-argmax. That is a naive
algorithm that ignores three things known to matter:

1. **Variance**: between two players with similar means, prefer the
   higher-floor player when defending a lead and the higher-variance
   player when chasing.
2. **Differential**: a captain captained by 50% of the league converts
   no rank gain even on a haul; a captain captained by 5% who hauls is
   massive rank gain. Standings position determines how much to weight
   this.
3. **Fixture quality independent of model**: a 4-0 expected blowout
   has different captain dynamics than a 1-1 expected draw, even if
   both produce similar mean predicted_points.
4. **Vice-captain coverage**: vice must play in a DIFFERENT match than
   the captain (so that if captain blanks, vice can replace). Naive
   argmax can pick vice from the same match.

This module implements a scoring function that combines all four
considerations into a single composite captain score, then picks the
two highest distinct-match-having players from the XI.

Usage:
    from fifa_fantasy.optimizer.captain import select_captain_vice

    decision = select_captain_vice(
        xi_predictions=md_round_predictions,
        standings_position=user_standings_pos,
        league_size=20,
    )
    print(decision.captain_id, decision.vice_id, decision.rationale)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Tuning knobs for the captain score components.
WEIGHT_MEAN = 1.00              # baseline weight on the ceiling-vs-mean base
WEIGHT_VARIANCE = 0.10          # mild penalty for blank-prone (high p10=0)
WEIGHT_DIFFERENTIAL = 0.20      # max bonus when chasing
WEIGHT_FIXTURE_GAP = 0.05       # bonus for big Elo gap fixtures

# Standings position threshold: if user is above this percentile in
# their league, prefer template captains; below, prefer differentials.
LEAD_THRESHOLD = 0.50           # 50th percentile (median)

# Ceiling weighting. The base captain term interpolates between the mean
# prediction and the q90 ceiling by how hard the user is chasing:
#   base = (1 - chase) * mean + chase * q90,   chase = standings_pos_pct.
# A leader (chase=0) captains the safe mean; a team in last (chase=1)
# captains the ceiling. This is not a preference, it is the empirical
# winner: on a leak-free per-round backtest over MD2, MD3 and R32
# (scripts wc_forward_validation context, docs 11g), captaining by q90
# captured 21 raw captain points against 11 for the mean-argmax rule the
# lineup solver had been using, with the optimal being 42.
CEILING_MAX_CHASE = 1.0         # cap on the chase interpolation weight


@dataclass(frozen=True)
class CaptainDecision:
    captain_id: int
    captain_name: str
    vice_id: int
    vice_name: str
    captain_score: float
    vice_score: float
    rationale: dict
    candidates_ranked: list[dict]


def _num_or(row, attr: str, default: float) -> float:
    """Numeric attribute with a default that also covers NaN.

    `getattr(row, attr, default)` alone is a trap: under the ensemble
    backend the quantile columns EXIST but hold NaN for the positions
    routed to point-estimate backends (poisson GK, heuristic DEF). A NaN
    p90 turns the composite score NaN, and sorting NaN keys is undefined
    order, which once crowned a goalkeeper captain. NaN must fall back
    exactly like a missing attribute.
    """
    value = getattr(row, attr, None)
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def _row_to_dict(row) -> dict:
    mean = float(row.predicted_points)
    return {
        "player_id": int(row.player_id),
        "name": row.full_name,
        "country": row.country_abbr,
        "position": row.position,
        "opponent": row.opponent_abbr,
        "is_home": bool(row.is_home),
        "predicted_points": mean,
        "ownership_pct": _num_or(row, "ownership_pct", 0.0),
        "p10": _num_or(row, "predicted_p10", mean),
        "p90": _num_or(row, "predicted_p90", mean),
        "elo_diff": _num_or(row, "country_elo_diff", 0.0),
        "fixture_id": int(_num_or(row, "fixture_id", -1)),
    }


def captain_composite_score(player: dict,
                            standings_pos_pct: float = 0.5,
                            league_size: int = 20) -> tuple[float, dict]:
    """Composite score for a player's captain suitability.

    Combines:
        mean_score    = WEIGHT_MEAN * predicted_points
        variance_pen  = -WEIGHT_VARIANCE * P(blank); proxied by max(0, mean - p10) / mean
        differential  = WEIGHT_DIFFERENTIAL * (1 - ownership) * lambda(standings)
                        where lambda = 1 when chasing, 0 when leading
        fixture_bonus = WEIGHT_FIXTURE_GAP * tanh(elo_diff / 400)

    Returns:
        (composite_score, breakdown_dict)
    """
    mean = player["predicted_points"]
    p10 = player["p10"]
    p90 = player["p90"]
    ownership_frac = player["ownership_pct"] / 100.0
    elo_diff = player["elo_diff"]

    # Ceiling-vs-mean base: interpolate toward q90 by how hard we chase.
    # When q90 is unavailable it equals the mean (see _row_to_dict default),
    # so this degrades to the mean term for point-estimate backends.
    chase = min(CEILING_MAX_CHASE, max(0.0, standings_pos_pct))
    base = (1.0 - chase) * mean + chase * p90
    mean_term = WEIGHT_MEAN * base

    blank_indicator = max(0.0, (mean - p10)) / max(mean, 0.01)
    variance_term = -WEIGHT_VARIANCE * blank_indicator * mean

    # Differential lambda: scales linearly from 0 (top of league) to 1
    # (bottom). standings_pos_pct = 0 means leader; 1.0 means last.
    lam = max(0.0, standings_pos_pct - LEAD_THRESHOLD) / (1.0 - LEAD_THRESHOLD)
    differential_term = WEIGHT_DIFFERENTIAL * (1.0 - ownership_frac) * lam * mean

    fixture_term = WEIGHT_FIXTURE_GAP * np.tanh(elo_diff / 400.0) * mean

    composite = mean_term + variance_term + differential_term + fixture_term
    return composite, {
        "mean_term": mean_term,
        "ceiling_base": base,
        "chase_weight": chase,
        "variance_term": variance_term,
        "differential_term": differential_term,
        "fixture_term": fixture_term,
        "differential_lambda": lam,
        "blank_indicator": blank_indicator,
    }


def select_captain_vice(xi_predictions: pd.DataFrame,
                       standings_pos_pct: float = 0.5,
                       league_size: int = 20) -> CaptainDecision:
    """Pick the best captain and vice from the XI.

    Args:
        xi_predictions: DataFrame with one row per starting XI player.
            Required columns: player_id, full_name, country_abbr,
            position, opponent_abbr, is_home, predicted_points.
            Optional but recommended: ownership_pct, predicted_p10,
            predicted_p90, country_elo_diff, fixture_id.
        standings_pos_pct: user's position percentile in league;
            0.0 = leader, 1.0 = last. Default 0.5 (median, neutral).
        league_size: number of teams in the personal league.

    Returns:
        CaptainDecision with captain_id, vice_id, scores, rationale.
    """
    candidates = []
    for r in xi_predictions.itertuples():
        p = _row_to_dict(r)
        score, breakdown = captain_composite_score(
            p, standings_pos_pct=standings_pos_pct, league_size=league_size
        )
        p["composite_score"] = float(score)
        p["score_breakdown"] = breakdown
        candidates.append(p)

    candidates.sort(key=lambda x: -x["composite_score"])
    captain = candidates[0]
    # Vice: highest-score candidate from a DIFFERENT match.
    vice = None
    for c in candidates[1:]:
        if c["fixture_id"] != captain["fixture_id"]:
            vice = c
            break
    if vice is None and len(candidates) > 1:
        # Fallback: same-match vice (e.g. all matches overlap).
        vice = candidates[1]

    return CaptainDecision(
        captain_id=captain["player_id"],
        captain_name=captain["name"],
        vice_id=vice["player_id"] if vice else captain["player_id"],
        vice_name=vice["name"] if vice else captain["name"],
        captain_score=captain["composite_score"],
        vice_score=vice["composite_score"] if vice else 0.0,
        rationale={
            "standings_pos_pct": standings_pos_pct,
            "league_size": league_size,
            "differential_lambda": captain["score_breakdown"]["differential_lambda"],
            "captain_breakdown": captain["score_breakdown"],
            "vice_breakdown": vice["score_breakdown"] if vice else {},
        },
        candidates_ranked=candidates,
    )
