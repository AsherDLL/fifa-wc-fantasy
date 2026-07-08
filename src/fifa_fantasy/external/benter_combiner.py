"""Benter (1994) logit combiner: model predictions + prediction-market signal.

This module implements the architectural separation defined in Section 11c
of the whitepaper. The soccer-analytics models (heuristic, Poisson, GBM,
Monte Carlo) produce per-player predicted_points. The prediction-market
signal (Polymarket, Kalshi) produces country-level implied probabilities.
This module combines them.

Two operational modes:

1. **Scaffold mode** (current, pre-tournament-end): use prior β values
   that have been documented but not empirically fit. The defaults are:
       β₀ = 0.0   (no intercept shift)
       β₁ = 1.0   (full model weight)
       β₂ = 0.20  (modest market weight)
   These are SCAFFOLD priors. We use them for the with-vs-without
   comparison framework to demonstrate the architecture; we do NOT
   claim they are optimal. Post-tournament fitting will replace them.

2. **Fit mode** (post-tournament): fit β coefficients by Bayesian
   logistic regression with PyMC against the historical paired
   (model_pred, market_implied, realised) data. Until WC 2026 final
   completes we don't have enough data; this is queued for future
   work.

The combination is multiplicative on player predictions:

    combined_pts = β₁ * model_pred * (1 + β₂ * (market_adjust - 1))

where `market_adjust` is a per-country multiplier derived from
Polymarket's country-advancement implied probability. If a country is
priced higher than our model says (market_adjust > 1), the combiner
upweights that country's players. If priced lower, it downweights.

When no market signal is available for a country, market_adjust = 1
and the combined prediction equals the raw model prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .prediction_markets import load_history

# Empirical β₂ optimum on WC 2026 MD1-3 is 0.00 (see Section 11d). The
# combiner is a no-op at this β; we keep the architecture for the
# Kalshi per-fixture markets when they become available, and for the
# post-tournament Bayesian refit.
DEFAULT_BETA = (0.0, 1.0, 0.00)   # (β₀, β₁, β₂) - empirical optimum


@dataclass(frozen=True)
class BenterConfig:
    """Combiner coefficients. Replace with fitted values post-tournament."""
    beta_0: float = DEFAULT_BETA[0]
    beta_1: float = DEFAULT_BETA[1]
    beta_2: float = DEFAULT_BETA[2]


def country_market_adjustments(snapshots: pd.DataFrame) -> dict[str, float]:
    """Derive per-country multiplicative adjustment from Polymarket prices.

    Reads the latest Polymarket WC-winner contracts and extracts
    each country's implied probability of winning the tournament.
    These are converted to multiplicative adjustments for player
    predictions in that country:

        country_adjustment[c] = implied_prob[c] / mean_implied_prob

    Countries above the average get a multiplier > 1; below, < 1.
    Returns a dict[country_name -> multiplier]. Missing countries
    default to 1.0 at the combine step.
    """
    if snapshots.empty:
        return {}
    latest = (snapshots[snapshots["provider"] == "polymarket"]
              .sort_values("snapshot_utc")
              .groupby("contract_id").tail(1))
    if latest.empty:
        return {}
    # Parse country name out of the contract title.
    # Pattern: "Will <country> win the 2026 FIFA World Cup?"
    import re
    pat = re.compile(r"Will (.+?) win the 2026 FIFA World Cup", re.IGNORECASE)
    rows = []
    for r in latest.itertuples():
        if r.yes_price is None or r.yes_price <= 0:
            continue
        m = pat.search(r.title or "")
        if not m: continue
        country = m.group(1).strip()
        rows.append((country, float(r.yes_price)))
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["country", "p_win"])
    # Drop zero-implied countries (eliminated or thin liquidity).
    df = df[df["p_win"] > 0.001]
    mean_p = df["p_win"].mean()
    df["adj"] = df["p_win"] / mean_p
    return dict(zip(df["country"], df["adj"]))


def combine(
    predictions: pd.DataFrame,
    config: BenterConfig = BenterConfig(),
    market_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add a `combined_predicted_points` column to a copy of predictions.

    `predictions` must have columns: player_id, country, predicted_points.
    `market_snapshots` is the output of prediction_markets.load_history;
    if None, it's loaded automatically.
    """
    out = predictions.copy()
    if market_snapshots is None:
        market_snapshots = load_history()
    adjustments = country_market_adjustments(market_snapshots)

    out["country_market_adjustment"] = (
        out["country"].map(adjustments).fillna(1.0)
    )
    model_pred = out["predicted_points"].astype(float).to_numpy()
    adj = out["country_market_adjustment"].astype(float).to_numpy()
    out["combined_predicted_points"] = (
        config.beta_0
        + config.beta_1 * model_pred * (1.0 + config.beta_2 * (adj - 1.0))
    ).clip(min=0)
    return out


def with_vs_without_summary(
    predictions: pd.DataFrame,
    realised: pd.DataFrame | None = None,
    config: BenterConfig = BenterConfig(),
) -> pd.DataFrame:
    """Side-by-side comparison rows.

    Returns the input predictions with two new columns:
      - `combined_predicted_points` (model + market via combiner)
      - `delta` (combined minus model)
    Plus, when `realised` is given (with realised_points column),
      - `model_error` and `combined_error` for both.
    """
    combined = combine(predictions, config=config)
    combined["delta"] = (combined["combined_predicted_points"]
                        - combined["predicted_points"])
    if realised is not None and "realised_points" in realised.columns:
        combined = combined.merge(
            realised[["player_id", "realised_points"]],
            on="player_id", how="left",
        )
        combined["model_error"] = (
            combined["predicted_points"] - combined["realised_points"]
        )
        combined["combined_error"] = (
            combined["combined_predicted_points"] - combined["realised_points"]
        )
    return combined
