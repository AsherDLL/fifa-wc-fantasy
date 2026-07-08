"""Per-position ensemble backend: route each position to the backend that
wins it on held-out labels.

The README and Section 07 validation table have said since MD1 that no
single backend is best everywhere: on the EPL held-out set Poisson wins
goalkeepers, the heuristic wins defenders, and the GBM wins midfielders
and forwards. Yet the runnable backends were all monolithic - each one
scored every position with its own formula, including the positions it
loses. This backend finally acts on the validation result: it runs the
component backends and, per position, keeps the prediction from the
routed winner.

Routing is data-driven with a documented default. `DEFAULT_ROUTING`
encodes the EPL held-out winners. Pass a different mapping to override,
or call `routing_from_report()` to derive it from a validation report
JSON (whichever backend has the lowest RMSE per position).

The GBM's quantile columns (predicted_q10/q50/q90) are carried through
for any position routed to the GBM; positions routed elsewhere get NaN
quantiles, since the heuristic and Poisson backends are point estimates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import heuristic_predict
from .gbm import DEFAULT_MODELS_DIR, load_models, predict as gbm_predict
from .poisson import poisson_predict

# Position -> backend name. Defaults are the EPL 2024-25 GW30-38 held-out
# RMSE winners (Section 07). The WC walk-forward validation
# (scripts/wc_forward_validation.py) shows the WC-retrained GBM competitive
# at every position, but the structural backends (heuristic, Poisson) are
# immune to EPL->WC distribution shift, so the default keeps them where the
# canonical benchmark says they win.
DEFAULT_ROUTING = {
    "GK": "poisson",
    "DEF": "heuristic",
    "MID": "gbm",
    "FWD": "gbm",
}

_QUANTILE_COLS = ("predicted_q10", "predicted_q50", "predicted_q90")


def routing_from_report(report_path: Path) -> dict[str, str]:
    """Derive per-position routing from a validation_report.json.

    Picks, for each position, the backend column with the smallest RMSE.
    Falls back to DEFAULT_ROUTING for any position missing from the report.
    """
    routing = dict(DEFAULT_ROUTING)
    try:
        payload = json.loads(Path(report_path).read_text())
    except (OSError, json.JSONDecodeError):
        return routing
    for row in payload.get("position_rmse", []):
        pos = row.get("position")
        candidates = {
            "heuristic": row.get("heuristic_rmse"),
            "poisson": row.get("poisson_rmse"),
            "gbm": row.get("gbm_rmse"),
        }
        candidates = {k: v for k, v in candidates.items() if v is not None}
        if pos and candidates:
            routing[pos] = min(candidates, key=candidates.get)
    return routing


def ensemble_predict(
    features: pd.DataFrame,
    routing: dict[str, str] | None = None,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> pd.DataFrame:
    """Return `features` with `predicted_points` routed per position.

    Runs only the backends the routing actually needs. Each component
    backend already zeroes non-playing / eliminated / benched rows, so the
    routed result inherits that behaviour without extra handling.
    """
    routing = routing or DEFAULT_ROUTING
    needed = set(routing.values())
    out = features.copy()

    preds: dict[str, pd.DataFrame] = {}
    if "heuristic" in needed:
        preds["heuristic"] = heuristic_predict(features)
    if "poisson" in needed:
        preds["poisson"] = poisson_predict(features)
    if "gbm" in needed:
        preds["gbm"] = gbm_predict(features, load_models(models_dir))

    result = np.zeros(len(out), dtype=float)
    positions = out["position"].astype(str).to_numpy()
    for pos, backend in routing.items():
        mask = positions == pos
        if not mask.any():
            continue
        src = preds.get(backend)
        if src is None:
            raise ValueError(f"routing references unavailable backend {backend!r}")
        result[mask] = src["predicted_points"].to_numpy()[mask]
    out["predicted_points"] = result

    # Carry GBM quantiles where the position is routed to the GBM; NaN
    # elsewhere (the structural backends do not emit quantiles).
    gbm_src = preds.get("gbm")
    for col in _QUANTILE_COLS:
        vals = np.full(len(out), np.nan)
        if gbm_src is not None and col in gbm_src.columns:
            gcol = gbm_src[col].to_numpy()
            for pos, backend in routing.items():
                if backend != "gbm":
                    continue
                mask = positions == pos
                vals[mask] = gcol[mask]
        out[col] = vals

    # Record which backend produced each row, for auditability.
    routed = np.array([routing.get(p, "") for p in positions], dtype=object)
    out["routed_backend"] = routed
    return out
