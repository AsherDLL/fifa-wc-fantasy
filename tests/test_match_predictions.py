"""Hermetic tests for the pure functions of scripts/match_predictions.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import match_predictions as mp


def _toy_matches() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    teams = [1, 2, 3, 4]
    for r in range(1, 5):
        for h, a in ((1, 2), (3, 4)):
            rows.append({
                "home_squad_id": h, "away_squad_id": a,
                "home_score_ext": int(rng.poisson(1.5)),
                "away_score_ext": int(rng.poisson(1.0)),
                "home_xg": 1.4, "away_xg": 1.1,
                "exposure": 1.0, "round_id": r,
            })
    return pd.DataFrame(rows)


def test_dc_gradient_matches_numeric():
    d = _toy_matches()
    teams = sorted(set(d["home_squad_id"]) | set(d["away_squad_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    args = (d["home_squad_id"].map(idx).to_numpy(),
            d["away_squad_id"].map(idx).to_numpy(),
            d["home_score_ext"].to_numpy(dtype=float),
            d["away_score_ext"].to_numpy(dtype=float),
            d["exposure"].to_numpy(dtype=float),
            np.ones(len(d)), n, 1.0)
    rng = np.random.default_rng(7)
    theta = rng.normal(0, 0.2, size=2 + 2 * n)
    theta[1] = 0.05
    err = optimize.check_grad(
        lambda th: mp.dc_nll_grad(th, *args)[0],
        lambda th: mp.dc_nll_grad(th, *args)[1], theta)
    assert err < 1e-4


def test_score_grid_normalizes_and_1x2_sums():
    g = mp.score_grid(1.7, 0.9, rho=-0.1)
    assert abs(g.sum() - 1.0) < 1e-9
    p = mp.grid_1x2(g)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[0] > p[2]  # stronger home rate


def test_rps_known_values():
    assert mp.rps(np.array([1.0, 0.0, 0.0]), 0) == 0.0
    # uniform vs home win: ((1/3-1)^2 + (2/3-1)^2)/2 = (4/9+1/9)/2
    assert abs(mp.rps(np.array([1 / 3, 1 / 3, 1 / 3]), 0) - 5 / 18) < 1e-12


def test_p_advance_bounds():
    p = mp.p_advance_from(1.6, 0.8)
    assert 0.5 < p < 1.0
    assert abs(mp.p_advance_from(1.0, 1.0) - 0.5) < 1e-9


def test_simplex_grid_and_weight_cap():
    grid = mp.simplex_grid(3, 0.25)
    assert np.allclose(grid.sum(axis=1), 1.0)
    preds = []
    rng = np.random.default_rng(0)
    for i in range(40):
        # component 0 is perfect, others are uniform
        outcome = int(rng.integers(3))
        perfect = np.zeros(3)
        perfect[outcome] = 1.0
        preds.append({
            "round_id": 2 + i % 3,
            "r90": outcome,
            "dixon_coles_1x2": 0.9 * perfect + 0.1 / 3,
            "xg_poisson_1x2": np.full(3, 1 / 3),
            "elo_1x2": np.full(3, 1 / 3),
        })
    w, s = mp.fit_weights(preds, "1x2", log_pool=False)
    assert w[0] == 1.0 and s < 0.02


def test_fit_xg_shrink_moderates():
    d = _toy_matches()
    sharp = mp.fit_xg(d, xi=1.0, shrink=1.0)
    flat = mp.fit_xg(d, xi=1.0, shrink=0.4)
    lh_s, la_s = sharp.lambdas(1, 2)
    lh_f, la_f = flat.lambdas(1, 2)
    assert abs(lh_f - flat.mu) <= abs(lh_s - sharp.mu) + 1e-9
