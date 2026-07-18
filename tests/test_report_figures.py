"""Smoke tests: every figure builder writes valid SVG from tiny inputs."""

from __future__ import annotations

import pandas as pd
import pytest

from fifa_fantasy.report import figures
from fifa_fantasy.report.registry import MODEL_REGISTRY


def _assert_svg(path):
    assert path is not None
    text = path.read_text()
    assert "<svg" in text
    assert path.stat().st_size > 500


@pytest.fixture()
def backtest():
    return {
        "rounds": [{"round_id": 1, "stage": "GROUP_MD1"},
                   {"round_id": 2, "stage": "GROUP_MD2"}],
        "series": {"heuristic": [10.0, 7.0], "gbm": [12.0, 9.0],
                   "user_actual": [5.0, 6.0], "random_baseline": [2.0, 3.0]},
        "cumulative": {"heuristic": [10.0, 17.0], "gbm": [12.0, 21.0],
                       "user_actual": [5.0, 11.0],
                       "random_baseline": [2.0, 5.0]},
        "totals": {},
    }


def test_backtest_figures(tmp_path, backtest):
    _assert_svg(figures.fig_backtest_cumulative(backtest, tmp_path / "a.svg"))
    _assert_svg(figures.fig_backtest_rounds(backtest, tmp_path / "b.svg"))


def test_backtest_figures_none_without_data(tmp_path):
    assert figures.fig_backtest_cumulative(None, tmp_path / "a.svg") is None
    assert figures.fig_backtest_rounds({}, tmp_path / "b.svg") is None


def test_holdout_rmse(tmp_path):
    rows = [{"position": "GK", "n": 10, "heuristic_rmse": 2.0,
             "poisson_rmse": 1.5, "gbm_rmse": 1.8}]
    routing = {"GK": "poisson"}
    _assert_svg(figures.fig_holdout_rmse(rows, routing, tmp_path / "c.svg"))
    assert figures.fig_holdout_rmse([], routing, tmp_path / "d.svg") is None


def test_walkforward(tmp_path):
    payload = {"pooled": {
        "A_epl_noform": {"GK": 3.5, "DEF": 3.1, "MID": 3.0, "FWD": 3.2,
                         "ALL": 3.2},
        "C_eplwc_form": {"GK": 2.4, "DEF": 2.9, "MID": 2.6, "FWD": 2.8,
                         "ALL": 2.7},
    }}
    _assert_svg(figures.fig_walkforward(payload, tmp_path / "e.svg"))
    assert figures.fig_walkforward(None, tmp_path / "f.svg") is None


def test_calibration(tmp_path):
    calib = pd.DataFrame({
        "position": ["FWD"] * 12 + ["GK"] * 3,
        "predicted_points": [float(i % 5) for i in range(15)],
        "realized": [float((i + 1) % 6) for i in range(15)],
    })
    _assert_svg(figures.fig_calibration(calib, tmp_path / "g.svg"))
    assert figures.fig_calibration(pd.DataFrame(), tmp_path / "h.svg") is None


def test_gk_sweep(tmp_path):
    rows = [{"dataset": "epl_holdout", "position": "GK", "n": 10,
             "rmse_v1": 2.5, "rmse_v2": 2.4, "delta_rmse": -0.1}]
    _assert_svg(figures.fig_gk_sweep(rows, tmp_path / "i.svg"))
    assert figures.fig_gk_sweep([], tmp_path / "j.svg") is None


def test_market_negative(tmp_path):
    payload = {"rows": [
        {"round_id": 1, "backend": "heuristic", "delta_rmse": 0.3},
        {"round_id": 2, "backend": "heuristic", "delta_rmse": 0.2},
    ]}
    _assert_svg(figures.fig_market_negative(payload, tmp_path / "k.svg"))
    assert figures.fig_market_negative(None, tmp_path / "l.svg") is None


def test_market_odds(tmp_path):
    hist = pd.DataFrame({
        "country": ["Spain", "Spain", "France", "France"],
        "ts": pd.to_datetime(["2026-07-01", "2026-07-02"] * 2, utc=True),
        "implied_prob": [0.2, 0.25, 0.3, 0.28],
        "volume_24h": [100.0] * 4,
    })
    _assert_svg(figures.fig_market_odds(hist, tmp_path / "m.svg"))
    assert figures.fig_market_odds(pd.DataFrame(), tmp_path / "n.svg") is None


def test_signal_volume(tmp_path):
    import datetime as dt
    volume = pd.DataFrame(
        {"risk": [3, 5], "boost": [1, 0]},
        index=[dt.date(2026, 7, 7), dt.date(2026, 7, 8)])
    _assert_svg(figures.fig_signal_volume(volume, tmp_path / "o.svg"))
    assert figures.fig_signal_volume(pd.DataFrame(), tmp_path / "p.svg") is None


def test_formula_renderer_covers_registry(tmp_path):
    for record in MODEL_REGISTRY:
        path = figures.render_formula(
            record.formula_mathtext, tmp_path / f"formula_{record.key}.svg")
        _assert_svg(path)
    assert figures.render_formula((), tmp_path / "empty.svg") is None


def test_forecast_skill(tmp_path):
    forecast = {
        "skill_1x2": {"elo": {"rps": 0.148, "n": 78},
                      "dixon_coles": {"rps": 0.204, "n": 78}},
        "skill_advance": {"elo": {"log_loss": 0.44, "n": 30},
                          "market": {"log_loss": 0.53, "n": 29}},
        "baselines": {"uniform_rps": 0.237},
    }
    _assert_svg(figures.fig_forecast_skill(forecast, tmp_path / "q.svg"))
    assert figures.fig_forecast_skill(None, tmp_path / "r.svg") is None
    assert figures.fig_forecast_skill({}, tmp_path / "s.svg") is None


def test_final_probs(tmp_path):
    forecast = {"matches": [{
        "match": "Spain v Argentina", "kind": "final",
        "p_1x2_90min": {"home": 0.4, "draw": 0.25, "away": 0.35},
        "winner": {"home": "Spain", "p_home_lifts": 0.54},
        "uncertainty": {"p_advance_ci90": [0.53, 0.54]},
    }]}
    _assert_svg(figures.fig_final_probs(forecast, tmp_path / "t.svg"))
    assert figures.fig_final_probs(None, tmp_path / "u.svg") is None
    assert figures.fig_final_probs({"matches": []}, tmp_path / "v.svg") is None
