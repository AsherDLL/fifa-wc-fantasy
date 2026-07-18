"""Render every dashboard page against a synthetic results directory."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fifa_fantasy.web.render import build_all


def _player(pid: int, position: str, role: str = "", starter: bool = True,
            bench_priority: int = 99) -> dict:
    return {
        "player_id": pid,
        "full_name": f"Player {pid}",
        "first_name": "P", "last_name": f"L{pid}", "known_name": None,
        "position": position, "country": "Argentina", "country_abbr": "ARG",
        "price_millions": 8.0, "ownership_fraction": 0.4, "status": "playing",
        "is_eliminated": False, "one_to_watch": False,
        "one_to_watch_text": None, "form": 5.0, "role": role,
        "in_starting_xi": starter, "bench_priority": bench_priority,
        "opponent_abbr": "FRA", "is_home": True,
        "predicted_points": 5.5, "effective_points": 5.5,
        "predicted_q10": 1.0, "predicted_q50": 5.0, "predicted_q90": 11.0,
    }


def _rec(stage: str, backend: str, generated: str) -> dict:
    squad = [
        _player(1, "GK"),
        _player(2, "DEF"),
        _player(3, "MID", role="Vice"),
        _player(4, "FWD", role="Captain"),
        _player(5, "MID", starter=False, bench_priority=1),
    ]
    return {
        "stage": stage, "model_backend": backend, "model_version": "v3form",
        "host": "test", "generated_at_utc": generated, "horizon_rounds": 1,
        "budget_used": 100.0, "budget_total": 105.0,
        "total_horizon_points": 80.0, "net_horizon_points": 80.0,
        "squad_player_ids": [1, 2, 3, 4, 5],
        "squad": squad,
        "lineup": {"round_id": 6, "formation": "3-4-3",
                   "starter_ids": [1, 2, 3, 4],
                   "bench_ids_priority_order": [5],
                   "captain_id": 4, "vice_captain_id": 3,
                   "expected_points": 55.0},
    }


TINY_SVG = ('<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
            'width="10" height="10"><rect width="10" height="10"/></svg>')


@pytest.fixture()
def results_dir(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "a_recommendation_ensemble_QF_1.json").write_text(
        json.dumps(_rec("QF", "ensemble", "2026-07-09T00:00:00+00:00")))
    # A newer run of another backend must not displace the official pick.
    (results / "a_recommendation_gbm_QF_2.json").write_text(
        json.dumps(_rec("QF", "gbm", "2026-07-10T00:00:00+00:00")))

    report_dir = results / "report"
    report_dir.mkdir()
    (report_dir / "report_data.json").write_text(json.dumps({
        "backtest": {
            "rounds": [{"round_id": 1, "stage": "GROUP_MD1",
                        "stage_label": "Group Matchday 1"}],
            "series": {"heuristic": [10.0], "gbm": [12.0],
                       "ensemble": [11.0], "user_actual": [5.0]},
            "cumulative": {"heuristic": [10.0], "gbm": [12.0],
                           "ensemble": [11.0], "user_actual": [5.0]},
            "totals": {"user_actual_net": 5},
        },
        "validation": [{"position": "GK", "n": 10, "heuristic_rmse": 2.0,
                        "poisson_rmse": 1.5, "gbm_rmse": 1.8}],
        "routing": {"GK": "poisson", "DEF": "heuristic",
                    "MID": "gbm", "FWD": "gbm"},
        "walkforward": {"generated_at_utc": "2026-07-09T00:00:00Z",
                        "holdout_rounds": [2, 3]},
        "calibration_summary": [
            {"model_backend": "ensemble", "position": "FWD", "n": 7,
             "rmse": 2.0, "mae": 1.5, "spearman_rho": 0.5}],
        "signals_coverage": {"articles_cached": 100, "signal_rows": 0,
                             "players_flagged": 0, "by_class": {},
                             "by_signal": {}},
        "market_latest": [{"country": "Spain", "implied_prob": 0.2,
                           "volume_24h": 1000.0}],
    }))

    figures_dir = results / "figures"
    figures_dir.mkdir()
    for name in ("fig_backtest_cumulative", "formula_heuristic",
                 "fig_market_odds"):
        (figures_dir / f"{name}.svg").write_text(TINY_SVG)
    return results


@pytest.fixture()
def pages(results_dir):
    return build_all(results_dir)


def test_all_four_pages_render(pages):
    assert set(pages) == {"index.html", "algorithms.html",
                          "intelligence.html", "research.html"}
    for html in pages.values():
        assert "{{" not in html and "{%" not in html


def test_official_banner_names_the_ensemble(pages):
    html = pages["index.html"]
    assert "Official squad" in html
    # The newer gbm run must not be featured; the ensemble file is linked.
    assert "a_recommendation_ensemble_QF_1.json" in html
    assert "official backend <b>ensemble</b>" in html
    assert "Caution" not in html


def test_overview_falls_back_with_caution(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "a_recommendation_gbm_QF_1.json").write_text(
        json.dumps(_rec("QF", "gbm", "2026-07-10T00:00:00+00:00")))
    html = build_all(results)["index.html"]
    assert "Caution" in html


def test_algorithms_page_has_all_tabs_and_formula(pages):
    html = pages["algorithms.html"]
    for label in ("Heuristic", "Poisson", "GBM", "Monte Carlo", "Ensemble"):
        assert f">{label}</label>" in html
    # The heuristic formula SVG fixture is inlined; plain-text fallback too.
    assert "<svg" in html
    assert "predicted = c[pos] * price" in html
    # The experimental backend explains itself instead of showing a squad.
    assert "not run by the\n      production tick" in html.replace("\r", "") \
        or "not run by the" in html


def test_intelligence_page_has_filters_and_market(pages):
    html = pages["intelligence.html"]
    assert 'id="sig-search"' in html
    assert "data-class" in html
    assert "Spain" in html
    assert "articles cached <b>100</b>" in html


def test_research_page_has_ledger_and_caveat(pages):
    html = pages["research.html"]
    assert "Negative results" in html
    assert "Benter" in html
    assert "In-sample" in html or "in-sample" in html
    assert "Walk-forward" in html or "walk-forward" in html
    assert "Match forecasting" in html


def test_pages_are_self_contained(pages):
    # No external stylesheets, scripts, fonts or images. External hrefs to
    # article URLs are content links and only appear when signals exist,
    # which this fixture does not provide.
    for name, html in pages.items():
        assert "<link" not in html, name
        assert not re.search(r'src="https?://', html), name
        assert not re.search(r"@import", html), name
