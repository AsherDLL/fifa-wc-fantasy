"""The model registry must stay consistent with the artifacts it cites."""

from __future__ import annotations

import json
from pathlib import Path

from fifa_fantasy.report.data import KNOWN_BACKENDS
from fifa_fantasy.report.registry import (
    ALLOWED_STATUSES,
    GENERATIONS,
    MODEL_REGISTRY,
    NEGATIVE_RESULTS,
    TIMELINE,
    registry_by_key,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_RECORDS = MODEL_REGISTRY + GENERATIONS


def test_statuses_restricted():
    for record in ALL_RECORDS:
        assert record.status in ALLOWED_STATUSES, record.key


def test_every_backtest_backend_has_a_record():
    keys = {r.key for r in ALL_RECORDS}
    for backend in KNOWN_BACKENDS:
        assert backend in keys, backend


def test_tab_records_have_formulas():
    for record in MODEL_REGISTRY:
        assert record.formula_text.strip(), record.key
        assert record.formula_mathtext, record.key


def test_evidence_paths_exist():
    for record in ALL_RECORDS:
        for rel in record.evidence + record.whitepaper_refs:
            assert (REPO_ROOT / rel).exists(), f"{record.key}: {rel}"
    for nr in NEGATIVE_RESULTS:
        for rel in nr.evidence + nr.whitepaper_refs:
            assert (REPO_ROOT / rel).exists(), f"{nr.key}: {rel}"


def test_backtest_agrees_with_registry_statuses():
    """Backends the daemon runs must be marked as in the production tick."""
    by_key = registry_by_key()
    for backend in ("heuristic", "poisson", "gbm", "ensemble"):
        assert by_key[backend].in_production_tick, backend
    assert not by_key["monte_carlo"].in_production_tick


def test_timeline_is_dated_and_ordered():
    dates = [d for d, _, _ in TIMELINE]
    assert dates == sorted(dates)
    for d, title, desc in TIMELINE:
        assert len(d) == 10 and d[4] == "-"
        assert title and desc


def test_negative_results_reference_measurements():
    for nr in NEGATIVE_RESULTS:
        assert nr.measured, nr.key
        assert nr.evidence, nr.key
