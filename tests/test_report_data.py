"""Unit tests for the report reconciliation loaders (report.data)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fifa_fantasy.report import data as rd


def _write_rec(results_dir: Path, name: str, stage: str, backend: str,
               generated: str, squad_ids: list[int] | None = None) -> None:
    payload = {
        "stage": stage,
        "model_backend": backend,
        "model_version": "",
        "generated_at_utc": generated,
        "squad_player_ids": squad_ids or [1, 2, 3],
        "squad": [],
        "lineup": {"formation": "3-4-3", "expected_points": 50.0},
        "budget_used": 100.0,
        "budget_total": 105.0,
        "total_horizon_points": 80.0,
    }
    (results_dir / name).write_text(json.dumps(payload))


class TestOfficialRecommendation:
    def test_prefers_official_backend_over_newer_other(self, tmp_path):
        _write_rec(tmp_path, "a_recommendation_ensemble_QF_1.json",
                   "QF", "ensemble", "2026-07-09T00:00:00+00:00")
        _write_rec(tmp_path, "a_recommendation_gbm_QF_2.json",
                   "QF", "gbm", "2026-07-10T00:00:00+00:00")
        rec, fallback = rd.official_recommendation(tmp_path, backend="ensemble")
        assert rec["model_backend"] == "ensemble"
        assert fallback is False

    def test_falls_back_with_flag_when_absent_at_top_stage(self, tmp_path):
        # The ensemble run exists only for an earlier stage.
        _write_rec(tmp_path, "a_recommendation_ensemble_R16_1.json",
                   "R16", "ensemble", "2026-07-05T00:00:00+00:00")
        _write_rec(tmp_path, "a_recommendation_gbm_QF_2.json",
                   "QF", "gbm", "2026-07-10T00:00:00+00:00")
        rec, fallback = rd.official_recommendation(tmp_path, backend="ensemble")
        assert rec["model_backend"] == "gbm"
        assert rec["stage"] == "QF"
        assert fallback is True

    def test_none_when_no_recommendations(self, tmp_path):
        rec, fallback = rd.official_recommendation(tmp_path)
        assert rec is None
        assert fallback is False

    def test_latest_per_backend_restricted_to_top_stage(self, tmp_path):
        _write_rec(tmp_path, "a_recommendation_ensemble_QF_1.json",
                   "QF", "ensemble", "2026-07-09T00:00:00+00:00")
        _write_rec(tmp_path, "a_recommendation_ensemble_QF_2.json",
                   "QF", "ensemble", "2026-07-10T00:00:00+00:00")
        _write_rec(tmp_path, "a_recommendation_poisson_R16_3.json",
                   "R16", "poisson", "2026-07-10T00:00:00+00:00")
        per = rd.latest_recommendation_per_backend(tmp_path)
        assert set(per) == {"ensemble"}
        assert per["ensemble"]["generated_at_utc"].startswith("2026-07-10")


class TestLoadBacktest:
    def test_cumulative_math(self, tmp_path):
        payload = {
            "rounds": [
                {"round_id": 1, "stage": "GROUP_MD1",
                 "backends": [{"backend": "heuristic", "realised_total": 10}],
                 "user_actual": {"net_total": 5},
                 "random_baseline": {"mean": 2.0}},
                {"round_id": 2, "stage": "GROUP_MD2",
                 "backends": [{"backend": "heuristic", "realised_total": 7}],
                 "user_actual": {"net_total": 6},
                 "random_baseline": {"mean": 3.0}},
            ],
            "totals_by_source": {"heuristic": 17},
        }
        (tmp_path / "backtest_summary.json").write_text(json.dumps(payload))
        out = rd.load_backtest(tmp_path)
        assert out["series"]["heuristic"] == [10.0, 7.0]
        assert out["cumulative"]["heuristic"] == [10.0, 17.0]
        assert out["cumulative"]["user_actual"] == [5.0, 11.0]
        assert out["cumulative"]["random_baseline"] == [2.0, 5.0]

    def test_none_when_absent(self, tmp_path):
        assert rd.load_backtest(tmp_path) is None


class TestWalkforwardAndFriends:
    def test_walkforward_none_when_absent(self, tmp_path):
        assert rd.load_walkforward(tmp_path) is None

    def test_gk_sweep_reads_newest(self, tmp_path):
        (tmp_path / "gk_formula_ab_2026-06-01.json").write_text(
            json.dumps({"rows": [{"dataset": "old"}]}))
        (tmp_path / "gk_formula_ab_2026-06-29.json").write_text(
            json.dumps({"rows": [{"dataset": "new"}]}))
        rows = rd.load_gk_sweep(tmp_path)
        assert rows == [{"dataset": "new"}]


def _signal_row(player_id: int, url: str, signal: str,
                published: str = "2026-07-08T00:00:00+00:00") -> dict:
    return {
        "player_id": player_id, "full_name": f"P{player_id}",
        "country_abbr": "ARG", "position": "FWD", "signal": signal,
        "signal_class": "risk", "proximity_chars": 10,
        "evidence": "text", "article_title": "t", "source_id": "src",
        "source_confidence": 0.8, "published_at_utc": published,
        "collected_at_utc": published, "url": url,
    }


class TestSignals:
    def test_dedup_and_squad_join(self, tmp_path):
        rows = [
            _signal_row(1, "u1", "injury"),
            _signal_row(1, "u1", "injury"),      # exact duplicate
            _signal_row(1, "u2", "injury"),      # second article
            _signal_row(2, "u1", "suspension"),  # other player
        ]
        df = pd.DataFrame(rows)
        df.to_parquet(tmp_path / "signals_2026-07-08T00-00-00Z.parquet")
        df.to_parquet(tmp_path / "signals_2026-07-08T06-00-00Z.parquet")

        merged = rd.load_signals(tmp_path, days=3.0)
        # 4 rows duplicated across two snapshots collapse to 3 findings.
        assert len(merged) == 3

        agg = rd.aggregate_signals(merged)
        p1 = agg[agg["player_id"] == 1]
        assert len(p1) == 1
        assert int(p1.iloc[0]["mentions"]) == 2

        squad = rd.signals_for_squad(agg, [2])
        assert squad["player_id"].tolist() == [2]

    def test_window_excludes_old_snapshots(self, tmp_path):
        pd.DataFrame([_signal_row(1, "u1", "injury")]).to_parquet(
            tmp_path / "signals_2026-07-01T00-00-00Z.parquet")
        pd.DataFrame([_signal_row(2, "u2", "injury")]).to_parquet(
            tmp_path / "signals_2026-07-08T00-00-00Z.parquet")
        merged = rd.load_signals(tmp_path, days=3.0)
        assert merged["player_id"].tolist() == [2]

    def test_empty_when_no_files(self, tmp_path):
        assert rd.load_signals(tmp_path).empty


class TestMarketHistory:
    def test_parses_and_filters_null_prices(self, tmp_path):
        lines = [
            {"provider": "polymarket", "snapshot_utc": "2026-07-08T00-00-00Z",
             "title": "Will Spain win the 2026 FIFA World Cup?",
             "yes_price": 0.2, "no_price": 0.8, "volume_24h": 100.0},
            {"provider": "polymarket", "snapshot_utc": "2026-07-08T00-00-00Z",
             "title": "Will New Zealand win the 2026 FIFA World Cup?",
             "yes_price": None, "no_price": 1.0, "volume_24h": None},
            {"provider": "polymarket", "snapshot_utc": "2026-07-08T00-00-00Z",
             "title": "Unrelated contract", "yes_price": 0.5},
        ]
        with (tmp_path / "snapshot_2026-07-08T00-00-00Z.jsonl").open("w") as fh:
            for obj in lines:
                fh.write(json.dumps(obj) + "\n")
        hist = rd.load_market_history(tmp_path)
        assert hist["country"].tolist() == ["Spain"]
        table = rd.market_latest_table(hist)
        assert table[0]["country"] == "Spain"
        assert table[0]["implied_prob"] == pytest.approx(0.2)

    def test_empty_when_no_files(self, tmp_path):
        assert rd.load_market_history(tmp_path).empty


def _pred_frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "predicted_q10": 0.5, "predicted_q50": 2.0, "predicted_q90": 5.0,
        "model_backend": "gbm", "routed_backend": "",
        "round_points": None,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


class TestCalibration:
    def test_label_alignment_and_snapshot_precedence(self, tmp_path):
        # Older snapshot: round 2 still scheduled, prediction 3.0.
        _pred_frame([
            {"player_id": 1, "position": "FWD", "round_id": 2,
             "predicted_points": 3.0, "fixture_status": "scheduled"},
        ]).to_parquet(tmp_path / "predictions_2026-06-20.parquet")
        # Newer pre-match snapshot revises the prediction to 4.0; this one
        # must win. Round 3 is also scheduled here for a player whose team
        # then never played it (short round_points array).
        _pred_frame([
            {"player_id": 1, "position": "FWD", "round_id": 2,
             "predicted_points": 4.0, "fixture_status": "scheduled"},
            {"player_id": 2, "position": "MID", "round_id": 3,
             "predicted_points": 2.0, "fixture_status": "scheduled"},
        ]).to_parquet(tmp_path / "predictions_2026-06-21.parquet")
        # Newest snapshot: everything complete; labels live here. Player 1
        # scored 7 in round 2 (index 1). Player 2's array stops after round
        # 2, so the round-3 prediction has no label and must be dropped.
        _pred_frame([
            {"player_id": 1, "position": "FWD", "round_id": 2,
             "predicted_points": 9.9, "fixture_status": "complete",
             "round_points": [1, 7]},
            {"player_id": 2, "position": "MID", "round_id": 3,
             "predicted_points": 9.9, "fixture_status": "complete",
             "round_points": [0, 2]},
        ]).to_parquet(tmp_path / "predictions_2026-06-22.parquet")

        calib = rd.load_calibration(tmp_path)
        assert len(calib) == 1
        row = calib.iloc[0]
        assert row["player_id"] == 1
        # The latest pre-match snapshot wins, never the hindsight 9.9.
        assert row["predicted_points"] == pytest.approx(4.0)
        # Label is round_points[round_id - 1] from the newest file.
        assert row["realized"] == pytest.approx(7.0)

    def test_summary_shapes(self, tmp_path):
        _pred_frame([
            {"player_id": i, "position": "FWD", "round_id": 1,
             "predicted_points": float(i), "fixture_status": "scheduled"}
            for i in range(1, 6)
        ]).to_parquet(tmp_path / "predictions_2026-06-20.parquet")
        _pred_frame([
            {"player_id": i, "position": "FWD", "round_id": 1,
             "predicted_points": 0.0, "fixture_status": "complete",
             "round_points": [i + 1]}
            for i in range(1, 6)
        ]).to_parquet(tmp_path / "predictions_2026-06-21.parquet")
        calib = rd.load_calibration(tmp_path)
        summary = rd.calibration_summary(calib)
        assert len(summary) == 1
        assert summary[0]["position"] == "FWD"
        assert summary[0]["n"] == 5
        assert summary[0]["spearman_rho"] == pytest.approx(1.0)

    def test_empty_when_no_files(self, tmp_path):
        assert rd.load_calibration(tmp_path).empty
        assert rd.calibration_summary(pd.DataFrame()) == []
