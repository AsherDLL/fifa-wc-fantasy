"""Tests for the Phase 4 optimizer."""

from __future__ import annotations

import pandas as pd
import pytest

from fifa_fantasy.collector.schemas import Stage
from fifa_fantasy.optimizer.pipeline import (
    aggregate_to_player,
    apply_scouting_bonus,
)
from fifa_fantasy.optimizer.solvers import (
    SQUAD_POSITION_COUNTS,
    SQUAD_SIZE,
    VALID_FORMATIONS,
    solve_lineup,
    solve_squad,
)
from fifa_fantasy.optimizer.stage_config import STAGE_CONFIGS
from fifa_fantasy.scoring import Position


# ---------------------------------------------------------------------------
# Stage configs (cross-check vs docs/Fantasy.md)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "stage, budget, cap",
    [
        (Stage.GROUP_MD1, 100.0, 3),
        (Stage.GROUP_MD2, 100.0, 3),
        (Stage.GROUP_MD3, 100.0, 3),
        (Stage.R32, 105.0, 3),
        (Stage.R16, 105.0, 4),
        (Stage.QF, 105.0, 5),
        (Stage.SF, 105.0, 6),
        (Stage.FINAL, 105.0, 8),
    ],
)
def test_stage_config_matches_official_rules(stage, budget, cap):
    cfg = STAGE_CONFIGS[stage]
    assert cfg.budget_millions == budget
    assert cfg.max_per_country == cap


@pytest.mark.parametrize(
    "stage, transfers",
    [
        (Stage.GROUP_MD1, None),
        (Stage.GROUP_MD2, 2),
        (Stage.GROUP_MD3, 2),
        (Stage.R32, None),
        (Stage.R16, 4),
        (Stage.QF, 4),
        (Stage.SF, 5),
        (Stage.FINAL, 6),
    ],
)
def test_stage_config_free_transfers(stage, transfers):
    assert STAGE_CONFIGS[stage].free_transfers == transfers


# ---------------------------------------------------------------------------
# Scouting bonus injection
# ---------------------------------------------------------------------------

def test_scouting_bonus_triggers_when_both_thresholds_met():
    df = pd.DataFrame([
        {"player_id": 1, "predicted_points": 6.0, "ownership_fraction": 0.01},
    ])
    out = apply_scouting_bonus(df)
    assert out["scouting_bonus"].iloc[0] == 2
    assert out["effective_points"].iloc[0] == pytest.approx(8.0)


@pytest.mark.parametrize("pred, own", [(4.0, 0.01), (6.0, 0.05), (4.0, 0.05)])
def test_scouting_bonus_strict_thresholds(pred, own):
    df = pd.DataFrame([
        {"player_id": 1, "predicted_points": pred, "ownership_fraction": own},
    ])
    out = apply_scouting_bonus(df)
    assert out["scouting_bonus"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregate_sums_in_scope_rounds():
    df = pd.DataFrame([
        {"player_id": 1, "round_id": 1, "predicted_points": 5.0,
         "ownership_fraction": 0.5, "full_name": "A",
         "position": "FWD", "country": "X", "country_abbr": "X",
         "squad_id": 1, "price_millions": 8.0, "status": "playing",
         "is_eliminated": False},
        {"player_id": 1, "round_id": 2, "predicted_points": 7.0,
         "ownership_fraction": 0.5, "full_name": "A",
         "position": "FWD", "country": "X", "country_abbr": "X",
         "squad_id": 1, "price_millions": 8.0, "status": "playing",
         "is_eliminated": False},
        {"player_id": 1, "round_id": 3, "predicted_points": 4.0,
         "ownership_fraction": 0.5, "full_name": "A",
         "position": "FWD", "country": "X", "country_abbr": "X",
         "squad_id": 1, "price_millions": 8.0, "status": "playing",
         "is_eliminated": False},
    ])
    out = aggregate_to_player(apply_scouting_bonus(df), rounds=[1, 2])
    assert len(out) == 1
    assert out["total_effective_points"].iloc[0] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Squad solver
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_pool() -> pd.DataFrame:
    """A small pool with enough players to satisfy 2/5/5/3 + country caps."""
    rows = []
    pid = 0
    # 8 countries × 4 positions × 2 players each at varied prices/points.
    for country in [f"C{i}" for i in range(8)]:
        for position in ["GK", "DEF", "MID", "FWD"]:
            for variant in range(2):
                pid += 1
                rows.append({
                    "player_id": pid,
                    "full_name": f"{country}-{position}-{variant}",
                    "position": position,
                    "country": country,
                    "country_abbr": country,
                    "squad_id": int(country[1]) + 1,
                    "price_millions": 4.0 + variant + (1 if position == "FWD" else 0),
                    "total_effective_points": 5.0 + variant + (1 if position == "FWD" else 0),
                    "total_predicted_points": 5.0 + variant + (1 if position == "FWD" else 0),
                    "is_eliminated": False,
                    "ownership_fraction": 0.10,
                    "status": "playing",
                })
    return pd.DataFrame(rows)


def test_solve_squad_picks_exactly_fifteen(fake_pool):
    cfg = STAGE_CONFIGS[Stage.GROUP_MD1]
    solution = solve_squad(fake_pool, cfg)
    assert len(solution.player_ids) == SQUAD_SIZE


def test_solve_squad_respects_position_counts(fake_pool):
    cfg = STAGE_CONFIGS[Stage.GROUP_MD1]
    solution = solve_squad(fake_pool, cfg)
    chosen = fake_pool[fake_pool["player_id"].isin(solution.player_ids)]
    counts = chosen["position"].value_counts().to_dict()
    for pos, n in SQUAD_POSITION_COUNTS.items():
        assert counts[pos.value] == n


def test_solve_squad_respects_budget(fake_pool):
    cfg = STAGE_CONFIGS[Stage.GROUP_MD1]
    solution = solve_squad(fake_pool, cfg)
    assert solution.budget_used <= cfg.budget_millions + 1e-6


def test_solve_squad_respects_nationality_cap(fake_pool):
    cfg = STAGE_CONFIGS[Stage.GROUP_MD1]  # cap = 3
    solution = solve_squad(fake_pool, cfg)
    chosen = fake_pool[fake_pool["player_id"].isin(solution.player_ids)]
    per_country = chosen["country"].value_counts()
    assert (per_country <= cfg.max_per_country).all()


def test_solve_squad_excludes_eliminated(fake_pool):
    fake_pool.loc[fake_pool["country"] == "C0", "is_eliminated"] = True
    cfg = STAGE_CONFIGS[Stage.GROUP_MD1]
    solution = solve_squad(fake_pool, cfg)
    chosen = fake_pool[fake_pool["player_id"].isin(solution.player_ids)]
    assert "C0" not in chosen["country"].values


# ---------------------------------------------------------------------------
# Lineup solver
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_squad() -> pd.DataFrame:
    """A 15-player squad with varied predicted points so the captain pick is unambiguous."""
    rows = []
    counts = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    pid = 0
    for position, n in counts.items():
        for i in range(n):
            pid += 1
            rows.append({
                "player_id": pid,
                "full_name": f"{position}-{i}",
                "position": position,
                "predicted_points": 1.0 + i + (10.0 if position == "FWD" and i == 0 else 0),
            })
    return pd.DataFrame(rows)


def test_solve_lineup_picks_eleven_starters(fake_squad):
    sol = solve_lineup(fake_squad)
    assert len(sol.starter_ids) == 11
    assert len(sol.bench_ids) == 4


def test_solve_lineup_uses_valid_formation(fake_squad):
    sol = solve_lineup(fake_squad)
    assert sol.formation in VALID_FORMATIONS


def test_solve_lineup_has_one_gk(fake_squad):
    sol = solve_lineup(fake_squad)
    starters = fake_squad[fake_squad["player_id"].isin(sol.starter_ids)]
    assert (starters["position"] == "GK").sum() == 1


def test_solve_lineup_outfield_matches_formation(fake_squad):
    sol = solve_lineup(fake_squad)
    starters = fake_squad[fake_squad["player_id"].isin(sol.starter_ids)]
    expected = VALID_FORMATIONS[sol.formation]
    assert (starters["position"] == "DEF").sum() == expected[0]
    assert (starters["position"] == "MID").sum() == expected[1]
    assert (starters["position"] == "FWD").sum() == expected[2]


def test_solve_lineup_captain_is_highest_predicted_starter(fake_squad):
    sol = solve_lineup(fake_squad)
    starters = fake_squad[fake_squad["player_id"].isin(sol.starter_ids)]
    captain_pred = fake_squad.loc[
        fake_squad["player_id"] == sol.captain_id, "predicted_points"
    ].iloc[0]
    assert captain_pred == starters["predicted_points"].max()


def test_solve_lineup_bench_outfield_priority_by_pred(fake_squad):
    sol = solve_lineup(fake_squad)
    bench = fake_squad.set_index("player_id").loc[sol.bench_ids]
    outfield = bench[bench["position"] != "GK"]["predicted_points"].to_list()
    assert outfield == sorted(outfield, reverse=True)
    # GK on bench is last.
    assert bench.iloc[-1]["position"] == "GK"
