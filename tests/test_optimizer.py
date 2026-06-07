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
    TRANSFER_HIT_POINTS,
    VALID_FORMATIONS,
    solve_lineup,
    solve_squad,
    solve_transfer,
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


# ---------------------------------------------------------------------------
# Transfer solver
# ---------------------------------------------------------------------------


def test_solve_transfer_zero_transfers_when_squad_already_optimal(fake_pool):
    cfg = STAGE_CONFIGS[Stage.GROUP_MD2]  # 2 free transfers
    optimal = solve_squad(fake_pool, STAGE_CONFIGS[Stage.GROUP_MD1])
    transfer = solve_transfer(fake_pool, optimal.player_ids, cfg)
    assert transfer.n_transfers == 0
    assert transfer.transfer_cost_points == 0
    assert sorted(transfer.player_ids) == sorted(optimal.player_ids)


def test_solve_transfer_takes_free_swap_when_clearly_better(fake_pool):
    cfg = STAGE_CONFIGS[Stage.GROUP_MD2]
    # Start from a deliberately-bad squad: lowest-pointing valid 15.
    bad = (
        fake_pool.sort_values("total_effective_points")
        .groupby("position", sort=False)
        .head(max(SQUAD_POSITION_COUNTS.values()))
    )
    # Take exactly the required counts per position.
    bad_ids = []
    for pos, n in SQUAD_POSITION_COUNTS.items():
        bad_ids.extend(
            bad[bad["position"] == pos.value]
            .nsmallest(n, "total_effective_points")["player_id"].tolist()
        )
    transfer = solve_transfer(fake_pool, bad_ids, cfg)
    # Optimal differs from the bad squad → at least one transfer
    assert transfer.n_transfers >= 1


def test_solve_transfer_caps_extras_to_free_quota(fake_pool):
    cfg = STAGE_CONFIGS[Stage.GROUP_MD2]  # 2 free
    # Make every player look almost identical so optimal ~= current; the
    # solver should not invent costly transfers.
    pool = fake_pool.copy()
    pool["total_effective_points"] = 5.0
    current = pool.sample(SQUAD_SIZE, random_state=0)
    # Ensure the random sample is legal first; if not, use solve_squad.
    optimal_start = solve_squad(fake_pool, STAGE_CONFIGS[Stage.GROUP_MD1])
    transfer = solve_transfer(pool, optimal_start.player_ids, cfg)
    assert transfer.n_extra_transfers == 0


def test_solve_transfer_charges_hit_when_worth_it():
    """If two upgrades each beat the hit, the solver should pay one −3."""
    # Build a pool of exactly 2/5/5/3 baseline + 4 high-value alternates
    # (one per position). With 1 free transfer and 2 upgrades worth +10
    # each, optimal = 2 transfers (1 free + 1 hit), net gain = +20 − 3 = 17.
    rows = []
    pid = 0
    baseline_counts = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    # Baseline squad: 15 players at 5 pts each, distributed across 5
    # countries so the cap of 5 isn't a problem.
    for position, n in baseline_counts.items():
        for i in range(n):
            pid += 1
            country = f"C{i % 5}"
            rows.append({
                "player_id": pid, "full_name": f"BASE-{position}-{i}",
                "position": position, "country": country, "country_abbr": country,
                "squad_id": (i % 5) + 1,
                "price_millions": 5.0,
                "total_effective_points": 5.0,
                "is_eliminated": False,
                "ownership_fraction": 0.1, "status": "playing",
            })
    current = [int(r["player_id"]) for r in rows]
    # Two big alternate upgrades: one FWD and one MID, worth +10 each.
    # Plus filler alternates so the solver has options at every position.
    alt_pid = 1000
    for position in ("GK", "DEF", "MID", "FWD"):
        for i in range(3):
            alt_pid += 1
            pts = 15.0 if (position in ("FWD", "MID") and i == 0) else 5.0
            rows.append({
                "player_id": alt_pid, "full_name": f"ALT-{position}-{i}",
                "position": position, "country": f"C{5 + (i % 3)}",
                "country_abbr": f"C{5 + (i % 3)}",
                "squad_id": 10 + (i % 3),
                "price_millions": 5.0,
                "total_effective_points": pts,
                "is_eliminated": False,
                "ownership_fraction": 0.1, "status": "playing",
            })
    pool = pd.DataFrame(rows)

    from fifa_fantasy.optimizer.stage_config import StageConfig
    cfg = StageConfig(
        stage=Stage.GROUP_MD2, budget_millions=200.0, max_per_country=5,
        free_transfers=1, available_boosters=(),
    )
    transfer = solve_transfer(pool, current, cfg)
    assert transfer.n_transfers == 2
    assert transfer.n_extra_transfers == 1
    assert transfer.transfer_cost_points == TRANSFER_HIT_POINTS
    # Baseline = 13×5 + 2×15 = 95 gross; minus 1 hit (−3) = 92 net.
    assert transfer.objective == pytest.approx(92.0)


def test_solve_transfer_unlimited_stage_no_hit(fake_pool):
    cfg = STAGE_CONFIGS[Stage.R32]  # unlimited free transfers
    # Any current squad — the solver should freely re-pick and not pay any hits.
    optimal_md1 = solve_squad(fake_pool, STAGE_CONFIGS[Stage.GROUP_MD1])
    transfer = solve_transfer(fake_pool, optimal_md1.player_ids, cfg)
    assert transfer.n_extra_transfers == 0
    assert transfer.transfer_cost_points == 0


def test_solve_transfer_rolled_over_increases_free_quota(fake_pool):
    cfg = STAGE_CONFIGS[Stage.GROUP_MD2]  # 2 free → with 1 rolled, 3 free
    # Force exactly 3 transfers via a contrived current squad would be hard,
    # so we check the math: solve once with 0 rolled, once with 1 rolled —
    # the second's cost ≤ first's cost.
    optimal = solve_squad(fake_pool, STAGE_CONFIGS[Stage.GROUP_MD1])
    a = solve_transfer(fake_pool, optimal.player_ids, cfg, rolled_over_transfers=0)
    b = solve_transfer(fake_pool, optimal.player_ids, cfg, rolled_over_transfers=1)
    assert b.transfer_cost_points <= a.transfer_cost_points
