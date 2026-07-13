"""Tests for scripts/sf_joint_analysis.py (joint XI-aware round optimizer).

The script is loaded by path because scripts/ is not a package. Solver tests
run on a small synthetic pool; one integration test reads the committed
semifinal decision-point snapshot so the registered pick comparison stays
reproducible from a fresh clone.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "sf_joint_analysis", REPO_ROOT / "scripts" / "sf_joint_analysis.py")
sfa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sfa)

from fifa_fantasy.collector.schemas import Stage  # noqa: E402
from fifa_fantasy.optimizer.solvers import VALID_FORMATIONS  # noqa: E402
from fifa_fantasy.optimizer.stage_config import STAGE_CONFIGS  # noqa: E402

SF_CONFIG = STAGE_CONFIGS[Stage.SF]
TEAMS = ("AAA", "BBB", "CCC", "DDD")
MATCHES = [("AAA", "BBB"), ("CCC", "DDD")]


def synthetic_pool() -> pd.DataFrame:
    """24 players: per team 1 GK, 2 DEF, 2 MID, 1 FWD. Deterministic effs."""
    rows = []
    pid = 100
    p_adv = {"AAA": 0.6, "BBB": 0.4, "CCC": 0.55, "DDD": 0.45}
    for team in TEAMS:
        for pos, count in (("GK", 1), ("DEF", 2), ("MID", 2), ("FWD", 1)):
            for k in range(count):
                eff = 2.0 + (pid % 7) * 0.9
                rows.append({
                    "player_id": pid,
                    "full_name": f"{team} {pos}{k}",
                    "position": pos,
                    "country": team,
                    "country_abbr": team,
                    "price_millions": 4.0 + (pid % 5) * 0.8,
                    "eff": eff,
                    "eff_q90": eff * 1.8,
                    "scout_ev": 0.0,
                    "availability_factor": 1.0,
                    "p_adv": p_adv[team],
                    "p_concede_one": 0.35,
                    "cs_pts": {"GK": 5.0, "DEF": 5.0,
                               "MID": 1.0, "FWD": 0.0}[pos],
                    "s_q10": eff * 0.4,
                    "s_q50": eff,
                    "s_q90": eff * 1.8,
                    "ownership_fraction": 0.10,
                })
                pid += 1
    return pd.DataFrame(rows)


def synthetic_ctx() -> dict:
    lam = {"AAA": 1.5, "BBB": 1.1, "CCC": 1.3, "DDD": 1.2}
    opp = {}
    for a, b in MATCHES:
        opp[a], opp[b] = b, a
    p_adv = {"AAA": 0.6, "BBB": 0.4, "CCC": 0.55, "DDD": 0.45}
    return {"lam": lam, "matches": MATCHES, "opp": opp, "p_adv": p_adv}


def formation_of(pool: pd.DataFrame, xi: list[int]) -> tuple[int, int, int]:
    pos = pool["position"]
    return (sum(1 for i in xi if pos[i] == "DEF"),
            sum(1 for i in xi if pos[i] == "MID"),
            sum(1 for i in xi if pos[i] == "FWD"))


def test_from_scratch_solution_is_legal():
    pool = synthetic_pool()
    plan = sfa.solve_joint(pool, SF_CONFIG, current_ids=None, free=5)
    assert len(plan["squad"]) == 15
    assert len(plan["xi"]) == 11
    assert set(plan["xi"]) <= set(plan["squad"])
    assert plan["cap"] in plan["xi"]
    price = pool["price_millions"][plan["squad"]].sum()
    assert price <= SF_CONFIG.budget_millions + 1e-9
    counts = pool["position"][plan["squad"]].value_counts()
    assert counts["GK"] == 2 and counts["DEF"] == 5
    assert counts["MID"] == 5 and counts["FWD"] == 3
    assert formation_of(pool, plan["xi"]) in VALID_FORMATIONS.values()
    per_team = pool["country_abbr"][plan["squad"]].value_counts()
    assert (per_team <= SF_CONFIG.max_per_country).all()


def test_transfer_hit_accounting():
    pool = synthetic_pool()
    # weakest legal 15 as the current squad, so upgrades exist
    by_eff = pool.sort_values("eff")
    current = (by_eff[by_eff.position == "GK"].player_id[:2].tolist()
               + by_eff[by_eff.position == "DEF"].player_id[:5].tolist()
               + by_eff[by_eff.position == "MID"].player_id[:5].tolist()
               + by_eff[by_eff.position == "FWD"].player_id[:3].tolist())
    plan = sfa.solve_joint(pool, SF_CONFIG, current, free=1)
    assert plan["hits"] == 3.0 * max(0, plan["n_new"] - 1)
    generous = sfa.solve_joint(pool, SF_CONFIG, current, free=15)
    assert generous["hits"] == 0.0


def test_eliminated_players_are_retainable():
    pool = synthetic_pool()
    # two dead squad members: zero eff, zero availability
    dead = pd.DataFrame([
        {"player_id": 900, "full_name": "Dead DEF", "position": "DEF",
         "country": "XXX", "country_abbr": "XXX", "price_millions": 4.0},
        {"player_id": 901, "full_name": "Dead MID", "position": "MID",
         "country": "YYY", "country_abbr": "YYY", "price_millions": 4.0},
    ])
    for col in ("eff", "eff_q90", "scout_ev", "availability_factor", "p_adv",
                "p_concede_one", "cs_pts", "s_q10", "s_q50", "s_q90",
                "ownership_fraction"):
        dead[col] = 0.0
    pool = pd.concat([pool, dead], ignore_index=True)
    # strong alive 13 + the two dead = current squad; outside pool is weak
    pool.loc[pool.player_id < 900, "eff"] = 1.0
    strong = pool[pool.player_id < 900].sort_values("eff", ascending=False)
    current = (strong[strong.position == "GK"].player_id[:2].tolist()
               + strong[strong.position == "DEF"].player_id[:4].tolist()
               + strong[strong.position == "MID"].player_id[:4].tolist()
               + strong[strong.position == "FWD"].player_id[:3].tolist()
               + [900, 901])
    pool.loc[pool.player_id.isin(current), "eff"] = 6.0
    pool.loc[pool.player_id.isin([900, 901]), "eff"] = 0.0
    plan = sfa.solve_joint(pool, SF_CONFIG, current, free=0)
    chosen_ids = {int(pool["player_id"][i]) for i in plan["squad"]}
    assert plan["n_new"] == 0, "0.9-point upgrades must not beat a -3 hit"
    assert {900, 901} <= chosen_ids


def test_simulation_is_deterministic_and_bounded():
    pool = synthetic_pool()
    ctx = synthetic_ctx()
    plan = sfa.solve_joint(pool, SF_CONFIG, current_ids=None, free=5,
                           booster="qualification")
    a = sfa.RoundSimulation(pool, ctx, n_sims=500, seed=11).summarize(plan)
    b = sfa.RoundSimulation(pool, ctx, n_sims=500, seed=11).summarize(plan)
    assert a == b
    assert 0.0 <= a["p_beat_field"] <= 1.0
    assert a["p_beat_field_by_15"] <= a["p_beat_field"]


def test_qualification_payoff_matches_bracket_structure():
    pool = synthetic_pool()
    ctx = synthetic_ctx()
    plan = sfa.solve_joint(pool, SF_CONFIG, current_ids=None, free=5)
    sim = sfa.RoundSimulation(pool, ctx, n_sims=300, seed=3)
    xi = plan["xi"]
    counts = pool["country_abbr"][xi].value_counts()
    payoff = 2.0 * sim.adv[:, xi].sum(axis=1)
    allowed = {2.0 * (counts.get(w1, 0) + counts.get(w2, 0))
               for w1 in ("AAA", "BBB") for w2 in ("CCC", "DDD")}
    assert set(np.unique(payoff)) <= allowed


@pytest.mark.skipif(
    not (REPO_ROOT / "data/processed/predictions_2026-07-12.parquet").exists(),
    reason="semifinal decision-point snapshot not present")
def test_build_pool_on_committed_snapshot():
    pool, ctx = sfa.build_pool(
        REPO_ROOT / "data/processed/predictions_2026-07-12.parquet",
        REPO_ROOT / "data/raw/fixtures_2026-07-12.parquet",
        round_id=7, market_snapshot=None,
        current_ids=[910, 500])          # one eliminated, one alive
    assert ((pool["scout_ev"] >= 0.0) & (pool["scout_ev"] <= 2.0)).all()
    for a, b in ctx["matches"]:
        assert ctx["p_adv"][a] + ctx["p_adv"][b] == pytest.approx(1.0)
    dead = pool[pool["player_id"] == 910]
    assert len(dead) == 1 and float(dead["eff"].iloc[0]) == 0.0
    assert (pool["s_q90"] >= pool["s_q50"]).all()
