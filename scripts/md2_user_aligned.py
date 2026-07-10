"""User-aligned scenario: force OUT Watkins + Pacho, optimise the 2 IN slots.

The user said Watkins is not getting a start for England and Pacho underperformed.
Constrain the transfer solver to drop those two and pick the best replacements
(same position counts) under MD2+MD3 ensemble expected points.
"""
from __future__ import annotations
import sys
import subprocess
import pandas as pd
import pulp

from fifa_fantasy.collector.schemas import Stage
from fifa_fantasy.optimizer.pipeline import apply_scouting_bonus, aggregate_to_player
from fifa_fantasy.optimizer.solvers import (
    SQUAD_SIZE, SQUAD_POSITION_COUNTS, solve_lineup,
)
from fifa_fantasy.optimizer.stage_config import STAGE_CONFIGS, DEFAULT_ROUND_HORIZON

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SQUAD_IDS = [45, 1709, 2053, 523, 521, 505, 57, 517, 501, 804, 1338,
             1523, 918, 1711, 2000]
FORCED_OUT = [1711, 2053]  # Watkins (FWD), Pacho (DEF)

STAGE = Stage.GROUP_MD2
CONFIG = STAGE_CONFIGS[STAGE]
HORIZON = DEFAULT_ROUND_HORIZON[STAGE]
FIRST_ROUND = HORIZON[0]

def solve_constrained(table, current_ids, forced_out):
    p = table[~table["is_eliminated"].astype(bool)].reset_index(drop=True)
    current_set = set(current_ids)
    forced_set = set(forced_out)
    prob = pulp.LpProblem("md2_forced", pulp.LpMaximize)
    x = {int(r.player_id): pulp.LpVariable(f"x_{int(r.player_id)}", cat="Binary")
         for r in p.itertuples()}
    prob += pulp.lpSum(x[int(r.player_id)] * float(r.total_effective_points)
                       for r in p.itertuples())
    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    for pos, count in SQUAD_POSITION_COUNTS.items():
        ids = [int(r.player_id) for r in p.itertuples() if r.position == pos.value]
        prob += pulp.lpSum(x[i] for i in ids) == count
    prob += pulp.lpSum(x[int(r.player_id)] * float(r.price_millions)
                       for r in p.itertuples()) <= CONFIG.budget_millions
    for country, group in p.groupby("country", sort=False):
        ids = [int(pid) for pid in group["player_id"]]
        prob += pulp.lpSum(x[i] for i in ids) <= CONFIG.max_per_country
    # Forced out: those player_ids cannot be in the new squad.
    for pid in forced_set:
        if pid in x:
            prob += x[pid] == 0
    # New picks = exactly 2 (replacing the two forced drops).
    new_picks = pulp.lpSum(x[int(r.player_id)] for r in p.itertuples()
                           if int(r.player_id) not in current_set)
    prob += new_picks <= 2
    # Also: every kept current player (not in forced_out) should stay if optimal,
    # but we don't force keep -- the solver may opt to drop someone else to fit
    # a better same-position replacement.
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[status] == "Optimal", pulp.LpStatus[status]
    return sorted(int(pid) for pid, v in x.items() if v.value() > 0.5), float(pulp.value(prob.objective))

def run_backend(backend):
    subprocess.check_call(
        [sys.executable, "-m", "fifa_fantasy.model", "--backend", backend],
        cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
    )
    preds = pd.read_parquet("data/processed/predictions_2026-06-18.parquet")
    preds = apply_scouting_bonus(preds)
    return preds, aggregate_to_player(preds, HORIZON)

tables = {}
preds_by_b = {}
for b in ("heuristic", "poisson", "gbm"):
    preds, t = run_backend(b)
    preds_by_b[b] = preds
    tables[b] = t

# Ensemble
ens_parts = []
for b, t in tables.items():
    ens_parts.append(t.set_index("player_id")[["total_effective_points"]]
                     .rename(columns={"total_effective_points": f"e_{b}"}))
ens = pd.concat(ens_parts, axis=1).fillna(0)
ens["total_effective_points"] = ens.mean(axis=1)
meta = tables["heuristic"].set_index("player_id").drop(
    columns=["total_effective_points", "total_predicted_points"])
ens_table = meta.join(ens["total_effective_points"]).reset_index()

chosen, obj = solve_constrained(ens_table, SQUAD_IDS, FORCED_OUT)
preds_h = preds_by_b["heuristic"]

transfers_in  = sorted(set(chosen) - set(SQUAD_IDS))
transfers_out = sorted(set(SQUAD_IDS) - set(chosen))
print("USER-ALIGNED SCENARIO (force OUT Watkins + Pacho)")
print(f"Ensemble horizon (MD2+MD3) expected: {obj:.2f}")
print(f"Transfers in : {len(transfers_in)}, out: {len(transfers_out)}")
for pid in transfers_in:
    r = preds_h[preds_h["player_id"] == pid].iloc[0]
    rr = preds_h[(preds_h["round_id"] == FIRST_ROUND) & (preds_h["player_id"] == pid)]
    md2 = float(rr["predicted_points"].iloc[0]) if len(rr) else 0.0
    print(f"  IN  {r['full_name']:<22} {r['country_abbr']:<4} {r['position']:<4} ${float(r['price_millions']):>4.1f}M  E[MD2 heur]={md2:.2f}")
for pid in transfers_out:
    r = preds_h[preds_h["player_id"] == pid].iloc[0]
    print(f"  OUT {r['full_name']:<22} {r['country_abbr']:<4} {r['position']:<4} ${float(r['price_millions']):>4.1f}M")

# MD2-only XI with new squad
md2 = preds_h[(preds_h["round_id"] == FIRST_ROUND) & (preds_h["player_id"].isin(chosen))]
md2 = md2[["player_id", "full_name", "position", "country_abbr",
           "price_millions", "predicted_points", "is_home", "opponent_abbr"]]
lineup = solve_lineup(md2)
print(f"\nMD2 XI expected (heuristic backend): {lineup.objective:.2f} pts, formation {lineup.formation}")
cap = md2[md2['player_id']==lineup.captain_id].iloc[0]
vc  = md2[md2['player_id']==lineup.vice_captain_id].iloc[0]
print(f"Captain: {cap['full_name']} ({cap['country_abbr']}) E[MD2]={cap['predicted_points']:.2f}")
print(f"Vice   : {vc['full_name']} ({vc['country_abbr']}) E[MD2]={vc['predicted_points']:.2f}")
