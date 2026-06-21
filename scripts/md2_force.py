"""User-directed scenario: force OUT Lautaro, lock IN Olise, pick best 2 transfers.

The user disagrees with the model's "drop Olise" call. Olise scored your highest
MD1 (6) and is a tournament-favorite winger; the user wants him kept. The user
also wants Lautaro out after a 1-pt captain blank. We honor both and let the
solver pick the second transfer freely (capped at 2 total).
"""
from __future__ import annotations
import subprocess
import pandas as pd
import pulp

from fifa_fantasy.collector.schemas import Stage
from fifa_fantasy.optimizer.pipeline import apply_scouting_bonus, aggregate_to_player
from fifa_fantasy.optimizer.solvers import (
    SQUAD_SIZE, SQUAD_POSITION_COUNTS, solve_lineup,
)
from fifa_fantasy.optimizer.stage_config import STAGE_CONFIGS, DEFAULT_ROUND_HORIZON

SQUAD = {
    45:   ("Emiliano Martínez",  "ARG", "GK",  "Start"),
    1709: ("Reece James",        "ENG", "DEF", "Start"),
    2053: ("Willian Pacho",      "ECU", "DEF", "Start"),
    523:  ("Jonathan Tah",       "GER", "DEF", "Start"),
    521:  ("David Raum",         "GER", "DEF", "Start"),
    505:  ("Désiré Doué",        "FRA", "MID", "Start"),
    57:   ("Enzo Fernández",     "ARG", "MID", "Start"),
    517:  ("Michael Olise",      "FRA", "MID", "Start"),
    501:  ("Ousmane Dembélé",    "FRA", "MID", "Start"),
    804:  ("Cody Gakpo",         "NED", "FWD", "Start"),
    1338: ("Lautaro Martínez",   "ARG", "FWD", "CAPTAIN"),
    1523: ("Mike Penders",       "BEL", "GK",  "Bench"),
    918:  ("Diogo Dalot",        "POR", "DEF", "Bench"),
    1711: ("Ollie Watkins",      "ENG", "FWD", "Bench"),
    2000: ("Nico Williams",      "ESP", "MID", "Bench"),
}
SQUAD_IDS = list(SQUAD.keys())
FORCED_OUT = [1338]   # Lautaro Martínez
LOCKED_IN  = [517]    # Olise must stay

STAGE = Stage.GROUP_MD2
CONFIG = STAGE_CONFIGS[STAGE]
HORIZON = DEFAULT_ROUND_HORIZON[STAGE]
FIRST_ROUND = HORIZON[0]

def solve(table, current_ids):
    p = table[~table["is_eliminated"].astype(bool)].reset_index(drop=True)
    current_set = set(current_ids)
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
    for pid in FORCED_OUT:
        if pid in x:
            prob += x[pid] == 0
    for pid in LOCKED_IN:
        if pid in x:
            prob += x[pid] == 1
    new_picks = pulp.lpSum(x[int(r.player_id)] for r in p.itertuples()
                           if int(r.player_id) not in current_set)
    prob += new_picks <= 2
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[status] == "Optimal", pulp.LpStatus[status]
    return sorted(int(pid) for pid, v in x.items() if v.value() > 0.5), float(pulp.value(prob.objective))

def run_backend(b):
    subprocess.check_call([".venv/bin/python", "-m", "fifa_fantasy.model", "--backend", b],
        cwd="/opt/fifa_wc_fantasy", stdout=subprocess.DEVNULL)
    preds = pd.read_parquet("data/processed/predictions_2026-06-18.parquet")
    preds = apply_scouting_bonus(preds)
    return preds, aggregate_to_player(preds, HORIZON)

tables = {}; preds_by = {}
for b in ("heuristic", "poisson", "gbm"):
    preds, t = run_backend(b)
    preds_by[b] = preds; tables[b] = t
parts = [tables[b].set_index("player_id")[["total_effective_points"]]
         .rename(columns={"total_effective_points": f"e_{b}"})
         for b in ("heuristic","poisson","gbm")]
ens = pd.concat(parts, axis=1).fillna(0)
ens["total_effective_points"] = ens.mean(axis=1)
meta = tables["heuristic"].set_index("player_id").drop(
    columns=["total_effective_points", "total_predicted_points"])
ens_table = meta.join(ens["total_effective_points"]).reset_index()

# --- per-backend ---
print("PER-BACKEND (Lautaro forced OUT, Olise locked IN, <=2 transfers)")
for b in ("heuristic", "poisson", "gbm"):
    chosen, obj = solve(tables[b], SQUAD_IDS)
    tin = sorted(set(chosen) - set(SQUAD_IDS))
    tout = sorted(set(SQUAD_IDS) - set(chosen))
    preds = preds_by[b]
    print(f"\n  [{b}] horizon expected: {obj:.2f}")
    for pid in tin:
        r = preds[preds["player_id"] == pid].iloc[0]
        rr = preds[(preds["round_id"]==FIRST_ROUND) & (preds["player_id"]==pid)]
        md2 = float(rr["predicted_points"].iloc[0]) if len(rr) else 0
        print(f"    IN  {r['full_name']:<22} {r['country_abbr']:<4} {r['position']:<4} ${float(r['price_millions']):>4.1f}M  E[MD2]={md2:.2f}")
    for pid in tout:
        nm, c, p, _ = SQUAD[pid]
        print(f"    OUT {nm:<22} {c:<4} {p:<4}")

# --- ensemble ---
print("\nENSEMBLE (mean of 3 backends)")
chosen, obj = solve(ens_table, SQUAD_IDS)
tin = sorted(set(chosen) - set(SQUAD_IDS))
tout = sorted(set(SQUAD_IDS) - set(chosen))
preds_h = preds_by["heuristic"]
print(f"  horizon ensemble expected: {obj:.2f}")
for pid in tin:
    r = preds_h[preds_h["player_id"] == pid].iloc[0]
    rr = preds_h[(preds_h["round_id"]==FIRST_ROUND) & (preds_h["player_id"]==pid)]
    md2 = float(rr["predicted_points"].iloc[0]) if len(rr) else 0
    print(f"    IN  {r['full_name']:<22} {r['country_abbr']:<4} {r['position']:<4} ${float(r['price_millions']):>4.1f}M  E[MD2]={md2:.2f}")
for pid in tout:
    nm, c, p, _ = SQUAD[pid]
    print(f"    OUT {nm:<22} {c:<4} {p:<4}")

# MD2-only XI with new squad (heuristic backend)
md2 = preds_h[(preds_h["round_id"] == FIRST_ROUND) & (preds_h["player_id"].isin(chosen))]
md2 = md2[["player_id", "full_name", "position", "country_abbr",
           "price_millions", "predicted_points", "is_home", "opponent_abbr"]]
lineup = solve_lineup(md2)
print(f"\nMD2 XI expected (heuristic): {lineup.objective:.2f} pts, formation {lineup.formation}")
cap = md2[md2['player_id']==lineup.captain_id].iloc[0]
vc  = md2[md2['player_id']==lineup.vice_captain_id].iloc[0]
print(f"Captain: {cap['full_name']} ({cap['country_abbr']}) E[MD2]={cap['predicted_points']:.2f}")
print(f"Vice   : {vc['full_name']} ({vc['country_abbr']}) E[MD2]={vc['predicted_points']:.2f}")

# Show all FWDs sorted by ensemble EV so the user can see the alternatives.
print("\nTOP 12 FWDs (ensemble), playing, not eliminated")
fwds = ens_table[(ens_table["position"]=="FWD") & (~ens_table["is_eliminated"])]
fwds = fwds.sort_values("total_effective_points", ascending=False).head(12)
md2_fwd = preds_h[(preds_h["round_id"]==FIRST_ROUND) & (preds_h["position"]=="FWD")][
    ["player_id","predicted_points","is_home","opponent_abbr"]].rename(columns={"predicted_points":"md2_h"})
fwds = fwds.merge(md2_fwd, on="player_id", how="left")
fwds["in_squad"] = fwds["player_id"].isin(SQUAD_IDS)
print(f"{'name':<22} {'cty':<4} {'price':>5} {'own%':>5} {'E[2+3]':>7} {'E[MD2h]':>8} {'opp':<8} {'in_sq':<5}")
for r in fwds.itertuples():
    opp = f"{'vs' if r.is_home else '@'}{r.opponent_abbr}" if pd.notna(r.opponent_abbr) else "-"
    print(f"{r.full_name:<22} {r.country_abbr:<4} {r.price_millions:>5.1f} {r.ownership_fraction*100:>5.1f} {r.total_effective_points:>7.2f} {r.md2_h if pd.notna(r.md2_h) else 0:>8.2f} {opp:<8} {'YES' if r.in_squad else '':<5}")
