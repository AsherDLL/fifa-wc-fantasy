"""Explore feasible 2-transfer paths to get Mbappé and/or Kane into the squad.

Constraints:
- 2 transfers max, no hit
- Position counts fixed (2 GK / 5 DEF / 5 MID / 3 FWD)
- Country cap 3 per country
- Budget $100M

Position math forces a FWD-out per FWD-in. France cap is already at 3
(Doué + Olise + Dembélé), so adding Mbappé requires dropping a French
player. The user wants Olise kept, so the candidates are Doué and Dembélé.
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
    45:("Emi Martínez","ARG","GK"), 1709:("Reece James","ENG","DEF"),
    2053:("Pacho","ECU","DEF"), 523:("Tah","GER","DEF"), 521:("Raum","GER","DEF"),
    505:("Doué","FRA","MID"), 57:("Enzo","ARG","MID"), 517:("Olise","FRA","MID"),
    501:("Dembélé","FRA","MID"), 804:("Gakpo","NED","FWD"),
    1338:("Lautaro","ARG","FWD"), 1523:("Penders","BEL","GK"),
    918:("Dalot","POR","DEF"), 1711:("Watkins","ENG","FWD"),
    2000:("Nico Williams","ESP","MID"),
}
SQUAD_IDS = list(SQUAD.keys())

MBAPPE = None  # resolved below
KANE = None

STAGE = Stage.GROUP_MD2
CONFIG = STAGE_CONFIGS[STAGE]
HORIZON = DEFAULT_ROUND_HORIZON[STAGE]
FIRST_ROUND = HORIZON[0]

def solve(table, locked_in, forced_out):
    p = table[~table["is_eliminated"].astype(bool)].reset_index(drop=True)
    current_set = set(SQUAD_IDS)
    prob = pulp.LpProblem("premium", pulp.LpMaximize)
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
    for pid in forced_out:
        if pid in x: prob += x[pid] == 0
    for pid in locked_in:
        if pid in x: prob += x[pid] == 1
    new_picks = pulp.lpSum(x[int(r.player_id)] for r in p.itertuples()
                           if int(r.player_id) not in current_set)
    prob += new_picks <= 2
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None, None
    return sorted(int(pid) for pid, v in x.items() if v.value() > 0.5), float(pulp.value(prob.objective))

tables = {}; preds_by = {}
for b in ("heuristic","poisson","gbm"):
    subprocess.check_call([".venv/bin/python","-m","fifa_fantasy.model","--backend",b],
        cwd="/opt/fifa_wc_fantasy", stdout=subprocess.DEVNULL)
    preds = pd.read_parquet("data/processed/predictions_2026-06-18.parquet")
    preds = apply_scouting_bonus(preds)
    preds_by[b] = preds; tables[b] = aggregate_to_player(preds, HORIZON)
parts = [tables[b].set_index("player_id")[["total_effective_points"]]
         .rename(columns={"total_effective_points": f"e_{b}"})
         for b in ("heuristic","poisson","gbm")]
ens = pd.concat(parts, axis=1).fillna(0)
ens["total_effective_points"] = ens.mean(axis=1)
meta = tables["heuristic"].set_index("player_id").drop(
    columns=["total_effective_points","total_predicted_points"])
ens_table = meta.join(ens["total_effective_points"]).reset_index()

# resolve Mbappé and Kane ids
preds_h = preds_by["heuristic"]
MBAPPE = int(preds_h[preds_h["full_name"].str.contains("Mbapp", na=False)].iloc[0]["player_id"])
KANE   = int(preds_h[preds_h["full_name"].str.startswith("Harry Kane", na=False)].iloc[0]["player_id"])
print(f"Mbappé id={MBAPPE}, Kane id={KANE}")

# Add a "realised MD1 bump" to the model's expected points for Mbappé/Kane?
# No — instead show what models predict alongside what they actually scored.
def md1(pid):
    raw = pd.read_parquet("data/raw/players_2026-06-18.parquet")
    rp = raw[raw["player_id"]==pid].iloc[0]["round_points"]
    return list(rp)[0] if rp is not None and len(list(rp)) else 0

def describe(pid, label):
    r = preds_h[preds_h["player_id"]==pid].iloc[0]
    rr = preds_h[(preds_h["round_id"]==FIRST_ROUND) & (preds_h["player_id"]==pid)]
    md2 = float(rr["predicted_points"].iloc[0]) if len(rr) else 0
    return (f"  {label} {r['full_name']:<22} {r['country_abbr']} {r['position']} "
            f"${float(r['price_millions']):>4.1f}M  MD1={md1(pid)}  "
            f"E[MD2 heur]={md2:.2f}")

print("\nTarget premiums:")
print(describe(MBAPPE, "MBAPPE"))
print(describe(KANE,   "KANE  "))

# --- Scenarios -------------------------------------------------------------

SCENARIOS = [
    # (name, locked_in, forced_out)
    ("S1: Mbappé in (drop Lautaro + Doué)",
        [MBAPPE, 517],      # lock Mbappé + Olise
        [1338, 505]),       # drop Lautaro + Doué
    ("S2: Mbappé in (drop Lautaro + Dembélé)",
        [MBAPPE, 517],
        [1338, 501]),
    ("S3: Mbappé in (drop Watkins + Doué)  [keeps Lautaro]",
        [MBAPPE, 517],
        [1711, 505]),
    ("S4: Kane in (drop Lautaro + Watkins)",
        [KANE,  517],
        [1338, 1711]),
    ("S5: Kane in (drop Lautaro only)  - infeasible due to budget? we'll see",
        [KANE, 517],
        [1338]),
    ("S6: Kane + Mbappé in - 3 transfers needed; show that here",
        [KANE, MBAPPE, 517],
        []),
]

for name, locked, forced in SCENARIOS:
    chosen, obj = solve(ens_table, locked, forced)
    print("\n" + name)
    if chosen is None:
        print("  INFEASIBLE under constraints")
        continue
    tin = sorted(set(chosen) - set(SQUAD_IDS))
    tout = sorted(set(SQUAD_IDS) - set(chosen))
    if len(tin) > 2:
        print(f"  Needs {len(tin)} transfers (you only have 2). Hit would be -{3*(len(tin)-2)} pts.")
    print(f"  ensemble horizon: {obj:.2f}")
    for pid in tin:
        r = preds_h[preds_h["player_id"]==pid].iloc[0]
        rr = preds_h[(preds_h["round_id"]==FIRST_ROUND) & (preds_h["player_id"]==pid)]
        md2 = float(rr["predicted_points"].iloc[0]) if len(rr) else 0
        print(f"    IN  {r['full_name']:<22} {r['country_abbr']} {r['position']} ${float(r['price_millions']):>4.1f}M  MD1={md1(pid)}  E[MD2 heur]={md2:.2f}")
    for pid in tout:
        n, c, p = SQUAD[pid]
        print(f"    OUT {n:<22} {c} {p}")
    md2 = preds_h[(preds_h["round_id"]==FIRST_ROUND) & (preds_h["player_id"].isin(chosen))]
    md2 = md2[["player_id","full_name","position","country_abbr",
               "price_millions","predicted_points","is_home","opponent_abbr"]]
    lineup = solve_lineup(md2)
    cap = md2[md2["player_id"]==lineup.captain_id].iloc[0]
    print(f"    MD2 XI exp={lineup.objective:.2f}  formation={lineup.formation}  "
          f"captain={cap['full_name']} E={cap['predicted_points']:.2f}")
