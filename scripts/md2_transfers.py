"""Analyze MD1 actuals and recommend exactly <=2 MD2 transfers.

Per-backend predictions, aggregated over the MD2+MD3 horizon. The user has
only 2 free transfers and does not want a hit, so we hard-cap transfers
in <=2 rather than letting the optimizer pay -3 per extra. We also report
an ensemble (mean of three backends) and the same per-player breakdown
showing who each backend wants in/out and at what MD2 expected points.
"""
from __future__ import annotations
import subprocess
from collections import defaultdict
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
CAPTAIN_ID = 1338

STAGE = Stage.GROUP_MD2
CONFIG = STAGE_CONFIGS[STAGE]
HORIZON = DEFAULT_ROUND_HORIZON[STAGE]
FIRST_ROUND = HORIZON[0]
MAX_TRANSFERS = 2  # user's free quota; no hit allowed

raw = pd.read_parquet("data/raw/players_2026-06-18.parquet")
raw_md1 = {int(r.player_id): (list(r.round_points)[0] if r.round_points is not None and len(list(r.round_points)) >= 1 else 0)
           for r in raw.itertuples()}
raw_meta = raw.set_index("player_id")

def solve_capped_transfer(players: pd.DataFrame, current_ids: list[int]) -> tuple[list[int], float]:
    """Best 15-player squad with at most MAX_TRANSFERS new picks vs current."""
    players = players[~players["is_eliminated"].astype(bool)].reset_index(drop=True)
    current_set = set(current_ids)
    prob = pulp.LpProblem("md2_transfer", pulp.LpMaximize)
    x = {int(r.player_id): pulp.LpVariable(f"x_{int(r.player_id)}", cat="Binary")
         for r in players.itertuples()}
    prob += pulp.lpSum(x[int(r.player_id)] * float(r.total_effective_points)
                       for r in players.itertuples())
    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    for position, count in SQUAD_POSITION_COUNTS.items():
        ids = [int(r.player_id) for r in players.itertuples() if r.position == position.value]
        prob += pulp.lpSum(x[i] for i in ids) == count
    prob += pulp.lpSum(x[int(r.player_id)] * float(r.price_millions)
                       for r in players.itertuples()) <= CONFIG.budget_millions
    for country, group in players.groupby("country", sort=False):
        ids = [int(pid) for pid in group["player_id"]]
        prob += pulp.lpSum(x[i] for i in ids) <= CONFIG.max_per_country
    new_picks = pulp.lpSum(x[int(r.player_id)] for r in players.itertuples()
                           if int(r.player_id) not in current_set)
    prob += new_picks <= MAX_TRANSFERS
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[status] == "Optimal", f"solver: {pulp.LpStatus[status]}"
    chosen = sorted(int(pid) for pid, v in x.items() if v.value() > 0.5)
    obj = float(pulp.value(prob.objective))
    return chosen, obj

def run_backend(backend):
    subprocess.check_call(
        [".venv/bin/python", "-m", "fifa_fantasy.model", "--backend", backend],
        cwd="/opt/fifa_wc_fantasy", stdout=subprocess.DEVNULL,
    )
    preds = pd.read_parquet("data/processed/predictions_2026-06-18.parquet")
    preds = apply_scouting_bonus(preds)
    table = aggregate_to_player(preds, HORIZON)
    chosen, obj = solve_capped_transfer(table, SQUAD_IDS)
    # MD2-only frame for the lineup solver
    md2 = preds[(preds["round_id"] == FIRST_ROUND) & (preds["player_id"].isin(chosen))]
    md2 = md2[["player_id", "full_name", "position", "country_abbr",
               "price_millions", "predicted_points", "is_home", "opponent_abbr"]]
    lineup = solve_lineup(md2)
    return preds, table, chosen, obj, md2, lineup

# === MD1 scorecard ===
print("=" * 90)
print("MD1 SCORECARD - your actual squad")
print("=" * 90)
print(f"{'role':<8} {'name':<22} {'pos':<4} {'cty':<5} {'price':>6} {'own%':>5} {'MD1':>4}")
md1_total = 0
for pid, (name, cty, pos, role) in SQUAD.items():
    pts = raw_md1.get(pid, 0)
    cap = "x2" if pid == CAPTAIN_ID else ""
    price = raw_meta.loc[pid]["price_millions"]
    own = raw_meta.loc[pid]["ownership_fraction"] * 100
    print(f"{role:<8} {name:<22} {pos:<4} {cty:<5} {price:>6.1f} {own:>5.1f} {pts:>4}{cap}")
    if role in ("Start", "CAPTAIN"):
        md1_total += pts
        if pid == CAPTAIN_ID:
            md1_total += pts
print(f"\nMD1 starter total (incl. captain x2): {md1_total}")

# === Per-backend with hard cap of 2 transfers ===
results = {}
for backend in ("heuristic", "poisson", "gbm"):
    print("\n" + "=" * 90)
    print(f"BACKEND: {backend}   (cap: <={MAX_TRANSFERS} transfers, no hit)")
    print("=" * 90)
    preds, table, chosen, obj, md2, lineup = run_backend(backend)
    results[backend] = (preds, table, chosen, obj, md2, lineup)
    transfers_in  = sorted(set(chosen) - set(SQUAD_IDS))
    transfers_out = sorted(set(SQUAD_IDS) - set(chosen))
    print(f"Transfers used: {len(transfers_in)} / {MAX_TRANSFERS}")
    print(f"Horizon (MD2+MD3) total expected effective_points: {obj:.2f}")
    md2_only_obj = md2["predicted_points"].sum()
    print(f"MD2-only XI expected: {lineup.objective:.2f}  (formation {lineup.formation})")
    print(f"Captain: {md2[md2['player_id']==lineup.captain_id].iloc[0]['full_name']} "
          f"E[MD2]={md2[md2['player_id']==lineup.captain_id].iloc[0]['predicted_points']:.2f}")
    print(f"Vice   : {md2[md2['player_id']==lineup.vice_captain_id].iloc[0]['full_name']} "
          f"E[MD2]={md2[md2['player_id']==lineup.vice_captain_id].iloc[0]['predicted_points']:.2f}")

    def detail_in(pid):
        r = preds[preds["player_id"] == pid].iloc[0]
        r2 = preds[(preds["round_id"] == FIRST_ROUND) & (preds["player_id"] == pid)]
        md2_e = float(r2["predicted_points"].iloc[0]) if len(r2) else 0.0
        return f"  IN  {r['full_name']:<22} {r['country_abbr']:<4} {r['position']:<4} ${float(r['price_millions']):>4.1f}M  E[MD2]={md2_e:.2f}"
    def detail_out(pid):
        n, c, p, _ = SQUAD[pid]
        price = float(raw_meta.loc[pid]["price_millions"])
        md1 = raw_md1.get(pid, 0)
        return f"  OUT {n:<22} {c:<4} {p:<4} ${price:>4.1f}M  MD1={md1}"
    for pid in transfers_in:
        print(detail_in(pid))
    for pid in transfers_out:
        print(detail_out(pid))

# === ENSEMBLE: average effective_points across the 3 backends ===
print("\n" + "=" * 90)
print(f"ENSEMBLE (mean of three backends)   cap: <={MAX_TRANSFERS} transfers")
print("=" * 90)
preds_h, table_h, _, _, _, _ = results["heuristic"]
tables = []
for b in ("heuristic", "poisson", "gbm"):
    _, t, _, _, _, _ = results[b]
    tables.append(t.set_index("player_id")[["total_effective_points"]]
                  .rename(columns={"total_effective_points": f"eff_{b}"}))
ens = pd.concat(tables, axis=1).fillna(0)
ens["total_effective_points"] = ens.mean(axis=1)
meta = table_h.set_index("player_id").drop(
    columns=["total_effective_points", "total_predicted_points"])
ens_table = meta.join(ens["total_effective_points"]).reset_index()
chosen, obj = solve_capped_transfer(ens_table, SQUAD_IDS)
transfers_in  = sorted(set(chosen) - set(SQUAD_IDS))
transfers_out = sorted(set(SQUAD_IDS) - set(chosen))
print(f"Transfers used: {len(transfers_in)} / {MAX_TRANSFERS}")
print(f"Horizon ensemble expected: {obj:.2f}")
def name_country_pos(pid):
    r = preds_h[preds_h["player_id"] == pid].iloc[0]
    return f"{r['full_name']:<22} {r['country_abbr']:<4} {r['position']:<4} ${float(r['price_millions']):>4.1f}M"
for pid in transfers_in:
    r2 = preds_h[(preds_h["round_id"] == FIRST_ROUND) & (preds_h["player_id"] == pid)]
    md2_e = float(r2["predicted_points"].iloc[0]) if len(r2) else 0.0
    print(f"  IN  {name_country_pos(pid)}  E[MD2 heur]={md2_e:.2f}")
for pid in transfers_out:
    n, c, p, _ = SQUAD[pid]
    price = float(raw_meta.loc[pid]["price_millions"])
    print(f"  OUT {n:<22} {c:<4} {p:<4} ${price:>4.1f}M  MD1={raw_md1.get(pid, 0)}")

# === Per-position upgrade scan: what every starter loses by swapping ===
# Useful sanity-check: who is the optimizer rejecting?
print("\n" + "=" * 90)
print(f"PLAYER-BY-PLAYER MD2 OUTLOOK (mean of 3 backends, ensemble effective MD2+MD3)")
print("=" * 90)
ens2 = ens_table.set_index("player_id")
print(f"{'name':<22} {'cty':<4} {'pos':<4} {'price':>6} {'own%':>5} {'MD1':>4} {'E[2+3]':>8}")
# Sort starters by ensemble value desc to surface clear-weak-links.
rows = []
for pid in SQUAD_IDS:
    name, cty, pos, role = SQUAD[pid]
    e = float(ens2.loc[pid]["total_effective_points"]) if pid in ens2.index else 0.0
    price = float(raw_meta.loc[pid]["price_millions"])
    own = raw_meta.loc[pid]["ownership_fraction"] * 100
    rows.append((role, e, name, cty, pos, price, own, raw_md1.get(pid, 0)))
rows.sort(key=lambda r: (r[0] != "CAPTAIN", -r[1]))
for role, e, name, cty, pos, price, own, md1 in rows:
    tag = "C" if role == "CAPTAIN" else ("B" if role == "Bench" else " ")
    print(f"{tag} {name:<22} {cty:<4} {pos:<4} {price:>6.1f} {own:>5.1f} {md1:>4} {e:>8.2f}")

# === Top candidates per position (ensemble) for context ===
print("\n" + "=" * 90)
print("TOP 8 CANDIDATES PER POSITION (ensemble), excluding eliminated")
print("=" * 90)
preds_h, _, _, _, _, _ = results["heuristic"]
ens_meta = ens_table.copy()
ens_meta["in_squad"] = ens_meta["player_id"].isin(SQUAD_IDS)
# MD2-only ensemble: average predicted_points across the three runs.
# Cheaper: just use heuristic's MD2 predicted_points as a proxy column.
md2_preds_h = preds_h[preds_h["round_id"] == FIRST_ROUND][
    ["player_id", "predicted_points", "is_home", "opponent_abbr"]
].rename(columns={"predicted_points": "md2_e_heur"})
ens_meta = ens_meta.merge(md2_preds_h, on="player_id", how="left")

for pos in ("GK", "DEF", "MID", "FWD"):
    sub = ens_meta[(ens_meta["position"] == pos) & (~ens_meta["is_eliminated"])]
    sub = sub.sort_values("total_effective_points", ascending=False).head(8)
    print(f"\n--- {pos} (top 8) ---")
    print(f"{'name':<22} {'cty':<4} {'price':>5} {'own%':>5} {'E[2+3]':>7} {'E[MD2h]':>8} {'opp':<8} {'in_sq':<5}")
    for r in sub.itertuples():
        flag = "YES" if r.in_squad else ""
        opp = f"{'vs' if r.is_home else '@'}{r.opponent_abbr}" if pd.notna(r.opponent_abbr) else "-"
        print(f"{r.full_name:<22} {r.country_abbr:<4} {r.price_millions:>5.1f} {r.ownership_fraction*100:>5.1f} {r.total_effective_points:>7.2f} {r.md2_e_heur if pd.notna(r.md2_e_heur) else 0:>8.2f} {opp:<8} {flag:<5}")
