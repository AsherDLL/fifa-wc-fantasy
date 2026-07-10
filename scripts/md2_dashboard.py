"""Generate a focused MD2-decision HTML dashboard for the user's squad.

What it shows:
- Current squad with MD1 score + a minutes-risk flag derived from MD1 pts
  (0 -> almost certainly bench / DNP, 1 -> sub <60min, 2+ -> started)
- Ranked transfer scenarios (1 or 2 transfers, with hit when applicable)
- Captain options with E[MD2] from each backend and the realised MD1 anchor
- Top 10 candidates per position with fixture + MD1
- Sanity flags: models that disagree, MD1 hauls outside the squad
"""
from __future__ import annotations
import sys
import subprocess
import socket
import re
from datetime import datetime, timezone
import pandas as pd
import pulp

from fifa_fantasy.collector.schemas import Stage
from fifa_fantasy.optimizer.pipeline import apply_scouting_bonus, aggregate_to_player
from fifa_fantasy.optimizer.solvers import (
    SQUAD_SIZE, SQUAD_POSITION_COUNTS, solve_lineup, TRANSFER_HIT_POINTS,
)
from fifa_fantasy.optimizer.stage_config import STAGE_CONFIGS, DEFAULT_ROUND_HORIZON

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

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
STAGE = Stage.GROUP_MD2
CONFIG = STAGE_CONFIGS[STAGE]
HORIZON = DEFAULT_ROUND_HORIZON[STAGE]
FIRST_ROUND = HORIZON[0]
TOTAL_BUDGET = CONFIG.budget_millions

MESSI, MBAPPE, KANE = 21, 500, 468  # resolved below; defaults wrong
BARCOLA = 1431
EZE = 1710

# -----------------------------------------------------------
# Build the three backends' predictions and an ensemble
# -----------------------------------------------------------
tables = {}; preds_by = {}
for b in ("heuristic", "poisson", "gbm"):
    subprocess.check_call(
        [sys.executable, "-m", "fifa_fantasy.model", "--backend", b],
        cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
    )
    preds = pd.read_parquet("data/processed/predictions_2026-06-18.parquet")
    preds_by[b] = apply_scouting_bonus(preds)
    tables[b] = aggregate_to_player(preds_by[b], HORIZON)

parts = [tables[b].set_index("player_id")[["total_effective_points"]]
         .rename(columns={"total_effective_points": f"e_{b}"})
         for b in ("heuristic", "poisson", "gbm")]
ens = pd.concat(parts, axis=1).fillna(0)
ens["total_effective_points"] = ens.mean(axis=1)
meta = tables["heuristic"].set_index("player_id").drop(
    columns=["total_effective_points", "total_predicted_points"])
ens_table = meta.join(ens["total_effective_points"]).reset_index()

raw_players = pd.read_parquet("data/raw/players_2026-06-18.parquet")
raw_md1 = {int(r.player_id): (list(r.round_points)[0] if r.round_points is not None and len(list(r.round_points)) >= 1 else 0)
           for r in raw_players.itertuples()}
raw_total = {int(r.player_id): int(r.total_points) for r in raw_players.itertuples()}
raw_form = {int(r.player_id): float(r.form) for r in raw_players.itertuples()}
raw_status = {int(r.player_id): str(r.status) for r in raw_players.itertuples()}

# Resolve premium IDs by name
preds_h = preds_by["heuristic"]
def find_id(needle):
    m = preds_h[preds_h["full_name"].str.contains(needle, case=False, na=False)]
    return int(m.iloc[0]["player_id"]) if len(m) else None

MESSI  = find_id("Lionel Messi")
MBAPPE = find_id("Mbapp")
KANE   = find_id("Harry Kane")
BARCOLA = find_id("Bradley Barcola")
EZE    = find_id("Eberechi Eze")
SAKA   = find_id("Bukayo Saka")
CUNHA  = find_id("Matheus Cunha")
TONEY  = find_id("Ivan Toney")

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def minutes_flag(md1):
    if md1 == 0:
        return ("danger", "Did not score / likely DNP")
    if md1 == 1:
        return ("warn", "Likely sub appearance (1 pt)")
    if md1 == 2:
        return ("ok", "Played 60+ no contributions")
    return ("good", f"Played and contributed ({md1} pts)")

def md_pred(pid, rnd, backend):
    p = preds_by[backend]
    r = p[(p["round_id"] == rnd) & (p["player_id"] == pid)]
    return float(r["predicted_points"].iloc[0]) if len(r) else 0.0

def md_ens(pid, rnd):
    return sum(md_pred(pid, rnd, b) for b in ("heuristic", "poisson", "gbm")) / 3.0

def fixture(pid, rnd):
    p = preds_by["heuristic"]
    r = p[(p["round_id"] == rnd) & (p["player_id"] == pid)]
    if not len(r):
        return "-"
    r = r.iloc[0]
    return f"{'vs' if r['is_home'] else '@'}{r['opponent_abbr']}"

def lookup(pid):
    r = preds_h[preds_h["player_id"] == pid].iloc[0]
    return dict(name=r["full_name"], country=r["country_abbr"],
                position=r["position"], price=float(r["price_millions"]),
                ownership=float(r["ownership_fraction"]) * 100)

# -----------------------------------------------------------
# Run the transfer scenarios
# -----------------------------------------------------------
def solve(table, max_transfers, locked_in, forced_out):
    p = table[~table["is_eliminated"].astype(bool)].reset_index(drop=True)
    cur = set(SQUAD_IDS)
    prob = pulp.LpProblem("p", pulp.LpMaximize)
    x = {int(r.player_id): pulp.LpVariable(f"x_{int(r.player_id)}", cat="Binary")
         for r in p.itertuples()}
    extra = pulp.LpVariable("extra", lowBound=0, cat="Continuous")
    prob += pulp.lpSum(x[int(r.player_id)] * float(r.total_effective_points)
                       for r in p.itertuples()) - TRANSFER_HIT_POINTS * extra
    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    for pos, count in SQUAD_POSITION_COUNTS.items():
        ids = [int(r.player_id) for r in p.itertuples() if r.position == pos.value]
        prob += pulp.lpSum(x[i] for i in ids) == count
    prob += pulp.lpSum(x[int(r.player_id)] * float(r.price_millions)
                       for r in p.itertuples()) <= TOTAL_BUDGET
    for c, g in p.groupby("country", sort=False):
        ids = [int(pid) for pid in g["player_id"]]
        prob += pulp.lpSum(x[i] for i in ids) <= CONFIG.max_per_country
    for pid in forced_out:
        if pid in x: prob += x[pid] == 0
    for pid in locked_in:
        if pid in x: prob += x[pid] == 1
    new_picks = pulp.lpSum(x[int(r.player_id)] for r in p.itertuples()
                           if int(r.player_id) not in cur)
    prob += new_picks <= max_transfers
    prob += new_picks - 2 <= extra
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None
    chosen = sorted(int(pid) for pid, v in x.items() if v.value() > 0.5)
    return dict(
        chosen=chosen,
        net_horizon=float(pulp.value(prob.objective)),
        extra=int(round(extra.value())),
    )

# Compute model-only base values for comparison (no realised-MD1 bump).
SCENARIOS = []

def scenario(name, locked_in, forced_out, max_t=2, note=""):
    r = solve(ens_table, max_t, locked_in, forced_out)
    if r is None:
        SCENARIOS.append(dict(name=name, infeasible=True, note=note)); return
    tin = sorted(set(r["chosen"]) - set(SQUAD_IDS))
    tout = sorted(set(SQUAD_IDS) - set(r["chosen"]))
    # MD2 XI on heuristic
    md2 = preds_by["heuristic"][
        (preds_by["heuristic"]["round_id"] == FIRST_ROUND)
        & (preds_by["heuristic"]["player_id"].isin(r["chosen"]))
    ][["player_id", "full_name", "position", "country_abbr",
       "price_millions", "predicted_points", "is_home", "opponent_abbr"]]
    lineup = solve_lineup(md2)
    SCENARIOS.append(dict(
        name=name, note=note, infeasible=False,
        n_transfers=len(tin),
        hit=TRANSFER_HIT_POINTS * r["extra"],
        net_horizon=r["net_horizon"],
        md2_xi=lineup.objective,
        captain_id=lineup.captain_id,
        captain_e=float(md2[md2["player_id"] == lineup.captain_id].iloc[0]["predicted_points"]),
        captain_name=str(md2[md2["player_id"] == lineup.captain_id].iloc[0]["full_name"]),
        formation=lineup.formation,
        ins=[(pid, lookup(pid), fixture(pid, FIRST_ROUND), raw_md1.get(pid, 0)) for pid in tin],
        outs=[(pid, SQUAD[pid][0], SQUAD[pid][1], SQUAD[pid][2], raw_md1.get(pid, 0)) for pid in tout],
        chosen=r["chosen"],
    ))

# --- realistic 2-transfer plans ---
scenario("S-MESSI-1: Messi swap (drop Lautaro + Pacho)",
         [MESSI, 517], [1338, 2053],
         note="Pure Messi-in upgrade. Pacho is your lowest projected DEF.")
scenario("S-MESSI-2: Messi + a real MID (drop Lautaro + Dembélé)",
         [MESSI, 517], [1338, 501],
         note="Messi at FWD + cheaper MID (model picks). Drops Dembélé.")
scenario("S-MESSI-3: Messi + Saka (drop Lautaro + Olise)  [breaks user's keep-Olise]",
         [MESSI, SAKA], [1338, 517],
         note="Only here for comparison. You said keep Olise; this drops him.")
scenario("S-KANE-1: Kane + Barcola (drop Lautaro + Dembélé)",
         [KANE, 517], [1338, 501],
         note="Earlier S5. Barcola hauled 9 in MD1; minutes question stands.")
scenario("S-KANE-2: Kane + Saka (drop Lautaro + Olise)  [breaks user's keep-Olise]",
         [KANE, SAKA], [1338, 517],
         note="Both ENG premium attackers - 2 ENG only, OK with cap.")
scenario("S-MBAPPE-1: Mbappé + Eze (drop Lautaro + Dembélé)",
         [MBAPPE, 517], [1338, 501],
         note="Earlier S2. Eze blanked MD1.")
scenario("S-MBAPPE-MESSI: Mbappé + Messi (drop Lautaro + Dembélé)",
         [MBAPPE, MESSI, 517], [1338, 501], max_t=2,
         note="Two premiums. Both FWD - need to drop 2 FWDs - 3 transfers. Likely infeasible at <=2.")
scenario("S-MK-MESSI: Mbappé + Kane + Messi (3 transfers, -3 hit)",
         [MBAPPE, KANE, MESSI, 517], [], max_t=3,
         note="The 'all premium' fantasy. -3 hit; loses a real starter for bench filler.")
scenario("S-MK-BOTH: Mbappé + Kane (3 transfers, -3 hit)",
         [MBAPPE, KANE, 517], [], max_t=3,
         note="Both elite captains. -3 hit + lose Gakpo.")

# Best 2-transfer with no locks (data-only baseline)
scenario("BASELINE: best 2-transfer (model only)",
         [517], [], max_t=2,
         note="Pure model pick, only Olise locked per your call.")

# -----------------------------------------------------------
# Captain ranking for MD2 (across all 15 chosen in each plan? No --- across the
# top names you might field)
# -----------------------------------------------------------
CAPTAIN_CANDIDATES = [
    MESSI, MBAPPE, KANE, 1338, 517, 501, 505, 804, 469, BARCOLA, 57, 45, 1711, CUNHA, TONEY,
]
CAPTAIN_CANDIDATES = [p for p in CAPTAIN_CANDIDATES if p is not None]

# -----------------------------------------------------------
# Build the dashboard HTML
# -----------------------------------------------------------
def fmt(v, n=2):
    return f"{v:.{n}f}" if v is not None else "-"

def hostname():
    h = socket.gethostname() or "host"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", h)

NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

def color_for(flag):
    return {"danger":"#c0392b","warn":"#d68910","ok":"#7d6608","good":"#196f3d"}[flag]

# Sort scenarios by net horizon descending (feasible only first).
feasible = [s for s in SCENARIOS if not s.get("infeasible")]
feasible.sort(key=lambda s: -s["net_horizon"])

html = []
html.append("<!doctype html><html><head><meta charset='utf-8'>")
html.append("<title>FIFA Fantasy WC 2026 - MD2 transfer dashboard</title>")
html.append("""<style>
body{font-family:-apple-system,system-ui,sans-serif;max-width:1400px;margin:1.5rem auto;padding:0 1rem;color:#111;}
h1{font-size:1.4rem;margin:0 0 0.3rem;}
h2{font-size:1.1rem;margin-top:2rem;padding:8px 12px;background:#2c3e50;color:#fff;border-radius:4px;}
h3{font-size:1.0rem;margin-top:1.2rem;border-bottom:1px solid #ddd;padding-bottom:3px;}
.meta{color:#555;font-size:0.85rem;margin-bottom:1rem;}
table{border-collapse:collapse;width:100%;font-size:0.85rem;margin-bottom:0.8rem;}
th,td{padding:5px 9px;text-align:left;border-bottom:1px solid #eee;}
th{background:#f5f5f5;font-weight:600;}
tr.cap{background:#fff5d6;}
tr.vc{background:#eaf3ff;}
tr.bench td{color:#777;}
.flag{padding:1px 6px;border-radius:3px;font-size:0.75rem;color:#fff;display:inline-block;}
.note{font-size:0.82rem;color:#555;margin-top:-0.3rem;margin-bottom:0.5rem;font-style:italic;}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.75rem;margin-right:4px;background:#eef;}
.bad{background:#fadbd8;color:#922b21;}
.ok{background:#d4edda;color:#155724;}
.warn{background:#fcf3cf;color:#7d6608;}
.top{background:#fef9e7;}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:0.85rem;margin:0.5rem 0;}
.kv b{color:#555;}
.cmp{font-size:0.78rem;color:#666;}
.win{color:#196f3d;font-weight:600;}
.lose{color:#922b21;}
</style></head><body>""")

html.append(f"<h1>MD2 transfer dashboard</h1>")
html.append(f"<div class='meta'>Generated {NOW} on {hostname()}. Three backends ensembled (heuristic, Poisson, GBM v2). Current budget used: ${raw_players[raw_players.player_id.isin(SQUAD_IDS)]['price_millions'].astype(float).sum():.1f}M / $100.0M.</div>")

# ---------- Current squad ----------
html.append("<h2>Your MD1 squad - what actually happened</h2>")
html.append("<table><thead><tr><th>Role</th><th>Player</th><th>Cty</th><th>Pos</th><th>$M</th><th>Own%</th><th>MD1 pts</th><th>Form</th><th>Status</th><th>Minutes risk</th></tr></thead><tbody>")
md1_total = 0
for pid, (name, cty, pos, role) in SQUAD.items():
    md1 = raw_md1.get(pid, 0)
    flag_cls, flag_msg = minutes_flag(md1)
    capt = " *2" if role == "CAPTAIN" else ""
    if role in ("Start", "CAPTAIN"):
        md1_total += md1 * (2 if role == "CAPTAIN" else 1)
    price = float(raw_players[raw_players["player_id"]==pid].iloc[0]["price_millions"])
    own = float(raw_players[raw_players["player_id"]==pid].iloc[0]["ownership_fraction"]) * 100
    row_cls = "cap" if role == "CAPTAIN" else "bench" if role == "Bench" else ""
    html.append(f"<tr class='{row_cls}'><td>{role}</td><td>{name}</td><td>{cty}</td><td>{pos}</td><td>${price:.1f}</td><td>{own:.1f}%</td><td>{md1}{capt}</td><td>{raw_form.get(pid,0):.1f}</td><td>{raw_status.get(pid,'-')}</td><td><span class='flag' style='background:{color_for(flag_cls)}'>{flag_msg}</span></td></tr>")
html.append("</tbody></table>")
html.append(f"<div class='meta'><b>MD1 total: {md1_total} pts</b> (includes captain x2). Field median is typically 35-45 for a balanced squad.</div>")

# ---------- Premium FWDs side-by-side ----------
html.append("<h2>Premium FWD candidates - the MD2 picture</h2>")
html.append("<table><thead><tr><th>Player</th><th>Cty</th><th>$M</th><th>Own%</th><th>MD1</th><th>MD2 fixture</th><th>E[MD2] heur</th><th>E[MD2] poi</th><th>E[MD2] GBM</th><th>E[MD2] ens</th><th>Captain x2</th></tr></thead><tbody>")
for pid in [MESSI, MBAPPE, KANE, 1338, 1711, CUNHA, TONEY, 804]:
    if pid is None: continue
    info = lookup(pid)
    md1 = raw_md1.get(pid, 0)
    e_h = md_pred(pid, FIRST_ROUND, "heuristic")
    e_p = md_pred(pid, FIRST_ROUND, "poisson")
    e_g = md_pred(pid, FIRST_ROUND, "gbm")
    e_e = (e_h + e_p + e_g) / 3.0
    cap_double = e_e * 2
    in_sq = " <span class='pill ok'>in squad</span>" if pid in SQUAD_IDS else ""
    html.append(f"<tr><td>{info['name']}{in_sq}</td><td>{info['country']}</td><td>${info['price']:.1f}</td><td>{info['ownership']:.1f}%</td><td><b>{md1}</b></td><td>{fixture(pid, FIRST_ROUND)}</td><td>{e_h:.2f}</td><td>{e_p:.2f}</td><td>{e_g:.2f}</td><td><b>{e_e:.2f}</b></td><td>{cap_double:.2f}</td></tr>")
html.append("</tbody></table>")
html.append("<div class='note'>Note: the GBM and heuristic are EPL-trained; they cannot 'see' Messi (no EPL data) and tend to undervalue elite WC names. Treat Messi/Mbappé/Kane E[MD2] as a price-based floor, not a ceiling. The MD1 column is the realised evidence.</div>")

# ---------- Scenarios ranked ----------
html.append("<h2>Transfer scenarios, ranked by ensemble MD2+MD3 net horizon</h2>")
html.append("<div class='note'>'Net horizon' = expected MD2+MD3 effective points across the new 15-player squad, minus the -3 transfer hits. Higher is better. The MD2 XI line is what you'll likely score this round under optimal lineup + captain.</div>")
html.append("<table><thead><tr><th>Plan</th><th>#Trf</th><th>Hit</th><th>Net horizon</th><th>MD2 XI E</th><th>Captain</th><th>Captain E[MD2]</th><th>IN</th><th>OUT</th></tr></thead><tbody>")
for i, s in enumerate(feasible):
    cap_name = s["captain_name"]
    ins = " <br>".join(f"{lookup(pid)['name']} ({lookup(pid)['country']} {lookup(pid)['position']} ${lookup(pid)['price']:.1f}M, MD1={md1})" for pid, info, fx, md1 in s["ins"])
    outs = " <br>".join(f"{n} ({c} {p}, MD1={md1})" for pid, n, c, p, md1 in s["outs"])
    top_cls = "top" if i == 0 else ""
    html.append(f"<tr class='{top_cls}'><td><b>{s['name']}</b><div class='note'>{s.get('note','')}</div></td><td>{s['n_transfers']}</td><td>{'-'+str(s['hit']) if s['hit'] else '0'}</td><td><b>{s['net_horizon']:.2f}</b></td><td>{s['md2_xi']:.2f} ({s['formation']})</td><td>{cap_name}</td><td>{s['captain_e']:.2f}</td><td>{ins}</td><td>{outs}</td></tr>")
for s in [s for s in SCENARIOS if s.get("infeasible")]:
    html.append(f"<tr><td><b>{s['name']}</b><div class='note'>{s.get('note','')}</div></td><td colspan='8'>INFEASIBLE</td></tr>")
html.append("</tbody></table>")

# ---------- Captain board ----------
html.append("<h2>MD2 captain board (the decision that swings the round)</h2>")
html.append("<table><thead><tr><th>Player</th><th>Cty</th><th>Pos</th><th>$M</th><th>MD2 fixture</th><th>MD1 actual</th><th>E[MD2] heur</th><th>E[MD2] ens</th><th>x2 (captain)</th><th>Notes</th></tr></thead><tbody>")
cap_rows = []
for pid in CAPTAIN_CANDIDATES:
    info = lookup(pid)
    md1 = raw_md1.get(pid, 0)
    e_h = md_pred(pid, FIRST_ROUND, "heuristic")
    e_e = md_ens(pid, FIRST_ROUND)
    notes = []
    if pid not in SQUAD_IDS:
        notes.append("not in squad")
    if md1 == 0:
        notes.append("DNP/blanked MD1")
    if info["ownership"] > 50:
        notes.append("template (>50%)")
    if info["ownership"] < 5:
        notes.append("differential (<5%)")
    cap_rows.append((info, md1, e_h, e_e, notes, pid))
cap_rows.sort(key=lambda x: -x[3])  # by ensemble
for info, md1, e_h, e_e, notes, pid in cap_rows:
    note_str = ", ".join(notes)
    in_sq = " <span class='pill ok'>in squad</span>" if pid in SQUAD_IDS else ""
    html.append(f"<tr><td>{info['name']}{in_sq}</td><td>{info['country']}</td><td>{info['position']}</td><td>${info['price']:.1f}</td><td>{fixture(pid, FIRST_ROUND)}</td><td><b>{md1}</b></td><td>{e_h:.2f}</td><td><b>{e_e:.2f}</b></td><td>{e_e*2:.2f}</td><td>{note_str}</td></tr>")
html.append("</tbody></table>")

# ---------- Top MD1 hauls outside squad ----------
html.append("<h2>Top MD1 hauls you don't own (the &quot;look what I missed&quot; list)</h2>")
out_squad = preds_h[~preds_h["player_id"].isin(SQUAD_IDS)].drop_duplicates("player_id").copy()
out_squad["md1"] = out_squad["player_id"].map(lambda pid: raw_md1.get(int(pid), 0))
top_md1 = out_squad.sort_values("md1", ascending=False).head(20)
html.append("<table><thead><tr><th>Player</th><th>Cty</th><th>Pos</th><th>$M</th><th>Own%</th><th>MD1</th><th>MD2 fix</th><th>E[MD2] ens</th></tr></thead><tbody>")
for r in top_md1.itertuples():
    pid = int(r.player_id)
    e_e = md_ens(pid, FIRST_ROUND)
    html.append(f"<tr><td>{r.full_name}</td><td>{r.country_abbr}</td><td>{r.position}</td><td>${r.price_millions:.1f}</td><td>{r.ownership_fraction*100:.1f}%</td><td><b>{r.md1}</b></td><td>{fixture(pid, FIRST_ROUND)}</td><td>{e_e:.2f}</td></tr>")
html.append("</tbody></table>")

html.append("</body></html>")

out = f"results/{hostname()}_md2_dashboard_{NOW}.html"
with open(out, "w") as f:
    f.write("\n".join(html))
print("Dashboard:", out)
