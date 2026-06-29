"""Explore every feasible 2-transfer (and 3-transfer-with-hit) plan with
Dembélé LOCKED IN. User's call: drop Doué not Dembélé.

Then run Monte Carlo on the top survivors to compare expected points,
downside (p10), and upside (p90), so the captain + transfer pick is data-driven.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import pulp

from fifa_fantasy.optimizer.solvers import (
    SQUAD_SIZE, SQUAD_POSITION_COUNTS, solve_lineup, TRANSFER_HIT_POINTS,
)

# ---------- Load data ----------
h = pd.read_parquet('/tmp/h4.parquet')
g = pd.read_parquet('/tmp/g4.parquet')
p = pd.read_parquet('/tmp/p4.parquet')
raw = pd.read_parquet('data/raw/players_2026-06-24.parquet')

md3_h = h[h.round_id == 3].set_index('player_id')
md3_g = g[g.round_id == 3].set_index('player_id')
md3_p = p[p.round_id == 3].set_index('player_id')

df = md3_h[['full_name','country','country_abbr','position','price_millions',
            'is_home','opponent_abbr','predicted_points']].copy()
df = df.rename(columns={'predicted_points': 'heur'})
df['gbm']  = md3_g['predicted_points']
df['pois'] = md3_p['predicted_points']
df = df.reset_index()

ROT = {'ARG':0.78, 'FRA':0.93, 'GER':0.55, 'USA':0.65, 'MEX':0.70, 'NOR':0.93,
       'CUW':0.70, 'JOR':0.85, 'IRQ':0.75, 'TUN':0.85, 'KSA':0.85, 'UZB':0.80,
       'HAI':0.70, 'QAT':0.80, 'BIH':0.85, 'CZE':0.90, 'TUR':0.85}
df['risk'] = df['country_abbr'].map(ROT).fillna(1.0)
FORCE = ['Lionel Messi','Kylian Mbappé','Cristiano Ronaldo','Mohamed Salah',
         'Harry Kane','Erling Haaland','Jude Bellingham','Vinícius Júnior',
         'Cody Gakpo','Ousmane Dembélé']  # Dembélé forced full (FRA top-spot starter)
df.loc[df['full_name'].isin(FORCE), 'risk'] = 1.0

df['form'] = df['player_id'].map(raw.set_index('player_id')['form'].to_dict())
# Realised pts so far is a stronger signal than model for elite players.
df['total_pts'] = df['player_id'].map(raw.set_index('player_id')['total_points'].to_dict())
df['ownership_pct'] = df['player_id'].map(raw.set_index('player_id')['ownership_fraction'].to_dict()) * 100

df['ens'] = (df['heur'] + df['gbm']) / 2
df['adj'] = df['ens'] * df['risk']
# Form bump is now MD-aware: realised average so far drives a stronger anchor
# for elite players (15+ total pts). Bigger weight than before.
df['avg_per_md'] = (df['total_pts'] / 2).clip(lower=0)
df['final'] = (
    df['adj'] * 0.55
    + df['avg_per_md'] * 0.30
    + df['form'].clip(0, 15) * 0.15
)

meta = raw.set_index('player_id')
df['is_eliminated'] = df['player_id'].map(meta['is_eliminated']).astype(bool)

SQUAD_IDS = [45,1709,2053,523,521,505,57,517,501,804,1338,1523,918,1711,2000]
def nid(n):
    m = raw[raw['full_name'] == n]
    return int(m.iloc[0]['player_id']) if len(m) else None

LAUTARO = 1338; DEMBELE = 501; DOUE = 505; RAUM = 521; TAH = 523
WATKINS = 1711; OLISE = 517
MESSI = nid('Lionel Messi'); BELL = nid('Jude Bellingham')
SALAH = nid('Mohamed Salah'); KANE = nid('Harry Kane')
RONALDO = nid('Cristiano Ronaldo'); RASHFORD = nid('Marcus Rashford')
EZE = nid('Eberechi Eze'); CUNHA = nid('Matheus Cunha')

# ---------- Solver ----------
def solve(locked, forced, max_t=2):
    cands = df[~df['is_eliminated']].reset_index(drop=True)
    prob = pulp.LpProblem('p', pulp.LpMaximize)
    x = {int(r.player_id): pulp.LpVariable(f'x_{int(r.player_id)}', cat='Binary')
         for r in cands.itertuples()}
    extra = pulp.LpVariable('extra', lowBound=0, cat='Continuous')
    prob += (pulp.lpSum(x[int(r.player_id)] * float(r.final)
                        for r in cands.itertuples())
             - TRANSFER_HIT_POINTS * extra)
    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    for pos, count in SQUAD_POSITION_COUNTS.items():
        ids = [int(r.player_id) for r in cands.itertuples() if r.position == pos.value]
        prob += pulp.lpSum(x[i] for i in ids) == count
    prob += pulp.lpSum(x[int(r.player_id)] * float(r.price_millions)
                       for r in cands.itertuples()) <= 100.0
    for c, gp in cands.groupby('country', sort=False):
        ids = [int(pid) for pid in gp['player_id']]
        prob += pulp.lpSum(x[i] for i in ids) <= 3
    for pid in forced:
        if pid in x: prob += x[pid] == 0
    for pid in locked:
        if pid in x: prob += x[pid] == 1
    nps = pulp.lpSum(x[int(r.player_id)] for r in cands.itertuples()
                     if int(r.player_id) not in set(SQUAD_IDS))
    prob += nps <= max_t
    prob += nps - 2 <= extra
    st = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[st] != 'Optimal':
        return None
    chosen = sorted(int(pid) for pid, v in x.items() if v.value() > 0.5)
    return chosen, float(pulp.value(prob.objective)), int(round(extra.value()))

# Dembélé must stay; explore all feasible 2-transfer combos.
LOCKS = [DEMBELE]

scenarios = {}
def add(label, locked, forced, max_t=2, notes=''):
    res = solve(locked + LOCKS, forced, max_t)
    scenarios[label] = (res, notes)

# Single-premium options
add('A: Drop Lautaro+Doué, IN Messi + cheaper MID',
    [MESSI], [LAUTARO, DOUE])
add('B: Drop Lautaro+Doué, IN Bellingham + cheaper FWD',
    [BELL], [LAUTARO, DOUE])
add('C: Drop Lautaro+Doué, IN Salah + cheaper FWD',
    [SALAH], [LAUTARO, DOUE])
add('D: Drop Lautaro+Doué, IN Kane + cheaper MID',
    [KANE], [LAUTARO, DOUE])

# Combo within Doué+Raum drops (no Lautaro)
add('E: Drop Doué+Raum, IN Bellingham + better DEF',
    [BELL], [DOUE, RAUM])
add('F: Drop Doué+Raum, IN Salah + DEF',
    [SALAH], [DOUE, RAUM])

# Drop Watkins-related
add('G: Drop Doué+Watkins, IN Bellingham + FWD',
    [BELL], [DOUE, WATKINS])
add('H: Drop Watkins+Lautaro, IN Messi + FWD (keeps Doué!)',
    [MESSI], [WATKINS, LAUTARO])

# Dembélé+Lautaro out — for reference (violates Dembélé lock by definition,
# so we skip the lock here as a baseline)
res = solve([MESSI, BELL], [LAUTARO, DEMBELE], 2)
scenarios['Z: BASELINE (drop Dembélé) — both stars'] = (res, 'reference only, ignores Dembélé-lock')

# 3-transfer hits
add('I: 3-trf hit | Drop Lautaro+Doué+Raum, IN Messi+Bellingham+DEF',
    [MESSI, BELL], [LAUTARO, DOUE, RAUM], max_t=3)
add('J: 3-trf hit | Drop Lautaro+Doué+Watkins, IN Messi+Bellingham+FWD',
    [MESSI, BELL], [LAUTARO, DOUE, WATKINS], max_t=3)
add('K: 3-trf hit | Drop Lautaro+Doué+Tah, IN Messi+Bellingham+DEF',
    [MESSI, BELL], [LAUTARO, DOUE, TAH], max_t=3)

# ---------- Print scenarios ----------
print('=' * 100)
print('FEASIBLE 2-TRANSFER PLANS (Dembélé locked in)')
print('=' * 100)
ranked = []
for label, (res, notes) in scenarios.items():
    if res is None:
        print(f'\n{label}: INFEASIBLE  {notes}')
        continue
    chosen, obj, ex = res
    tin = sorted(set(chosen) - set(SQUAD_IDS))
    tout = sorted(set(SQUAD_IDS) - set(chosen))
    sq = df[df.player_id.isin(chosen)][['player_id','full_name','position','country_abbr',
                                         'price_millions','final','is_home','opponent_abbr']
                                       ].rename(columns={'final':'predicted_points'})
    lin = solve_lineup(sq)
    cap = sq[sq.player_id == lin.captain_id].iloc[0]
    vc  = sq[sq.player_id == lin.vice_captain_id].iloc[0]
    ranked.append((label, obj, ex, tin, tout, lin, cap, vc, chosen))
    print(f'\n--- {label} --- obj={obj:.2f} trf={len(tin)} hit=-{TRANSFER_HIT_POINTS*ex}')
    if notes: print(f'  {notes}')
    for pid in tin:
        r = df[df.player_id==pid].iloc[0]
        print(f'  IN  {r["full_name"]:<22} {r["country_abbr"]} {r["position"]} '
              f'${r.price_millions:>4.1f}M own={r.ownership_pct:>4.1f}% tot={int(r.total_pts):>3} '
              f'form={r.form:.1f} final={r.final:.2f}')
    for pid in tout:
        r = df[df.player_id==pid]
        if r.empty: continue
        r = r.iloc[0]
        print(f'  OUT {r["full_name"]:<22} {r["country_abbr"]} {r["position"]} '
              f'${r.price_millions:>4.1f}M tot={int(r.total_pts):>3} final={r.final:.2f}')
    print(f'  XI E={lin.objective:.2f} form={lin.formation}  '
          f'Cap: {cap["full_name"]} ({cap["country_abbr"]}) E={cap["predicted_points"]:.2f}  '
          f'Vice: {vc["full_name"]} E={vc["predicted_points"]:.2f}')

# ---------- Monte Carlo simulation on top 5 plans ----------
print('\n' + '=' * 100)
print('MONTE CARLO: 10,000 simulated MD3 rounds for top survivors')
print('=' * 100)

ranked = sorted(ranked, key=lambda x: -x[1])
top5 = ranked[:5]

# Per-player simulation distribution. Use a noise model anchored to:
# - mean = final score
# - std proportional to ceiling (premium players have higher variance)
# - rotation risk reduces the *probability of playing* not the per-play score
RNG = np.random.default_rng(seed=42)

def simulate_player(player_row, captain_id=None):
    """Sample one MD3 outcome for the player. Returns sampled points."""
    pid = int(player_row['player_id'])
    risk = float(player_row['risk'])
    if risk < 0.55:
        # Heavy rotation: 30% chance of bench, 70% chance of cameo
        plays = RNG.random() < (0.40 if pid not in {LAUTARO} else 0.95)
    elif risk < 0.80:
        plays = RNG.random() < 0.75
    else:
        plays = RNG.random() < 0.97
    if not plays:
        return 0.0
    mean = float(player_row['final']) / max(risk, 0.5)  # de-risked mean (already played)
    # Sigma: bigger for attackers + premium price
    sigma = 2.5 + 0.3 * float(player_row['price_millions'])
    if player_row['position'] in ('FWD','MID'):
        sigma += 1.0
    pts = RNG.normal(loc=mean, scale=sigma)
    # Heavy-tail bonus: 5% chance of explosion
    if RNG.random() < 0.05:
        pts += abs(RNG.normal(8, 3))
    return max(0.0, pts)

def simulate_plan(label, chosen, lin, captain_override=None):
    sq = df[df.player_id.isin(chosen)].reset_index(drop=True)
    starters = set(lin.starter_ids)
    captain_id = captain_override if captain_override else lin.captain_id
    vice_id = lin.vice_captain_id
    bench_order = lin.bench_ids
    sims = []
    for _ in range(10_000):
        # Sim per-player points
        pts = {int(r.player_id): simulate_player(r) for _, r in sq.iterrows()}
        # Auto-sub: if any starter scored 0 (didn't play), substitute from bench
        zero_starters = [pid for pid in starters if pts[pid] == 0]
        bench_pool = list(bench_order)
        for s in zero_starters:
            # Find an eligible bench replacement that played
            star_pos = sq[sq.player_id == s].iloc[0]['position']
            for b in bench_pool:
                if pts[b] > 0:
                    b_pos = sq[sq.player_id == b].iloc[0]['position']
                    # Simple: same position swap. GK only swap with GK.
                    if (star_pos == 'GK') == (b_pos == 'GK'):
                        pts[s] = pts[b]
                        pts[b] = 0
                        bench_pool.remove(b)
                        break
        # Captain x2 (vice if captain blanked)
        cap_pts = pts[captain_id] if pts[captain_id] > 0 else pts[vice_id]
        starters_total = sum(pts[s] for s in starters)
        total = starters_total + cap_pts  # captain counted twice
        sims.append(total)
    arr = np.array(sims)
    return arr.mean(), np.percentile(arr,10), np.percentile(arr,50), np.percentile(arr,90), arr.std()

print(f'\n{"label":<60} {"mean":>7} {"p10":>7} {"med":>7} {"p90":>7} {"std":>6} {"cap"}')
print('-' * 110)
results = []
for label, obj, ex, tin, tout, lin, cap, vc, chosen in top5:
    # Apply hit
    hit = TRANSFER_HIT_POINTS * ex
    mu, p10, p50, p90, sd = simulate_plan(label, chosen, lin)
    mu, p10, p50, p90 = mu - hit, p10 - hit, p50 - hit, p90 - hit
    results.append((label, mu, p10, p50, p90, sd, cap['full_name'], chosen, lin))
    print(f'{label[:60]:<60} {mu:>7.2f} {p10:>7.2f} {p50:>7.2f} {p90:>7.2f} {sd:>6.2f}  {cap["full_name"]}')

# Now also simulate with MESSI as captain in plans where lineup picked Bellingham
print('\nALT CAPTAIN — same plans, force Messi as captain when feasible:')
for label, obj, ex, tin, tout, lin, cap, vc, chosen in top5:
    if MESSI not in chosen:
        continue
    hit = TRANSFER_HIT_POINTS * ex
    mu, p10, p50, p90, sd = simulate_plan(label, chosen, lin, captain_override=MESSI)
    mu, p10, p50, p90 = mu - hit, p10 - hit, p50 - hit, p90 - hit
    print(f'{label[:60]+" [Messi cap]":<60} {mu:>7.2f} {p10:>7.2f} {p50:>7.2f} {p90:>7.2f} {sd:>6.2f}  Messi')

# Also try with BELLINGHAM captain forced
print('\nALT CAPTAIN — same plans, force Bellingham as captain when feasible:')
for label, obj, ex, tin, tout, lin, cap, vc, chosen in top5:
    if BELL not in chosen:
        continue
    hit = TRANSFER_HIT_POINTS * ex
    mu, p10, p50, p90, sd = simulate_plan(label, chosen, lin, captain_override=BELL)
    mu, p10, p50, p90 = mu - hit, p10 - hit, p50 - hit, p90 - hit
    print(f'{label[:60]+" [Bell cap]":<60} {mu:>7.2f} {p10:>7.2f} {p50:>7.2f} {p90:>7.2f} {sd:>6.2f}  Bellingham')

# Also try Olise captain
print('\nALT CAPTAIN — same plans, force Olise as captain:')
for label, obj, ex, tin, tout, lin, cap, vc, chosen in top5:
    if OLISE not in chosen:
        continue
    hit = TRANSFER_HIT_POINTS * ex
    mu, p10, p50, p90, sd = simulate_plan(label, chosen, lin, captain_override=OLISE)
    mu, p10, p50, p90 = mu - hit, p10 - hit, p50 - hit, p90 - hit
    print(f'{label[:60]+" [Olise cap]":<60} {mu:>7.2f} {p10:>7.2f} {p50:>7.2f} {p90:>7.2f} {sd:>6.2f}  Olise')
