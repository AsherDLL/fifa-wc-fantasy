"""R32 full-squad rebuild with Wildcard.

Constraints:
- 15-player squad: 2 GK / 5 DEF / 5 MID / 3 FWD
- Max 3 per country
- Budget $107.5M (R32 base $105 + boost $2.5, or user can adjust)
- Eliminated teams' players banned
- Single round horizon (R32 = round_id 4)
- Knockout: every team plays full strength, no rotation worry

Scoring blend: realised production carries the most weight in elimination
games. We use:
  final = 0.55 * ensemble_predicted  +  0.30 * avg_pts_per_game  +  0.15 * form

Then optimize. Outputs ranked plans and a captain board.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import pulp
from fifa_fantasy.optimizer.solvers import (
    SQUAD_SIZE, SQUAD_POSITION_COUNTS, solve_lineup,
)

R32_ROUND = 4
BUDGET = 107.5  # base 105 + 2.5 boost; adjust if user clarifies
MAX_PER_COUNTRY = 3

h = pd.read_parquet('/tmp/h_r32.parquet')
g = pd.read_parquet('/tmp/g_r32.parquet')
p = pd.read_parquet('/tmp/p_r32.parquet')
raw = pd.read_parquet('data/raw/players_2026-06-28.parquet')

md_h = h[h.round_id == R32_ROUND].set_index('player_id')
md_g = g[g.round_id == R32_ROUND].set_index('player_id')
md_p = p[p.round_id == R32_ROUND].set_index('player_id')

df = md_h[['full_name','country','country_abbr','position','price_millions',
           'is_home','opponent_abbr','predicted_points']].copy()
df = df.rename(columns={'predicted_points': 'heur'})
df['gbm']  = md_g['predicted_points']
df['pois'] = md_p['predicted_points']
df = df.reset_index()

# Realised performance carries the day in knockouts.
df['total_pts']  = df['player_id'].map(raw.set_index('player_id')['total_points'].to_dict())
df['ownership']  = df['player_id'].map(raw.set_index('player_id')['ownership_fraction'].to_dict()) * 100
df['form']       = df['player_id'].map(raw.set_index('player_id')['form'].to_dict())
df['status']     = df['player_id'].map(raw.set_index('player_id')['status'].to_dict())
df['is_eliminated'] = df['player_id'].map(raw.set_index('player_id')['is_eliminated'].to_dict()).astype(bool)

df['avg_per_md'] = (df['total_pts'] / 3).clip(lower=0)
df['ens'] = (df['heur'] + df['gbm']) / 2
df['final'] = (
    df['ens'] * 0.55
    + df['avg_per_md'] * 0.30
    + df['form'].clip(0, 15) * 0.15
)

# Eligibility: alive, playing status.
elig = df[(~df['is_eliminated']) & (df['status'] == 'playing')].reset_index(drop=True)

# ---------- Solve the fresh squad ----------
def solve_fresh():
    prob = pulp.LpProblem('r32_fresh', pulp.LpMaximize)
    x = {int(r.player_id): pulp.LpVariable(f'x_{int(r.player_id)}', cat='Binary')
         for r in elig.itertuples()}
    prob += pulp.lpSum(x[int(r.player_id)] * float(r.final) for r in elig.itertuples())
    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    for pos, count in SQUAD_POSITION_COUNTS.items():
        ids = [int(r.player_id) for r in elig.itertuples() if r.position == pos.value]
        prob += pulp.lpSum(x[i] for i in ids) == count
    prob += pulp.lpSum(x[int(r.player_id)] * float(r.price_millions)
                       for r in elig.itertuples()) <= BUDGET
    for c, gp in elig.groupby('country', sort=False):
        ids = [int(pid) for pid in gp['player_id']]
        prob += pulp.lpSum(x[i] for i in ids) <= MAX_PER_COUNTRY
    st = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[st] == 'Optimal', pulp.LpStatus[st]
    chosen = sorted(int(pid) for pid, v in x.items() if v.value() > 0.5)
    return chosen, float(pulp.value(prob.objective))

chosen, obj = solve_fresh()
print('=' * 100)
print(f'R32 OPTIMAL FRESH SQUAD (Wildcard, ${BUDGET}M budget, max {MAX_PER_COUNTRY}/country)')
print('=' * 100)
sq = elig[elig.player_id.isin(chosen)].sort_values(['position','final'], ascending=[True,False])
print(f'{"name":<22} {"cty":<4} {"pos":<4} {"$M":>5} {"tot":>4} {"form":>5} {"own%":>5} {"heur":>5} {"gbm":>5} {"final":>6} {"fix":<8}')
print('-' * 105)
total_cost = 0
for r in sq.itertuples():
    fix_str = f'{"vs " if r.is_home else "@ "}{r.opponent_abbr}'
    print(f'{r.full_name:<22} {r.country_abbr:<4} {r.position:<4} {r.price_millions:>5.1f} {int(r.total_pts):>4} {r.form:>5.1f} {r.ownership:>5.1f} {r.heur:>5.2f} {r.gbm:>5.2f} {r.final:>6.2f} {fix_str:<8}')
    total_cost += r.price_millions
print(f'\nTotal cost: ${total_cost:.1f}M / ${BUDGET}M  (slack: ${BUDGET-total_cost:.1f}M)')
print(f'Sum final: {obj:.2f}')

# ---------- Best XI + captain ----------
md = elig[elig.player_id.isin(chosen)][
    ['player_id','full_name','position','country_abbr','price_millions','final','is_home','opponent_abbr']
].rename(columns={'final':'predicted_points'})
lineup = solve_lineup(md)
cap = md[md.player_id == lineup.captain_id].iloc[0]
vc  = md[md.player_id == lineup.vice_captain_id].iloc[0]
print(f'\nBest XI: {lineup.objective:.2f} pts in {lineup.formation}')
print(f'Captain: {cap["full_name"]} ({cap["country_abbr"]}) E={cap["predicted_points"]:.2f}')
print(f'Vice:    {vc["full_name"]} ({vc["country_abbr"]}) E={vc["predicted_points"]:.2f}')
print(f'Bench (auto-sub order):')
for pid in lineup.bench_ids:
    r = md[md.player_id == pid].iloc[0]
    print(f'  - {r["full_name"]} ({r["country_abbr"]} {r["position"]}) E={r["predicted_points"]:.2f}')

# ---------- Top 10 candidates per position so user can substitute ----------
print('\n' + '=' * 100)
print('TOP 10 CANDIDATES PER POSITION (alive, by final score)')
print('=' * 100)
for pos in ('GK','DEF','MID','FWD'):
    print(f'\n--- {pos} ---')
    sub = elig[elig.position == pos].sort_values('final', ascending=False).head(10)
    in_squad_set = set(chosen)
    for r in sub.itertuples():
        flag = ' ←' if r.player_id in in_squad_set else ''
        fix_str = f'{"vs " if r.is_home else "@ "}{r.opponent_abbr}'
        print(f'  {r.full_name:<22} {r.country_abbr:<4} ${r.price_millions:>4.1f}M tot={int(r.total_pts):>3} form={r.form:>4.1f} own={r.ownership:>5.1f}% final={r.final:.2f} {fix_str:<8}{flag}')

# ---------- Captain board (top scorers among alive ranked by realised total) ----------
print('\n' + '=' * 100)
print('CAPTAIN BOARD (alive players ranked by realised total points)')
print('=' * 100)
cap_board = elig.sort_values('total_pts', ascending=False).head(15)
print(f'{"name":<22} {"cty":<4} {"pos":<4} {"$M":>5} {"tot":>4} {"form":>5} {"own%":>5} {"final":>6} {"fix":<8}')
for r in cap_board.itertuples():
    fix_str = f'{"vs " if r.is_home else "@ "}{r.opponent_abbr}'
    print(f'{r.full_name:<22} {r.country_abbr:<4} {r.position:<4} {r.price_millions:>5.1f} {int(r.total_pts):>4} {r.form:>5.1f} {r.ownership:>5.1f} {r.final:>6.2f} {fix_str:<8}')

# ---------- Country survival overview ----------
print('\n' + '=' * 100)
print('R32 FIXTURE OUTLOOK (alive teams + opponent Elo gap)')
print('=' * 100)
fixtures = elig[['country_abbr','opponent_abbr','is_home']].drop_duplicates().sort_values('country_abbr')
# Elo: get from features parquet
feat = pd.read_parquet('data/processed/features_2026-06-28.parquet')
feat_r32 = feat[feat.round_id == R32_ROUND][['country_abbr','opponent_abbr','country_elo','opp_country_elo','country_elo_diff','is_home']].drop_duplicates()
feat_r32 = feat_r32.sort_values('country_elo_diff', ascending=False)
print(f'{"home":<5} {"cty":<4} {"opp":<4} {"elo":>7} {"opp_elo":>7} {"diff":>6}')
for r in feat_r32.itertuples():
    h_mark = 'H' if r.is_home else 'A'
    print(f'  {h_mark:<3} {r.country_abbr:<4} {r.opponent_abbr:<4} {r.country_elo:>7.0f} {r.opp_country_elo:>7.0f} {r.country_elo_diff:>+6.0f}')
