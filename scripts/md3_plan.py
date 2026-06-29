"""MD3 transfer plan with rotation-risk adjustment.

Key insight for MD3: teams that have clinched their group with 6 pts and
already secured top spot will rotate. Star scorers (Messi, Mbappé) keep
playing because of the top-scorer race. Teams chasing qualification
(BRA, ENG, ESP, EGY, NED, JPN, POR, COL) play full strength.

Plan adjusts model predictions by a rotation-risk haircut, then runs the
optimizer with the user's MD2 squad as the starting point.
"""
from __future__ import annotations
import pandas as pd
import pulp

from fifa_fantasy.optimizer.pipeline import apply_scouting_bonus
from fifa_fantasy.optimizer.solvers import (
    SQUAD_SIZE, SQUAD_POSITION_COUNTS, solve_lineup,
)

# User's MD2 squad (no transfers were made for MD3 since they're planning now)
SQUAD = {
    45:   ("Emiliano Martínez",  "ARG", "GK",  "Start"),
    1709: ("Reece James",        "ENG", "DEF", "Start"),
    2053: ("Willian Pacho",      "ECU", "DEF", "Start"),
    523:  ("Jonathan Tah",       "GER", "DEF", "Start"),
    521:  ("David Raum",         "GER", "DEF", "Start"),
    505:  ("Désiré Doué",        "FRA", "MID", "Start"),
    57:   ("Enzo Fernández",     "ARG", "MID", "Start"),
    517:  ("Michael Olise",      "FRA", "MID", "CAPTAIN"),  # Olise was MD2 captain
    501:  ("Ousmane Dembélé",    "FRA", "MID", "Vice"),
    804:  ("Cody Gakpo",         "NED", "FWD", "Start"),
    1338: ("Lautaro Martínez",   "ARG", "FWD", "Start"),
    1523: ("Mike Penders",       "BEL", "GK",  "Bench"),
    918:  ("Diogo Dalot",        "POR", "DEF", "Bench"),
    1711: ("Ollie Watkins",      "ENG", "FWD", "Bench"),
    2000: ("Nico Williams",      "ESP", "MID", "Bench"),
}
SQUAD_IDS = list(SQUAD.keys())

# Rotation risk multiplier per country, for MD3 ONLY.
# 1.0 = full strength expected
# 0.7 = mild rotation (top players play but some starters rest)
# 0.5 = significant rotation (best XI not expected)
# 0.0 = effectively no game (e.g. eliminated and given up)
#
# Standings rationale below. Stars who are in the top-scorer race
# (Messi, Mbappé) override the country haircut at the player level.
ROTATION_RISK: dict[str, float] = {
    # Clinched + top spot likely:
    "ARG": 0.70,  # 6 pts, vs JOR (0 pts), top of group secured
    "FRA": 0.80,  # 6 pts, vs NOR (6 pts) - top-spot battle, may rotate moderately
    "GER": 0.55,  # 6 pts, vs ECU (1 pt), top secured - HEAVY rotation expected
    "USA": 0.60,  # 6 pts, vs TUR (0 pts), top secured
    "MEX": 0.65,  # 6 pts, vs CZE (1 pt)
    "NOR": 0.80,  # 6 pts, vs FRA (6 pts) - top-spot battle
    # Still fighting for qualification or top spot:
    "BRA": 1.00,  # 4 pts, vs SCO (3 pts), need result
    "ENG": 1.00,  # 4 pts, vs PAN (0 pts), need to be sure
    "ESP": 1.00,  # 4 pts, vs URU (2 pts)
    "NED": 0.95,  # 4 pts, vs TUN (0 pts), still wants top
    "JPN": 1.00,  # 4 pts, vs SWE (3 pts), tight
    "POR": 1.00,  # 4 pts, vs COL (3 pts)
    "COL": 1.00,
    "EGY": 1.00,
    "CIV": 1.00,
    "AUS": 1.00,
    "CAN": 1.00,
    "SUI": 1.00,
    "MAR": 1.00,
    "SCO": 1.00,
    "PAR": 1.00,
    "BEL": 1.00,
    "URU": 1.00,
    "CPV": 1.00,
    "IRN": 1.00,
    "NZL": 1.00,
    "ALG": 1.00,
    "AUT": 1.00,
    "COD": 1.00,
    "GHA": 1.00,
    "CRO": 1.00,
    "PAN": 1.00,
    "ECU": 1.00,  # needs win to advance as best 3rd
    "KOR": 1.00,
    "CZE": 0.90,
    "RSA": 1.00,
    "BIH": 0.85,
    "QAT": 0.85,
    "PAR": 0.90,
    "TUR": 0.85,
    # Realistically eliminated:
    "HAI": 0.70,
    "IRQ": 0.75,
    "JOR": 0.80,
    "SEN": 0.95,  # still fighting for best 3rd
    "TUN": 0.85,
    "KSA": 0.85,
    "UZB": 0.80,
    "CUW": 0.75,
}
# Player-level overrides: top-scorer chasers play through rotation.
PLAYER_FORCE_PLAY: set[int] = set()  # filled below by name

PROTECT_NAMES = {
    "Lionel Messi", "Kylian Mbappé", "Lautaro Martínez", "Cristiano Ronaldo",
    "Harry Kane", "Bukayo Saka", "Mohamed Salah", "Erling Haaland",
    # Olise scored MD2; Argentina's top scorers chase Messi.
}

def main():
    h = pd.read_parquet('/tmp/h3.parquet')
    p = pd.read_parquet('/tmp/p3.parquet')
    g = pd.read_parquet('/tmp/g3.parquet')
    md3 = 3

    # Ensemble of (heuristic + GBM) for stability. Poisson is too aggressive
    # post-MD2 (mean 6.03 vs 2.5 heuristic); using it would overweight
    # cheap players. Use heur+gbm as the primary ensemble, expose Poisson
    # as a tiebreaker.
    keys = ['player_id','full_name','country_abbr','position','price_millions']
    e = h[h.round_id==md3][keys+['predicted_points','is_home','opponent_abbr']].copy()
    e = e.rename(columns={'predicted_points':'heur'})
    e['pois'] = h[h.round_id==md3]['predicted_points'].values  # placeholder
    e['pois'] = e.merge(p[p.round_id==md3][['player_id','predicted_points']]
                        .rename(columns={'predicted_points':'pois'}),
                        on='player_id', how='left')['pois_y'] if False else \
                e['player_id'].map(p[p.round_id==md3].set_index('player_id')['predicted_points'])
    e['gbm']  = e['player_id'].map(g[g.round_id==md3].set_index('player_id')['predicted_points'])

    e['ens_2'] = (e['heur'] + e['gbm']) / 2
    e['ens_3'] = (e['heur'] + e['pois'] + e['gbm']) / 3

    # Apply rotation risk haircut.
    e['risk'] = e['country_abbr'].map(ROTATION_RISK).fillna(1.0)
    raw = pd.read_parquet('data/raw/players_2026-06-23.parquet')
    name_to_id = raw.set_index('full_name')['player_id'].to_dict()
    forced = {name_to_id[n] for n in PROTECT_NAMES if n in name_to_id}
    e.loc[e['player_id'].isin(forced), 'risk'] = 1.0
    e['adj_h'] = e['heur'] * e['risk']
    e['adj_g'] = e['gbm'] * e['risk']
    e['adj_p'] = e['pois'] * e['risk']
    e['adj_ens'] = (e['adj_h'] + e['adj_g']) / 2

    # Player metadata for budget/country/eliminated
    meta = raw.set_index('player_id')

    # MD3 horizon = (3,) so we just use the MD3 predicted points directly.
    e['total_effective_points'] = e['adj_ens']
    e['is_eliminated'] = e['player_id'].map(meta['is_eliminated']).astype(bool)
    e['country'] = e['player_id'].map(meta['country'])
    # If a country isn't playing MD3 (already eliminated from group? all 48
    # do play MD3 in groups, so all squads have rows.)

    print('=== Top 30 candidates for MD3 (adj ensemble, NOT in current squad) ===')
    pool = e[~e['player_id'].isin(SQUAD_IDS) & (~e['is_eliminated'])]
    print(pool.sort_values('adj_ens', ascending=False).head(30)[
        ['full_name','country_abbr','position','price_millions','is_home',
         'opponent_abbr','heur','gbm','risk','adj_ens']
    ].to_string(index=False))

    print()
    print('=== Current squad MD3 outlook ===')
    sq = e[e['player_id'].isin(SQUAD_IDS)].copy()
    sq['role'] = sq['player_id'].map({k: v[3] for k, v in SQUAD.items()})
    print(sq.sort_values('adj_ens', ascending=False)[
        ['role','full_name','country_abbr','position','price_millions','is_home',
         'opponent_abbr','heur','gbm','risk','adj_ens']
    ].to_string(index=False))

    # Solve 2-transfer move using risk-adjusted ensemble.
    BUDGET = 100.0
    MAX_PER_COUNTRY = 3
    MAX_TRANSFERS = 2

    def solve(locked_in: list[int], forced_out: list[int]):
        candidates = e.copy()
        candidates['country'] = candidates['player_id'].map(meta['country'])
        prob = pulp.LpProblem('md3', pulp.LpMaximize)
        x = {int(r.player_id): pulp.LpVariable(f'x_{int(r.player_id)}', cat='Binary')
             for r in candidates.itertuples()}
        prob += pulp.lpSum(x[int(r.player_id)] * float(r.total_effective_points)
                           for r in candidates.itertuples())
        prob += pulp.lpSum(x.values()) == SQUAD_SIZE
        for pos, count in SQUAD_POSITION_COUNTS.items():
            ids = [int(r.player_id) for r in candidates.itertuples() if r.position == pos.value]
            prob += pulp.lpSum(x[i] for i in ids) == count
        prob += pulp.lpSum(x[int(r.player_id)] * float(r.price_millions)
                           for r in candidates.itertuples()) <= BUDGET
        for c, gp in candidates.groupby('country', sort=False):
            ids = [int(pid) for pid in gp['player_id']]
            prob += pulp.lpSum(x[i] for i in ids) <= MAX_PER_COUNTRY
        for pid in forced_out:
            if pid in x: prob += x[pid] == 0
        for pid in locked_in:
            if pid in x: prob += x[pid] == 1
        # Drop eliminated.
        for r in candidates.itertuples():
            if bool(r.is_eliminated):
                prob += x[int(r.player_id)] == 0
        new_picks = pulp.lpSum(x[int(r.player_id)] for r in candidates.itertuples()
                               if int(r.player_id) not in set(SQUAD_IDS))
        prob += new_picks <= MAX_TRANSFERS
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != 'Optimal':
            return None
        return sorted(int(pid) for pid, v in x.items() if v.value() > 0.5), float(pulp.value(prob.objective))

    # === SCENARIOS ===
    print()
    print('=' * 90)
    print('MD3 TRANSFER SCENARIOS')
    print('=' * 90)

    def describe(label, result, notes=''):
        if result is None:
            print(f'\n{label}: INFEASIBLE  {notes}')
            return
        chosen, obj = result
        tin = sorted(set(chosen) - set(SQUAD_IDS))
        tout = sorted(set(SQUAD_IDS) - set(chosen))
        print(f'\n--- {label} ---  (MD3 expected ensemble: {obj:.2f})')
        if notes: print(f'  {notes}')
        for pid in tin:
            r = e[e.player_id==pid].iloc[0]
            print(f'    IN  {r["full_name"]:<22} {r["country_abbr"]:<4} {r["position"]:<4} ${r["price_millions"]:.1f}M  '
                  f'heur={r["heur"]:.1f} gbm={r["gbm"]:.1f} risk={r["risk"]:.2f} adj={r["adj_ens"]:.2f}')
        for pid in tout:
            nm, cty, pos, role = SQUAD[pid]
            r = e[e.player_id==pid].iloc[0] if len(e[e.player_id==pid]) else None
            adj = r['adj_ens'] if r is not None else 0
            risk = r['risk'] if r is not None else 1
            print(f'    OUT {nm:<22} {cty:<4} {pos:<4} risk={risk:.2f} adj={adj:.2f}')
        # MD3 XI + captain
        md3_squad = e[e.player_id.isin(chosen)][
            ['player_id','full_name','position','country_abbr','price_millions','adj_ens','is_home','opponent_abbr']
        ].rename(columns={'adj_ens':'predicted_points'})
        lineup = solve_lineup(md3_squad)
        cap = md3_squad[md3_squad.player_id==lineup.captain_id].iloc[0]
        vc  = md3_squad[md3_squad.player_id==lineup.vice_captain_id].iloc[0]
        print(f'  MD3 XI E: {lineup.objective:.2f}  form: {lineup.formation}')
        print(f'  Captain: {cap["full_name"]} ({cap["country_abbr"]}) adj E={cap["predicted_points"]:.2f}')
        print(f'  Vice   : {vc["full_name"]} ({vc["country_abbr"]}) adj E={vc["predicted_points"]:.2f}')

    # S1: model's free pick (no locks, no forced)
    describe('S1: Free pick', solve([], []), 'optimizer chooses both transfers')

    # S2: drop Lautaro (rotation risk) + drop Raum (rotation risk, lowest perf)
    describe('S2: Drop Lautaro + Raum',
             solve([], [1338, 521]),
             'OUT both high-rotation-risk + low MD2 performers')

    # S3: drop Tah + Raum (both GER, both rotation risk)
    describe('S3: Drop Tah + Raum (both GER rotation)',
             solve([], [523, 521]))

    # S4: drop Watkins + Lautaro
    describe('S4: Drop Watkins + Lautaro',
             solve([], [1711, 1338]))

    # S5: Messi locked in (use one transfer to bring him)
    messi_id = name_to_id.get('Lionel Messi')
    if messi_id:
        describe('S5: Lock Messi in',
                 solve([messi_id], [1338]),  # drop Lautaro to fit ARG cap
                 'Lautaro → Messi (ARG-for-ARG); optimizer picks 2nd')

    # S6: Kane locked in
    kane_id = name_to_id.get('Harry Kane')
    if kane_id:
        describe('S6: Lock Kane in',
                 solve([kane_id], [1711]),
                 'Watkins → Kane (ENG-for-ENG); optimizer picks 2nd')

    # S7: Ronaldo locked in
    ron_id = name_to_id.get('Cristiano Ronaldo')
    if ron_id:
        describe('S7: Lock Ronaldo in',
                 solve([ron_id], [918]),
                 'Dalot → Ronaldo (POR-for-POR); optimizer picks 2nd')

    # S8: Messi + Kane (two premiums)
    if messi_id and kane_id:
        describe('S8: Messi + Kane (FWD-heavy)',
                 solve([messi_id, kane_id], [1338, 1711]),
                 'Lautaro → Messi, Watkins → Kane')

if __name__ == '__main__':
    main()
