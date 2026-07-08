# Phase 4 - Optimizer

Two MILPs solved with PuLP/CBC: pick the 15-player squad, then pick the
starting XI + formation + captain. Stage-aware constraints come from
`stage_config.py`, themselves codifying `docs/Fantasy.md`.

## Inputs

- `data/processed/predictions_<date>.parquet` - per-(player, round) point
  predictions from Phase 3a (or 3b when it lands).
- A target `Stage`. Default: `GROUP_MD1`, horizon `(1, 2, 3)` - the
  pre-tournament selection must last all three group-stage matchdays.

## Pipeline

1. **Scouting bonus**: `apply_scouting_bonus` adds +2 to any
   (player, round) row where `predicted_points > 4` AND
   `ownership_fraction < 0.05`. Thresholds reused from
   `fifa_fantasy.scoring` so the rule lives in one place.
2. **Aggregation**: `aggregate_to_player` sums `effective_points` across
   the horizon rounds, one row per player.
3. **Squad MILP**: `solve_squad` maximizes total horizon points subject to:
   - exactly 15 players, with 2 GK / 5 DEF / 5 MID / 3 FWD
   - total price ≤ stage budget ($100M group, $105M knockout)
   - ≤ `max_per_country` players from any country (3 → 8 by stage)
   - eliminated squads excluded entirely
4. **Lineup MILP**: `solve_lineup` picks 11 starters under one of the
   seven valid formations (4-4-2, 4-3-3, 4-5-1, 3-4-3, 3-5-2, 5-4-1,
   5-3-2), maximizing target-round `predicted_points`.
5. **Captain** = highest-predicted starter. **Vice-captain** = second.
6. **Bench order** = outfield bench sorted by `predicted_points` desc,
   then the spare GK (the game auto-subs the second GK only when the
   first didn't play).

## Stage config table

| Stage | Budget | Cap | Free transfers | Boosters available |
|---|---|---|---|---|
| GROUP_MD1 | $100M | 3 | unlimited | 12th Man, Maximum Captain (Wildcard not allowed for MD1) |
| GROUP_MD2 | $100M | 3 | 2 | Wildcard, 12th Man, Maximum Captain |
| GROUP_MD3 | $100M | 3 | 2 | Wildcard, 12th Man, Maximum Captain |
| R32 | $105M | 3 | unlimited | 12th Man, Max Captain, Qualification, Mystery (Wildcard not allowed for R32) |
| R16 | $105M | 4 | 4 | all five |
| QF | $105M | 5 | 4 | all five |
| SF | $105M | 6 | 5 | all five |
| FINAL | $105M | 8 | 6 | all five |

## Output

A JSON file under `data/processed/`:

```
recommendation_<STAGE>_<UTC-date>.json
```

with `squad_player_ids`, `lineup.formation`, `lineup.starter_ids`,
`lineup.bench_ids_priority_order`, `lineup.captain_id`,
`lineup.vice_captain_id`, `lineup.expected_points`, and the
budget/objective summary.

The CLI also prints a human-readable squad/lineup/captain summary.

## Day-1 sample output

```
Stage: GROUP_MD1   horizon: [1, 2, 3]
Budget: $100.0M / $100.0M  (remaining $0.0M)
Total horizon points: 263.87
Starting XI (3-4-3) - expected 58.20 pts
Captain:      Lautaro Martínez (ARG, E=6.91 → 13.83 doubled)
Vice-captain: Ferran Torres (ESP, E=6.49)
```

The optimizer skews toward mid-priced starters and skips the £10.5M
premiums (Mbappé, Kane, Haaland) because the Phase 3a heuristic is
near-linear in price - freeing budget buys better marginal points
elsewhere. Phase 3b's quantile regression should give the premium
forwards a non-linear ceiling boost and shift the optimum back toward
them. Track this as a known signal.

## Why MILP, not greedy

A greedy "highest points-per-$M" pick blows past the position counts and
the nationality cap, then repairs by swapping - and there's no
guarantee of optimality. A 1481-variable MILP is solved by CBC in well
under a second; the integer constraints are exactly what we need.

## Transfer planning (between rounds)

`solve_transfer` is a strict generalization of `solve_squad`: given the
previous-round squad and the stage's free-transfer quota, it maximizes

```
Σ total_effective_points · x[p]   −   3 · max(0, transfers_in − free_quota)
```

with all the same squad-composition / budget / nationality / eliminated
constraints. The −3 piecewise penalty is encoded with a non-negative slack
variable (`extra`) bounded by `new_picks ≤ free + extra`; since the
objective only loses points to `extra`, CBC pushes it down to exactly
`max(0, transfers − free)`.

Stages with unlimited free transfers (`free_transfers is None` → MD1 and
R32) short-circuit to a fresh `solve_squad` and report the resulting diff
against the previous squad with zero cost - same `TransferSolution`
shape, so the CLI and report don't branch.

### Usage

```bash
# Fresh MD1 selection
python -m fifa_fantasy.optimizer

# MD2 planning from the MD1 recommendation
python -m fifa_fantasy.optimizer --stage GROUP_MD2 \
    --from results/<host>_recommendation_GROUP_MD1_<date>.json

# With one rolled-over free transfer from MD1 (Fantasy.md rule)
python -m fifa_fantasy.optimizer --stage GROUP_MD2 \
    --from <previous>.json --rolled-over 1
```

The JSON payload gains a `transfer` block (transfers in/out, hit count,
net horizon points) and the markdown report grows an **OUT / IN / hits**
section so the recommendation is self-explanatory.

### Day-1 simulation (MD2 from the MD1 squad)

```
Stage: GROUP_MD2
Budget: $100.0M / $100.0M
Transfers: 2 made / 2 free → 0 hit(s) = −0 pts
Horizon points: gross 174.72, net 174.72
OUT: Rüdiger (GER, DEF), Nübel (GER, GK)
IN : Marquinhos (BRA, DEF), Ederson (BRA, GK)
```

Germany's MD2/MD3 schedule has weaker matchup signals than Brazil's for
those positions, so the planner spends the free quota to capture the
expected gain - no hit incurred.

## Still not in scope

- **Booster timing**: the Wildcard / 12th Man / Maximum Captain decisions.
  Heuristics from the sketch §7 are a fine v1.
- **Live captain switching / sub advisor** - Phase 5.
- **Multi-round captain optimization** - currently captain is chosen for
  the target round only, not optimized across the horizon.
