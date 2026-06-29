# 08b — Cross-model retrospective backtest

Status: **DRAFT** (grows as more rounds complete)

This section answers the question: had we used each of our predictor
backends end-to-end through the tournament, with no human override,
how would each have scored? The answer establishes which backend
delivers the most value as an autonomous system, separately from
how the user's actual squad (which mixed model recommendations with
domain-expert overrides) performed.

## 8b.1 Methodology

For each completed round we:

1. Load the historical features snapshot closest in time to but before
   the round's deadline:
   - MD1: `features_2026-06-08.parquet` (pre-tournament)
   - MD2: `features_2026-06-18.parquet`
   - MD3: `features_2026-06-23.parquet`
   - R32: `features_2026-06-28.parquet` (used after round completes)
2. Run each backend on that snapshot.
3. Run the MILP optimiser to pick a 15-player squad and 11-player XI
   under the stage's budget and country-cap constraints.
4. Score the squad using realised `round_points` from the latest
   collector snapshot.
5. Also compute a random-baseline distribution by Monte Carlo over
   1000 valid random squads.

**Key caveat**: the backtest gives each model **unlimited transfers
each round**. The real user is constrained to 2 free transfers per
stage transition (with -3 per extra). This means the backtest is an
**upper bound on the model's standalone performance**, not a direct
apples-to-apples comparison against the user's actual squad. A
constrained-transfer version of the backtest is scoped in Section 8c.

The script is `scripts/model_backtest.py`. Reproducibility: run with a
fixed seed (default 42 in `_random_baseline`).

## 8b.2 Per-round results

### MD1

| Source | Starter pts | Captain raw | **Total** | Formation |
|---|---|---|---|---|
| heuristic | 44 | 2 | **46** | 3-4-3 |
| Monte Carlo | 63 | 14 | **77** | 3-4-3 |
| GBM | 38 | 6 | 44 | 3-5-2 |
| poisson | 29 | 2 | 31 | 3-4-3 |
| user actual | 31 | 1 | **33** (net 33) | 4-4-2 |
| random baseline | — | — | **mean 22, p90 37** | random |

Observations: Monte Carlo's strong MD1 came from sampling the goal-
scorer distribution well; its mean predicted captain pick produced 14
realised points. Heuristic was the second-best autonomous backend.
GBM matched heuristic on the starter pool but missed on captain. The
user's actual MD1 of 33 reflects the realised Lautaro captain blank.

### MD2

| Source | Starter pts | Captain raw | **Total** | Formation |
|---|---|---|---|---|
| heuristic | 68 | 13 | **81** | 3-4-3 |
| Monte Carlo | 75 | 4 | **79** | 3-5-2 |
| GBM | 60 | 12 | 72 | 3-5-2 |
| user actual | 75 | 9 | **84** (net 84) | 4-4-2 |
| poisson | 22 | 0 | 22 | 3-4-3 |
| random baseline | — | — | **mean 21, p90 36** | random |

Observations: Heuristic edged Monte Carlo because the heuristic's
captain pick (Olise) realised 13 raw points; Monte Carlo's captain
pick blanked at 4. The user's actual squad scored 84 with no
transfers used, beating every backend. The user-stated 102 in
project conversation was incorrect; the realised total from
`round_points[1]` is 84. The discrepancy may reflect a misremembered
auto-sub or app-display rounding; we use the realised number.

### MD3

| Source | Starter pts | Captain raw | **Total** | Formation |
|---|---|---|---|---|
| heuristic | 76 | 8 | **84** | 3-4-3 |
| Monte Carlo | 49 | 7 | 56 | 3-4-3 |
| GBM | 37 | 8 | 45 | 3-5-2 |
| user actual | 46 | 6 | **40** (net, after -6 hit) | 5-2-3 |
| poisson | 18 | 1 | 19 | 3-4-3 |
| random baseline | — | — | **mean 22, p90 38** | random |

Observations: MD3 was where the heuristic's group-stage edge
materialised most: 84 points, the round's high. The user took -6 in
transfer hits that returned roughly +6 of realised points (net 0),
showing the unforced-hit cost we documented in Section 10.5.

## 8b.3 Cumulative across MD1+MD2+MD3

| Source | Cumulative (Σ across 3 rounds) | Pts per round (avg) |
|---|---|---|
| **Monte Carlo** | **212** | 70.7 |
| **heuristic** | **211** | 70.3 |
| GBM | 161 | 53.7 |
| user actual (net, with hits) | 157 | 52.3 |
| poisson | 72 | 24.0 |
| random baseline (mean) | 63 | 21.0 |

The Monte Carlo and heuristic backends are statistically tied at the
top. The 1-point cumulative difference across three rounds is well
within the noise floor of any single backend's individual round score
(±5 pts of variance). For the paper we report them as **co-best**
backends pending more rounds of data.

The GBM trails by ~50 points across three rounds. This is consistent
with the held-out RMSE pattern (Section 7): GBM wins on MID and FWD
RMSE but loses on the captain selection because of its blindness to
non-EPL stars. The captain decision is single-game leverage; if your
model under-predicts Messi at $10M, your captain pick goes elsewhere
and your round score suffers disproportionately.

The Poisson backend trails by 140 points — a true mis-calibration
finding (Section 10.5b discussion). Even after the GK formula fix
(09c), the Poisson model systematically over-predicts mean points
(4.90 vs realised 2.50) and the resulting squad picks distort toward
high-volume / low-realised players. A full Poisson recalibration
pass is queued in Section 11.

The user's actual squad performed close to GBM. With unlimited
transfers (as the backtest gives the models) the user would have
been beaten by both heuristic and Monte Carlo. With realistic
constraints (only 2 free transfers per round, hit cost for extras)
the user's 157 is much closer to a fair model upper bound — see
Section 8c when written.

## 8b.4 Per-round captain pick comparison

The captain pick is the single highest-leverage decision in fantasy
football. Per-round optimal captain (highest realised raw points
within the chosen XI), vs each backend's captain pick:

| Round | Backend captain pick (realised raw) | User pick (realised raw) | Optimal pick in user's XI |
|---|---|---|---|
| MD1 | Monte Carlo: 14; heuristic: 2; GBM: 6 | Lautaro: 1 | Doué/Olise: 3-6 |
| MD2 | heuristic: 13; GBM: 12; Monte Carlo: 4 | Olise: 9 | Gakpo: 19 |
| MD3 | heuristic: 8; GBM: 8; Monte Carlo: 7 | Messi: 7 | Dembélé: 20 |

The user's MD3 captain pick was Messi (realised 7) when Dembélé in
the same XI realised 20. Both heuristic and Monte Carlo picked the
captain better in MD3, but neither reached Dembélé either; the
optimal pick was found only in retrospect.

This pattern is informative. The captain decision is the largest
single source of EV swing in the round, and *no backend* picked the
optimal captain in any round.

## 8b.5 Discussion: which backend "won"?

Strictly by cumulative realised points: Monte Carlo (212) and
heuristic (211) are co-winners.

By RMSE on held-out EPL: heuristic wins DEF; Poisson wins GK; GBM
wins MID and FWD. (Monte Carlo is not yet on the EPL held-out
leaderboard; that validation is queued.)

By interpretability: heuristic is most interpretable; Monte Carlo is
moderately interpretable; GBM is least interpretable.

By extensibility: Monte Carlo wins because it naturally produces a
distribution (p10/p50/p90) rather than a point estimate, which is
useful for differential captain selection and risk-aware lineup
choice in knockout rounds.

For the paper's recommendation to future practitioners: **Monte Carlo
and heuristic should both be deployed as primary backends; GBM as a
secondary check for non-EPL-attacking-tier picks; Poisson currently
needs recalibration before being used at all** (counter to its
held-out GK win, which we now suspect reflects the GK fix landing on
a small per-position sample).

## 8b.6 What we will track for the rest of the tournament

The backtest re-runs after each completed stage. The cumulative
leaderboard updates accordingly. Specifically:

- After R32 completes (~July 4): add round 4 to the table.
- After R16 (~July 8): add round 5.
- After QF (~July 12): add round 6.
- After SF (~July 15): add round 7.
- After Final (~July 18): add round 8 and produce the final
  cumulative leaderboard for the paper's headline finding.

The script is wired to read user-squad JSON files under
`data/user_squads/round_NN.json`; the user-actual column updates
automatically when those files are populated post-round.
