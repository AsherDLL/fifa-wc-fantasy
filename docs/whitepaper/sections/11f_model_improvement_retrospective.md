# 11f - Model improvement retrospective: what should have been built from the start

Status: **DRAFT**

This section is a critical retrospective on the predictive core. It
documents a defect that was present from the first matchday, was visible
in our own numbers the whole time, and was only corrected in the R16
window. It also documents a second failure that is arguably worse than
the first: the fix for the defect was designed and coded early, then
never actually deployed. The measurements here are reproducible via
`scripts/wc_forward_validation.py`, `src/fifa_fantasy/training/validate.py`
and `scripts/model_backtest.py`.

## 11f.1 The defect: every validated backend was form-blind

All three validated backends predicted a player's fantasy points from
two things only: how expensive the player is, and how strong their team
is relative to the opponent. None of them looked at how the player had
actually been performing in the tournament.

- The heuristic (`model/baseline.py`) computes
  `base = coef[position] * price`, then multiplies by a matchup factor
  and a home factor. Price and opponent strength. No recency term.
- The Poisson backend (`model/poisson.py`) distributes a structural
  team expected-goals estimate across positions with fixed shares. No
  player-level recency term.
- The GBM (`model/gbm.py`) trained on five features:
  `price_millions, is_home, strength_diff, squad_top_n_avg_price,
  opp_squad_top_n_avg_price`. Every one of them is a price or
  team-strength signal. The model had no column that told it whether a
  player had scored 15 points last round or 1.

The Monte Carlo backend (`model/monte_carlo.py`) was the only one that
used realised output, through a per-player form multiplier, and it was
explicitly marked experimental and never held-out validated.

The symptom was recorded in Section 09: for the R16 fixtures the
form-blind GBM predicted Lionel Messi at 1.69 and Kylian Mbappe at 2.18
while both were in double-digit form and priced at the ceiling. The
optimiser, fed those numbers, wanted to bench them and captain a
team-strength artefact instead. The human operators overrode the model
on captaincy and transfers repeatedly, and Section 08b shows no backend
picked the optimal captain in any group round; the user's captain
outperformed the Monte Carlo pick once (MD2). When
the humans have to routinely ignore the model on the single
highest-variance decision in the game, the model is not doing its job.

This was not a data-availability problem. The signal existed from day
one. The EPL training tables (`data/training/fpl_player_gameweek_*.parquet`)
have always carried `total_points` per gameweek. The WC player table has
always carried `round_points`, a list of realised points per completed
round. A trailing average of those numbers was computable at MD1. It was
simply never computed.

## 11f.2 The worse defect: the designed fix was never deployed

The project anticipated the EPL-to-WC distribution problem. `model/train.py`
has an `--include-wc` flag; `training/wc.py` mines realised WC round
points into training rows; the README states the plan in plain language:
"append our own labels to the training set and retrain on EPL + WC_so_far
... a 30-minute exercise on June 19."

That retrain was never run in production. The model artefacts in
`data/models/` were last written on June 21, from EPL data only. The
daemon (`docker/snapshot_loop.py`) refreshed data, ran features, ran
inference and re-solved the squad every twelve hours for the entire
group stage and R32, always against the same frozen June 21 EPL-only
model. Four rounds of realised World Cup labels accumulated in the raw
tables and were never fed back. The system had a learning loop drawn on
the whiteboard and an open switch on the wire.

A designed-but-undeployed fix is worth exactly zero points. This is the
central process lesson of the retrospective: building the retrain path
and building the schedule that runs it are two different tasks, and only
the second one scores.

## 11f.3 The fix, and why it is not the fix that failed before

The correction has two parts, and neither works without the other.

**Part 1: a leak-free lagged form feature.** `form_lag` is the trailing
mean of a player's own realised fantasy points over the previous three
matches, shifted by one so the current row's label never enters its own
feature. It is computed with one definition in all three places the
model reads data: EPL training (`training/features.add_lagged_form`,
grouped by season and player because the FPL element id is only unique
within a season), WC label extraction (`training/wc.py`, using
`round_points[:idx]`), and WC inference (`features/build._attach_form_lag`,
using the last three completed rounds). Missing values (a player's first
match) are left as NaN, which LightGBM handles natively.

**Part 2: retrain on EPL plus realised WC rounds.** With `--include-wc`
the GBM sees 3,639 WC rows alongside the 34,221 EPL rows, which
recalibrates the output scale to the tournament.

The important point for the paper is that Part 1 alone is close to
useless on the World Cup. With the form feature but EPL-only training,
the forward predictions barely moved (Mbappe went from 2.18 to 2.36 with
a `form_lag` of 10.7). The EPL-trained model had learned an output scale
that does not fit the WC and no single feature fixes that. Only Part 1
and Part 2 together produce sane forwards (Mbappe 7.89, Kane 8.03,
Haaland 8.84, ranked correctly by form).

This is also why the earlier heuristic v2 attempt (Section 05c) failed
and this did not. Heuristic v2 imposed a fixed 0.40 blend of realised
form onto the heuristic output and lost every backtest round
(cumulative 211 to 190). A fixed weight is noise at small sample sizes.
The GBM does not impose a weight; it learns per position from data how
much form is worth and can down-weight it where it is unreliable. Same
raw signal, opposite outcome, because one approach estimates the weight
and the other guesses it.

## 11f.4 Evidence

**EPL held-out RMSE** (`training/validate.py`, 2024-25 GW30-38, n as in
Section 07). Lower is better.

| Position | GBM v2 (form-blind) | GBM + form | Delta |
|---|---|---|---|
| GK | 2.650 | 2.566 | -0.084 |
| DEF | 3.187 | 3.147 | -0.040 |
| MID | 2.767 | 2.687 | -0.080 |
| FWD | 3.153 | 3.081 | -0.072 |

The form feature improves the GBM at every position, each delta at or
beyond the +/-0.05 determinism noise floor except DEF (which the
heuristic owns anyway).

**Leak-free WC walk-forward RMSE** (`scripts/wc_forward_validation.py`).
For each held-out WC round k the model trains only on EPL plus WC rounds
strictly before k, then predicts round k. Pooled across held-out rounds:

| Config | GK | DEF | MID | FWD | ALL |
|---|---|---|---|---|---|
| A: EPL only, form-blind (shipped v2) | 3.382 | 3.598 | 3.160 | 3.479 | 3.392 |
| B: EPL only, + form | 3.435 | 3.542 | 3.084 | 3.393 | 3.329 |
| C: EPL + WC, + form | 3.350 | 3.539 | 3.069 | 3.232 | 3.281 |

Config C wins at every position on genuinely out-of-sample WC data. The
forward RMSE, the position where the operators kept overriding the
model, drops 7.1 percent (3.479 to 3.232). Note that form alone (B)
slightly worsens goalkeepers (3.382 to 3.435); goalkeeper points are
dominated by clean sheets and saves, not personal scoring form, so a
personal-form feature is mostly noise there. Adding the WC labels (C)
recovers it. This is a clean, position-specific result, not a global
hand-wave.

**Production behaviour.** Config C is now installed in `data/models/`.
On the R16 features the top forwards by prediction are Haaland 8.84,
Kane 8.03, Mbappe 7.89, in place of the v2 ordering that buried all
three below team-strength midfielders.

The squad-level backtest (`scripts/model_backtest.py`) with Config C
shows the GBM and ensemble far ahead of the other backends, but that
harness now carries an in-sample caveat documented in its header: the
WC-trained model has seen the labels of the rounds it scores there, so
those two columns are upper bounds, not out-of-sample estimates. The
leak-free claim rests on the walk-forward RMSE above, not on that
harness.

## 11f.5 Second improvement: the per-position ensemble

Section 07 established from MD1 that no single backend wins every
position: Poisson wins goalkeepers, the heuristic wins defenders, the
GBM wins midfielders and forwards. That result sat unused. Every
runnable backend scored all four positions with its own formula,
including the positions it loses.

`model/ensemble.py` acts on the validation result. It runs the component
backends and, per position, keeps the prediction from the routed winner.
Routing defaults to the EPL held-out winners and can be re-derived from
any `validation_report.json` via `routing_from_report`. It is exposed as
`--backend ensemble`. This is not a new model; it is the refusal to keep
using each model where our own numbers say it is worst.

## 11f.6 Closing the learning loop

`docker/snapshot_loop.py` now retrains on EPL plus every completed WC
round at the start of each FIFA tick, before scoring
(`GBM_INCLUDE_WC=1`). The retrain is a few seconds on 37,860 rows. If it
fails the tick still scores with the previous models rather than
crashing. The model can no longer silently freeze while realised labels
pile up unused.

## 11f.7 What is still wrong

Honesty requires listing what this retrospective did not fix.

- **Minutes and rotation are still unmodelled.** The team-news pipeline
  (Section 11e) produces zero predicted-XI records per tick; the
  soccerdata proxy returns nothing for WC fixtures. Rotation risk, the
  thing Diego supplies by hand, is still entirely human. This is the
  single largest remaining gap because a benched starter scores near
  zero regardless of form or matchup.
- **Captaincy is still unmodelled.** Section 08b shows no backend picked
  the optimal captain in any round. Captaincy is a doubling multiplier
  on one player; it is where rounds are won and lost, and the system
  still outputs only a per-player mean with no explicit
  ceiling-versus-floor captain objective.
- **Goalkeeper form is noise.** The walk-forward table shows a personal
  form feature does not help goalkeepers. Their points come from team
  defence, not personal scoring; a team-level defensive-form feature
  would be the correct signal and is not built.
- **Distribution-shift handling is crude.** We concatenate EPL and WC
  rows and let the tree average them. There is no reweighting toward the
  scarcer, more relevant WC rows, no domain-adaptation term. As more WC
  rounds accumulate this matters less, but early in a future tournament
  it would matter a lot.
- **The squad-level backtest needs a leak-free mode.** It should retrain
  per round on WC-before-k to give an honest squad-points comparison,
  not only the RMSE comparison we trust today.

## 11f.8 The lesson in one line

The data to fix this was present at MD1, the code to fix it was written
by June, and the fix scored nothing until it was actually turned on in
July. Build the loop, validate the loop, and then, separately and
deliberately, run the loop.


## 11f.9 Addendum (2026-07-17): configs E and F, and the v4xg deployment

The walk-forward harness gained two configurations when the community
WC-2026 match dataset (real xG, real lineups; CC0) entered the pipeline:

| Config | Features added to C | Pooled RMSE (rounds 2-7) |
|---|---|---|
| C (deployed v3form) | - | 2.699 |
| E | real team xG for/against trailing form | **2.645** |
| F | real lagged start rates + minutes shares | 2.681 |

E improved every position and was deployed as **GBM v4xg** on 2026-07-17
under the pre-registered rule (pooled improvement >= 0.01, at most one
position regressing). F beat C only marginally, regressed FWD, and lost
to E in the same run - so even REAL minutes stay out of the point model
(cf. the config-D negative result above); they remain in the optimizer's
availability discount. The EPL holdout rerun under v4xg improves the GBM
at every position while leaving the ensemble routing unchanged
(`data/training/validation_report_v4xg.json`).
