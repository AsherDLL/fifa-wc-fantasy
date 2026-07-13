# 05d - Joint XI-aware optimization and the semifinal decision study

Status: **DRAFT**

This section documents a defect in the transfer optimizer's objective
function, the corrected formulation that replaced it for semifinal
decision-making, and a pre-registered head-to-head comparison between
the model's squad and the manager's squad that the correction enabled.
The correction shipped as an analysis script
(`scripts/sf_joint_analysis.py`); the production solvers were left
untouched mid-tournament by design (see 5d.7).

## 5d.1 The defect, and how a user constraint exposed it

`solve_transfer` (src/fifa_fantasy/optimizer/solvers.py) selects the
best 15-player squad by maximizing the sum of expected points over all
15 players, with the starting XI chosen afterwards by a separate
lineup solve. Only eleven players plus a doubled captain score in this
game, so the objective rewards spending budget on bench quality that
never converts to points.

The defect surfaced empirically during semifinal planning. A solve
constrained to keep two specific midfielders produced a starting XI
worth 67.6 expected points, while the unconstrained solve produced
67.3. A constrained optimum can never exceed an unconstrained optimum
of the same objective; the constrained run beat it on the metric that
matters (XI plus captain) because the unconstrained run was maximizing
a different one (the 15-player sum). The user's domain instinct did
not outperform optimization; it exposed a wrong objective, which is a
distinction worth recording.

## 5d.2 The joint formulation

One MILP now selects squad, XI, formation, captain, and transfers
simultaneously:

    max   sum_p eff_p (y_p + c_p)  +  0.1 sum_p eff_p (x_p - y_p)
          - 3 extra  (+ booster payoff, 5d.5)

    s.t.  x in {0,1}: squad membership, |x| = 15, positions 2/5/5/3,
          price(x) <= budget, per-country cap;
          y <= x, |y| = 11, exactly one legal formation
          (7 valid formations, same table as the lineup solver);
          c <= y, |c| = 1 (captain doubles);
          extra >= |new picks| - free transfers, extra >= 0.

The 0.1 bench weight is a small tiebreak for autosub insurance, small
enough never to trade an XI point for bench quality. Eliminated
players still owned by the manager enter the pool as retainable
zero-point rows: keeping a dead bench player is legal and free, and
with few transfers left it is often optimal, which a pool restricted
to alive players cannot express.

## 5d.3 Scouting bonus as an expectation, not a constant

The production pipeline adds a deterministic +2 to any player under 5
percent ownership (documented as coarse in pipeline.py). The joint
model instead adds `2 * P(points > 4)` evaluated on the player's
predicted quantile distribution, with missing quantiles backfilled
from the per-position walk-forward RMSE. The deterministic version
materially overrated low-ownership stars: it was the difference
between the solver captaining a 4.7-percent-owned winger and the
Monte Carlo layer (5d.5) preferring the high-ownership striker at
every realistic chase margin.

## 5d.4 Advance probabilities

Booster payoffs need per-team advance probabilities. Hand-setting them
would put an untracked judgment inside a measured pipeline, so they
are derived: the Poisson backend's team xG mapping gives a match win
probability (extra time settled proportionally to strength), blended
50/50 with prediction-market implied shares from the newest Polymarket
snapshot. For the semifinals this gave France 0.596 over Spain and
England 0.498 against Argentina.

## 5d.5 Monte Carlo layer and the captain reversal

Expected value cannot rank chase strategies, so 20,000 correlated
scenarios (Gaussian copula, shared team factor, rho 0.3) price every
plan against a template-manager XI built from ownership. Player scores
come from piecewise-linear inverse CDFs through the q10/q50/q90 heads;
match outcomes reuse the same team factors, so qualification payoffs
correlate with player scores. The layer reports P(beat the field by at
least M) for a range of margins.

Its headline finding reversed a deterministic conclusion: the
mean-argmax captain (the low-ownership differential) lost to the
template captain at every margin up to +20 per round once the scouting
bonus was priced at its true firing rate. Differential captaincy only
wins when a single round must produce a +30 swing. This refines the
ceiling-captain rule of section 11g with an explicit field model.

## 5d.6 Booster analysis as a two-round assignment problem

Boosters are one-use across the remaining rounds, so the question is
never "which booster is best this round" but "which assignment of
boosters to rounds maximizes the total." Single-round gains on the
manager's locked XI were qualification +11.2, clean-sheet shield +8.5,
twelfth man +6.4. The qualification booster, however, retains
approximately +11 of value in the final (8 plus 3 split under an
8-per-country cap, payoff 6 + 10p for favorite probability p), while
the shield retains roughly +7. Pairing totals therefore invert the
greedy choice: shield now plus qualification in the final scores 19.5
against 18.2 for the reverse. The manager's XI made the qualification
payoff distribution degenerate and auditable: exactly +14 if France
advanced, +8 otherwise.

## 5d.7 A pre-registered pick comparison

Both squads for the semifinal round, the manager's and the model's,
were committed to `data/evaluation/sf_pick_comparison.json` before the
round locked, with captain, bench order, booster, the exact generator
command (pinned inputs and seed), and the scoring rules to be applied
afterwards. Under identical simulation draws the model pick led 68.8
to 68.1 with the shield active, a gap well inside one player's match
variance. Registering the comparison before kickoff removes hindsight
freedom in how the result gets reported; section 08 will carry the
realized outcome.

## 5d.8 Scope and production status

The joint solver is deliberately an analysis script, not a production
change: swapping the optimizer objective days before a semifinal would
trade a known small bias for unknown regression risk, the same
reasoning that kept heuristic v2 out of production (05c). Folding the
XI-aware objective into `solve_transfer` and `solve_squad`, with the
probabilistic scouting expectation behind a flag, is queued as
post-tournament work (11_future_work). Cross-references: 11c for the
Monte Carlo simulator lineage, 11g for captaincy and variance, 08 for
realized results.
