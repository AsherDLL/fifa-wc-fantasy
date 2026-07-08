# 11h - The July 8 full-system review: what was wrong, what was fixed, and what should have been done from the beginning

Status: **DRAFT**

Between the round of 16 and the quarter-finals the whole system was put
through a structured correctness review: every prediction backend, the
optimizer stack, the training and validation pipeline, and all project
documentation, followed by a fresh leak-free re-validation on the five
completed rounds and a batch of improvement experiments. This section
records what the review found, what shipped, what was rejected, and the
list this project earns the hard way: what should have been done
differently from the beginning.

The review found no defect in the model mathematics that survived
validation. The defects clustered in three places the earlier sections
had not looked: the decision layer between predictions and picks, the
data contract with the FIFA API, and the documentation itself.

## 11h.1 The decision layer was the weak layer

Sections 11f and 11g fixed the predictive core. The July 8 review found
that the layer that turns predictions into picks had five distinct bugs,
three of them visible in the live quarter-final recommendation of that
morning.

1. **The availability discount ignored the players it existed for.**
   `availability_factor` mapped a missing participation history (NaN) to
   factor 1.0, "no history, no claim". That policy is correct before the
   tournament, when nobody has history. Five rounds in, NaN means the
   player has never taken the pitch, the strongest bench signal there is.
   Third-choice goalkeepers sailed through undiscounted while a
   two-of-three-rounds starter was discounted to 0.83. The policy is now
   context-aware: if anyone in the frame has history, NaN means never
   played and gets the floor.

2. **The XI solver optimized a different objective than the squad
   solver.** `solve_squad` maximized effective points (bonus and discount
   applied); `solve_lineup` maximized raw predicted points, because the
   column subset built between the two silently dropped
   `effective_points`. Consequence: the discount could keep a rotation
   risk out of the squad but could not keep him out of the starting XI
   once a cheap slot bought him in. The live output started a 0.9
   percent-owned backup keeper over a 13 percent-owned ever-present
   number one. Both solvers now optimize the same column.

3. **The Poisson backend double-counted the scouting bonus.** It baked
   the +2 into `predicted_points`; the optimizer pipeline, whose
   docstring says the bonus is encoded in exactly one place, added it
   again. Every low-ownership backup keeper got +4 over the team-level
   clean-sheet base, which is why the backup outscored the starter on
   paper (8.72 versus 4.72 for the same team). The backend now emits raw
   expectations and the bonus is applied once, in the pipeline.

4. **NaN poisoned the captain sort under the ensemble.** The ensemble
   carries quantile columns for every row but only fills them for
   gbm-routed positions; GK and DEF rows hold NaN. The captain selector's
   `getattr(row, "predicted_p90", mean)` fallback fires only when the
   attribute is absent, not when it is NaN, so the ceiling term went NaN
   and Python's sort with NaN keys returns arbitrary order. In one
   regenerated recommendation this crowned a 4.7-mean goalkeeper captain
   over an 11-mean forward. All numeric fallbacks are now NaN-aware, and
   a regression test asserts every composite score is finite.

5. **The Monte Carlo form multiplier was dimensionally wrong.** Its
   numerator is a per-match average; its denominator was the positional
   mean of tournament totals, which grows with every round played. By
   round five the ratio pinned every player to the 0.3 clip floor,
   erasing exactly the differentiation the multiplier exists to provide.
   The denominator is now a per-match average too.

None of these are modeling errors. They are plumbing errors, and unit
tests did not catch them because every component behaved as its own test
specified. What was missing were end-to-end invariants on the final
recommendation: a never-played player must not start, a captain score
must be finite, a bonus must be counted once. Those invariants are now
tests.

## 11h.2 The data contract was misread from day one

The most consequential finding: the FIFA API truncates each player's
`round_points` list at the last round the player personally recorded
anything. Interior DNPs are zero-filled; trailing DNPs simply vanish.
Nothing in the API documents this.

Every consumer of `round_points` assumed the list length equals the
squad's completed rounds. It does not, for 269 of the alive players at
the quarter-final snapshot, roughly 30 percent of everyone with history.
The failure case is precise and cruel: a player benched for the last two
rounds has those benchings deleted, so his trailing form is computed
over his last three matches PLAYED rather than his team's last three
rounds. Kevin De Bruyne entered the quarter-final data as form 5.33 and
participation 1.0; the truth was form 4.33 and participation 0.67. The
inflation lands on recently-dropped players, exactly the population the
availability discount exists to catch, and it defeated the discount for
all of them.

The fix reconstructs the truth from a second source the API cannot
truncate: the fixture table. A squad's completed-fixture count is the
number of rounds its players were available to appear in, so padding
`round_points` with zeros up to that count restores the trailing DNPs.
The padding is one shared function used by training extraction and
inference, so both sides see one feature definition
(`features/build.pad_round_points`).

Re-validated on the corrected population (which is also larger, because
never-scored squad players now correctly appear as zero rows, matching
what the deployed model actually predicts over), the champion
configuration holds and the margin widens:

| config | GK | DEF | MID | FWD | ALL |
|---|---|---|---|---|---|
| A epl_noform | 3.525 | 3.133 | 3.063 | 3.181 | 3.168 |
| B epl_form | 3.360 | 2.975 | 2.779 | 2.986 | 2.959 |
| C eplwc_form (shipped) | 2.463 | 2.901 | 2.646 | 2.774 | 2.736 |
| D eplwc_all | 2.458 | 2.891 | 2.667 | 2.818 | 2.747 |

(Leak-free walk-forward, rounds 2-5 held out, regenerated by
`scripts/wc_forward_validation.py`. These numbers are not comparable to
the pre-padding table in 11f because the evaluation population changed;
within either population the ordering is the same.)

## 11h.3 Improvement experiments: three more honest negatives

The same walk-forward harness evaluated three candidate improvements.
All three lost. They are recorded so nobody re-litigates them.

- **Form x opponent-strength interaction feature.** Pooled 3.243 versus
  3.250 for the champion on the pre-padding population, a 0.2 percent
  edge that loses on two of four held-out rounds. Noise. Not shipped.
- **Recency-weighting WC training rows** (3x and 5x sample weights).
  Pooled worse (3.255 and 3.269 versus 3.250), monotonically worse on
  goalkeepers. Not shipped.
- **Dropping DNP rows from WC training** to match the EPL pipeline's
  minutes filter. Decisively worse: pooled 3.033 versus 2.736 on the
  padded population, with goalkeeper RMSE exploding from 2.46 to 3.73.
  The zero-target rows are not label noise; they are how the model
  learns that a low-participation profile scores zero. What looked like
  an inconsistency between the EPL and WC pipelines is load-bearing.
  Not shipped, and the asymmetry is now documented as intentional.

The discipline from section 5c stands: four candidates entered
validation this round (these three plus the padding fix); one shipped.

## 11h.4 Operations: the daemon that ran June code through July

The review also closed an operational hole documented here because it
cost real points. The snapshot daemon is a long-lived process with the
source tree bind-mounted. Code changes on disk reach the subprocesses it
forks each tick, but never the loop itself. The loop that started on
June 29 was still running its June 29 self on July 7: the stage variable
still said R32 two rounds later, and the retrain-each-tick step added in
July existed on disk but had never executed. The models the daemon
scored with all tournament were frozen EPL-only weights until a manual
retrain.

A related latency: the collector runs every 12 hours, so bracket state
lags match results by up to half a day. On July 7 the system briefly
held that two advancing teams were eliminated because the snapshot
predated their round-of-16 wins. Any advice generated in that window
inherits the stale bracket silently.

Both are one design decision away from impossible: a long-lived process
must log its own code version at startup and be restarted on deploy, and
the collector cadence must densify on match days. The daemon now logs
its schedule at startup and the stage is pinned per bracket round in
compose; match-day cadence remains an open item.

## 11h.5 Documentation drift: 38 findings

A dedicated audit compared every claim in this whitepaper and the README
against the code. It found 38 discrepancies: the methodology section
still specified the five-feature GBM two versions after form_lag made it
six; the README described three backends when the CLI offers six and the
daemon's default is the ensemble; two sections disagreed about the same
round's realised score; a section referenced a companion section that
was never written; the validation chapter cited row counts three other
places contradicted; the package metadata carried a placeholder author.

Every number that was wrong had been typed by hand. Every number that
was right was generated by a script and pasted with its source cited.
The lesson is mechanical, not moral: documentation numbers must come
from script output, and a drift audit belongs before publication, not
after. The decision-layer experiments of 11g (availability discount,
captain rules, GK blend) remain session analyses without committed
regeneration scripts; converting them is an open item and the only
remaining violation of this paper's own reproducibility rule.

## 11h.6 What should have been done from the beginning

The list this project would hand to its day-one self:

1. **Verify the data contract against a second source before building
   on it.** One assertion comparing `len(round_points)` to the squad's
   completed fixtures would have caught the truncation in June. The
   cross-check existed in the data the whole time; nobody wrote it.
2. **One score column end to end.** The moment two solvers optimize
   different columns, every layer between them is a placement for a
   silent regression. The squad/XI objective split survived four rounds
   of live operation without a test noticing.
3. **End-to-end invariant tests on the artifact users consume.** Unit
   tests proved each component to its own spec while the composed
   pipeline started a third-choice goalkeeper. Invariants on the final
   recommendation (no never-played starters, finite captain scores,
   single bonus application, budget within cap) are cheap and would have
   caught three of the five decision-layer bugs at introduction.
4. **NaN policy is a design decision per column, not a default.** Three
   independent bugs in one review trace to NaN having no declared
   meaning: unknown-versus-never-played, missing-versus-not-applicable.
   Every nullable column should declare what NaN means and every
   consumer should be tested against it.
5. **A long-lived process is a deployment target.** Version-log at
   startup, restart on deploy, and alert when configuration (the stage)
   is older than the world it describes.
6. **Build the walk-forward harness first, not in week four.** Every
   modeling decision made before `wc_forward_validation.py` existed
   (the GK save-bonus constant, heuristic v2, the original captain
   rule) was evaluated on intuition or in-sample evidence, and two of
   the three were later reversed. The harness is around 150 lines. It
   is the cheapest component in the repository and the only one that
   has never been wrong.
7. **Generate documentation numbers.** Hand-copied figures drifted in
   38 places in five weeks. The whitepaper's own rule (every table
   cites a regenerating script) was right; it was the enforcement that
   was missing.

## 11h.7 State of the system after the review

| Layer | State |
|---|---|
| GBM | v3form, six features, retrained each tick on EPL + completed WC rounds with corrected (padded) extraction |
| Ensemble | poisson GK, heuristic DEF, gbm MID/FWD; production default |
| Poisson | scouting bonus removed from backend (applied once in pipeline); team-news gate added |
| Monte Carlo | form multiplier denominator corrected to per-match scale |
| Availability discount | context-aware NaN policy; applied in squad AND lineup objectives |
| Captain | ceiling-aware selector, NaN-proof; chase default 0.9 |
| Recommendation JSON | strictly valid (no NaN tokens); per-player effective_points included |
| Tests | 289 passing, including new invariants for every bug above |
| Live output | quarter-final recommendation starts the actual number-one goalkeeper; captain Dembele on merit (highest ceiling, differential edge) |

Two review passes, sections 11f through 11h, tell one story in three
acts: first the model could not see form, then the decisions ignored
what the model saw, and underneath both the data was quietly wrong in
the one place nobody checked. Each act was found by measurement, fixed
in one place, and pinned by a test. That is the method; the ranking
will say whether it was enough.
