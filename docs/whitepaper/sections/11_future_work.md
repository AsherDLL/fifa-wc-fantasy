# 11 - Better approaches and future work

Status: **DRAFT**

## 11.1 Player-level form weighting

The current blend in `scripts/md3_plan.py`:

> final = 0.55 * ensemble_predicted + 0.30 * avg_pts_per_match + 0.15 * form

is a hand-tuned anchor. A cleaner alternative: a Bayesian update where
the prior is the model prediction and the likelihood is the realised
match-by-match points. After N matches:

> posterior_mean = (model_prediction / sigma_model^2 + sum(realised) / sigma_data^2) / (1/sigma_model^2 + N/sigma_data^2)

with `sigma_data` shrinking as N increases. The fixed-blend
approximation works for the WC's short tournament; a longer-horizon
fantasy game (full FPL season) would benefit from the proper Bayesian
treatment.

## 11.2 Team news ingestion

The single biggest data gap we had was confirmed XIs published by
managers in pre-match press conferences. Sources to consider:

- Manager press conferences via the official tournament site
- Predicted XIs from major outlets (BBC, ESPN, beIN, Goal.com)
- Official team-sheet drops one hour before kickoff
- Social media accounts of national-team beat reporters

A scraper that monitors these sources and produces a per-player
"likelihood of starting MD<N>" before the deadline would close the gap
that hurt us most in the group stage clinched-group rotation.

## 11.3 Bayesian update per match

Beyond form weighting, a proper Bayesian model would update:

- Per-player skill posterior (intrinsic ability)
- Per-team strength posterior (offensive and defensive parameters)
- Per-manager rotation propensity posterior

This is essentially a fully-pooled hierarchical model with partial
pooling at the country and player levels. PyMC or NumPyro implementation
would fit in a Section 11 follow-up paper. The model would explicitly
quantify uncertainty in addition to point estimates, which the MILP
optimiser could consume as a chance-constrained variant.

## 11.4 Tournament-specific GBM

Training on EPL data was a pragmatic choice given pre-tournament data
availability. With all WC 2026 matches recorded, plus prior WC 2022,
WC 2018, Euros 2024, 2020, 2016, and Copa America tournaments, a
WC-and-Euros-specific GBM becomes feasible.

The challenge: the per-tournament sample sizes are small (64 matches
per WC; ~50 per Euros), so the model would need careful regularisation
to avoid the v1-style overfitting documented in Section 7.3.

## 11.5 Captain Monte Carlo with explicit league-ownership distribution

Our captain decisions blend EV (model-driven) with differential (user
domain). The honest formalism is:

> rank_improvement(captain) = P(captain delivers) * (1 - ownership) * captained_points - P(field captain delivers) * field_ownership * field_captained_points

A captain Monte Carlo that explicitly samples both your captain and the
field's likely captain across many simulated matchdays would produce a
ranked list of differential-adjusted captain picks. The
`scripts/md3_dembele_locked.py` Monte Carlo is a primitive version of
this; a proper one would model the field's captain distribution as a
function of ownership, not a single point estimate.

## 11.6 Stage-aware horizon weighting

The group stage has a multi-round horizon (squads must last MD1 plus
MD2 plus MD3 with limited transfers); the knockout stage is a single
game per round. Our MILP optimiser uses a `DEFAULT_ROUND_HORIZON` dict
to model this, but a smarter formulation would explicitly weight
horizon-future points by the probability of the player still being in
the squad after the round (which depends on whether the optimiser would
transfer them out).

This is a stochastic-programming generalisation of the current
deterministic MILP. The complexity goes up materially; the gain is
unclear.

## 11.7 Auto-sub mechanic modelling

`solve_lineup` produces a bench in order of predicted points within
position. The actual FIFA auto-sub rule swaps based on minimum starter
adjustment: an outfield bench player replaces a 0-minute outfield
starter only if the resulting formation is still valid.

A richer formulation would optimise the bench order itself, taking into
account each bench player's correlation with each starter (e.g. don't
put two GERs in the bench queue if both starters are also from
clinched-group teams; spread the rotation risk across countries).

We treat this as a Section 11 nice-to-have. In practice the
predicted-points ordering produces good-enough bench priorities.

## 11.8 Open source the code, data pipeline, and decision log

The repository is already public-ready in code terms. Releasing the
post-tournament decision log along with realised outcomes would let
others reproduce our results, identify failure modes we missed, and
benchmark their own predictors against our matchday calls.

This is the main planned follow-up after the WC final: clean up the
repo, publish the validation report and the live decision log, and
submit to a sports-analytics venue or workshop.

## 11.8b Fix the goalkeeper save-bonus bug (confirmed)

The flat `GK_SAVE_BONUS = 1.0` constant in `poisson.py` is empirically
wrong; the empirical formulation scales save bonus by opponent xG.

**Target implementation (validated before shipping):**

```python
# poisson.py
SHOT_PER_XG_RATIO = 4.0     # ~4 shots on target per unit of xG, empirical
SAVE_PCT          = 0.85    # ~85% save percentage for an average WC goalkeeper
SAVES_PER_BONUS   = 3       # FIFA Fantasy: +1 per 3 saves

def gk_save_bonus(opp_xg: float) -> float:
    expected_shots_on_target = opp_xg * SHOT_PER_XG_RATIO
    expected_saves           = expected_shots_on_target * SAVE_PCT
    return expected_saves / SAVES_PER_BONUS
    # equivalent to opp_xg * (4 * 0.85 / 3) = opp_xg * 1.13
```

For Dibu vs CPV (opp_xg ~ 0.4): GK save bonus ≈ 0.45 (was 1.0)
For Rangel vs ECU (opp_xg ~ 0.7): GK save bonus ≈ 0.79 (was 1.0)

**Validation gate.** Before shipping the fix, run
`python -m fifa_fantasy.training.validate_main` on EPL 2024-25 GW
30-38 held-out and confirm Poisson GK RMSE improves. The current
RMSE for Poisson at GK is 2.503 (already the best of the three
backends). If the fix improves it further, ship. If it regresses,
the empirical ratios above need re-tuning against EPL data
specifically.

**Marked for implementation before R16 deadline.**

## 11.9 What we would do differently from MD1

If we ran the system over again with the lessons learned:

1. Set the rotation-risk multiplier per-player based on form, minutes,
   and team standings dynamics, not a per-country scalar.
2. Add a `realised_form_anchor` term to all three backends from MD2
   onwards, weighting model EV down as realised data accumulates.
3. Use the Poisson backend for goalkeepers (per Section 7's RMSE
   winner) instead of the heuristic default.
4. Captain on the lowest-owned of the top three EV candidates, not on
   the single highest EV.
5. Never take a transfer hit unless the horizon gain exceeds 6 points
   (double the per-hit cost).
6. Scrape predicted XIs from at least two major outlets the night before
   every matchday.

Each of these is a single section's worth of follow-up work. The
combination would, on our retrospective analysis, have improved the
matchday score by roughly 8-15 points on average across the six
matchdays played.
