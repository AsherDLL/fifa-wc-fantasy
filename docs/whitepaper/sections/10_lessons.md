# 10 — Lessons learned

Status: **DRAFT**

## 10.1 Realised points must anchor model predictions for elite players

The single most consistent failure mode of the GBM backend is
under-prediction of premium international stars (Messi, Mbappé). The
model has never seen them; it falls back on price-and-strength signals
that compress them to roughly 5 expected points per match. Their
realised average through the group stage was 11-15 points per match.

The fix in retrospect: add a `realised_avg_per_match` bump to the
heuristic and Poisson backends (we did this in `scripts/md3_plan.py`'s
`final` blend), and exclude players with under N matches of GBM
training exposure from the GBM's premium-tier predictions entirely
(falling back to the structural prediction instead).

A cleaner architectural fix: a fourth backend that combines the
realised-form anchor with the structural prediction explicitly. This
fourth backend would dominate the GBM on premium tiers from MD2
onwards.

## 10.2 Rotation risk is a first-class feature, not a fudge factor

Hand-set per-country multipliers in a script (`ROT = {'GER': 0.55, ...}`)
are a code smell. They captured the right intuition but at the wrong
abstraction level: rotation is a per-player decision driven by manager
preference, recent minutes load, and tournament-context incentive. A
proper rotation-risk model would:

- Track per-player minutes by match across the tournament
- Score manager-specific rotation propensity (Scaloni, Tuchel,
  Deschamps, etc.) from historical data
- Detect the "clinched group" signal automatically from standings
- Surface a per-player Bernoulli probability of starting

We had none of this and paid for it in MD3.

## 10.3 Captain selection is where the human beats the model

Across the three group-stage matchdays the user's captain calls beat
the model's calls in two of three observed cases. Both wins came from
the user reading external signals (team news, expected manager
behaviour, player form) the model did not have. The one loss was the
initial Lautaro captain which the model and user agreed on.

The takeaway: model captain recommendations should be advisory, not
prescriptive. The MILP optimiser maximises within-squad point EV; it
does not optimise for ownership differential, narrative momentum
(top-scorer race), or fixture-specific motivation. A user with even
moderate football knowledge can correct the model on captain choices.

## 10.4 Differential strategy depends on standings position

Standard EV maximisation is correct when you are leading or tied with
your league. It is incorrect when you are at the bottom: you need
variance.

The simplest formalism: weight predicted points by
`(1 + lambda * (1 - ownership))` where lambda > 0 increases as your
position in the league worsens. We did not have this in the optimiser;
we approximated it by manually surfacing low-ownership options in the
candidate tables and recommending them when the user's standings
position warranted it.

A proper differential-aware MILP objective would be a useful Section 11
follow-up.

## 10.5 The cost of a transfer hit is consistently underestimated

In MD3 the user took multiple transfer hits and lost 6 points to them.
The expected gain from the rotated-in players was roughly +6 net of
hit, but realised rotation absorbed most of that. The expected-value
calculation that justified the hits was correct on its inputs but the
inputs were wrong (rotation risk was higher than modelled).

Rule of thumb adopted post-MD3: never take an unforced hit unless the
predicted horizon gain over the next two matchdays exceeds 6 points
(double the per-hit cost). This adds a margin of safety for the
rotation-risk under-estimation.

## 10.6 Auto-substitution mechanics matter

FIFA Fantasy's auto-sub rule replaces a 0-minute starter with the
highest-priority bench player who can fit the formation. The bench
order is set by the user in the app. This means the bench is not a
single backup; it is an ordered queue.

We never explicitly modelled the bench-order objective. The MILP
optimiser produces a starting XI and a bench but does not optimise the
bench order; the live decision tools have a sub advisor but it is a
single-shot recommendation, not a queue.

A bench-order optimiser would set the bench priority by predicted
points (highest first among outfield, GK always last). We produced this
manually as part of every transfer recommendation but it should be a
first-class output of `solve_lineup`.

## 10.7 The web report became central, faster than expected

We initially planned a FastAPI service; the user pushed back, asking
for a static HTML report. We pivoted. The static report quickly became
the highest-utility artefact of the project: every transfer decision
was made by opening the report, scrolling to the captain board, and
cross-checking with the user's domain priors.

The lesson generalises: produce decision-support artefacts that are
viewable without a server. Markdown for permanent records, HTML for
interactive review, JSON for downstream consumption.

## 10.8 The conversation log itself is a research artefact

Each decision came out of a chat exchange between the user and an AI
assistant. The model recommended; the user pushed back on specific
points; the model produced supporting data; the user made the call.
This back-and-forth produced better decisions than either party would
have made alone.

For the academic write-up, this raises a methodological question: how
do you report a decision that emerged from a multi-turn dialogue? Our
approach: the model and code produce the EV-maximising recommendation;
the user's domain priors produce the differential adjustments; the
final decision is documented per matchday with both inputs and
rationale. The whitepaper should disclose the AI-assisted methodology
honestly.
