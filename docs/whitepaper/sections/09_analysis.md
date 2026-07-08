# 09 - Critical analysis

Status: **DRAFT** (matures as live-results section fills in)

## 9.1 Where the model materially helped

**Held-out RMSE is honest signal.** The per-position best-backend table
in Section 7 generalised: the heuristic was the safest default for
defenders, the Poisson for goalkeepers, the GBM for attacking
positions. We never had a backend that all three rounds materially
disagreed with itself on; the rank ordering of player scores within a
position was consistent run-to-run.

**The Elo signal corrected real fixture mispricings.** When the static
FIFA ranking placed Norway behind Senegal but the rolling Elo (post-MD1)
had Norway ahead, the heuristic and Poisson backends adjusted Norway
players up by approximately one expected point. This kind of
realised-results-based correction is the canonical case for replacing
hand-maintained snapshots with derived live signals.

**The MILP optimiser kept us legal.** Budget, country cap, and transfer
quota constraints are non-trivial to satisfy by hand under the
attention available pre-deadline. The MILP solver never produced an
illegal squad and surfaced budget infeasibilities (e.g. "Lautaro to
Messi plus Doué to Bellingham costs $2.0M over budget") immediately
rather than after a manual mistake.

## 9.2 Where the model materially hurt

**GBM blindness to non-EPL stars.** Messi: heuristic 8.5, GBM 2.2. The
GBM has zero training rows for Messi and scores him on price alone,
adjusted down by the (rotation-conservative) status defaults. The user
who follows the GBM blind misses the tournament's top scorer.

We compensated with the heuristic backend, which does see Messi
correctly via its price-coef plus matchup signal, but a user trusting
the three-backend ensemble or the GBM alone would mis-allocate at the
premium tier.

**Rotation-risk modelling was too coarse.** Our rotation-risk multiplier
was a per-country scalar set by hand in `scripts/md3_plan.py` (e.g.
GER 0.55 in MD3 because Germany had clinched). This compressed three
distinct cases into one number: (a) which players actually play, (b)
how many minutes, (c) whether the substitute matters. A per-player
multiplier informed by manager press conferences would have been
materially better; we had no such feed.

**Cross-domain distribution shift hurt the GBM at the WC.** Section
7.4's discussion of the rejected v3 candidate generalises: any feature
whose range at training differs from its range at inference is a risk.
For attacking positions in particular, the GBM consistently
under-predicted premium international stars relative to their realised
totals.

## 9.3 Captain decisions

Captain is the single highest-leverage decision in fantasy football.
Across the six matchdays played, the captain decision broke down as:

| MD | Recommended captain | User captain | Realised raw | Captained pts | Decision quality |
|---|---|---|---:|---:|---|
| 1 | Lautaro (heuristic) | Lautaro | 1 | 2 | Model and user agreed; both wrong on outcome |
| 2 | Mbappé (model) / Olise (user swap) | Olise | 9 | 18 | User overrode and won |
| 3 | Bellingham / Messi (user) | Messi | 7 | 14 | User picked safe haul fixture, paid off |
| R32 | Messi (model) / Dembélé (user) | Dembélé | TBD | TBD | Differential play, pending result |

The pattern is clear: when the model and user disagreed on captain
because of differential or domain-knowledge reasons, the user's call
beat the model's in two of the three observed cases. We attribute this
to:

- The user's ownership-aware differential strategy is a function the
  model does not optimise for. The MILP picks max-EV; in a league at
  the bottom of standings, max-EV is not max-rank-improvement.
- The user has access to football-watcher information the model does
  not: who is in form, who looked tired at the weekend, who had a
  domestic-cup midweek.

## 9.4 The "random pick is doing better" claim

The user observed at MD3 that a random-pick player in their league was
outperforming our model-driven entry. This is a useful sanity check.
Two things to disentangle:

1. **Single-matchday variance is enormous.** A 42-point round (ours,
   MD3 after transfer hit) is in the long tail. The random picker may
   have caught a Mbappé hat-trick by chance.
2. **Hits are the silent killer.** Of the 60-point gap between MD2 (102)
   and MD3 (42), 6 points came from the transfer hit and the rest came
   from realised player rotation. If the user had made fewer
   transfers, the hit would have been smaller; if the model had a
   rotation-risk signal, the rotation cost would have been smaller.

A long-run comparison against random pick (Section 8 final table) is
the honest measure. Through R32 we expect the model-driven entry to
outperform a random-pick baseline by 15-25 points per round on
average, but to underperform a top-of-league human player who reads
team news.

## 9.5 The cost of an unforced transfer hit

Each transfer above the free quota costs 3 points. In group-stage
matches where the median XI scores around 50-60 points, a 3-point hit
is roughly 5% of the round. Hits compound: two transfer hits cost more
than 10% of expected matchday output.

The mathematical defence of a hit is that the gained player must
outperform the dropped player (and the captain swap implications must
net positive) by more than 3 points. In practice the user's hits in
MD3 each cost ~3 points and gained ~1 point in realised output. The
model's transfer-with-hit optimisation prefers to take the hit when the
expected horizon gain exceeds 3 over MD2 plus MD3; in retrospect, this
was systematically too aggressive because the model under-estimated
rotation.
