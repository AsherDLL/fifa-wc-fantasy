# 05c - Heuristic v2 attempt and failed validation

Status: **DRAFT**

This section documents an attempt to improve the heuristic backend by
incorporating expert practitioner knowledge and additional structural
components. The v2 attempt **failed empirical validation**, scoring
21 points lower (190 vs 211) than v1 on the WC 2026 cumulative
backtest. We ship v1 as the production primary and keep v2 in the
repository as a documented failed experiment.

## 5c.1 What v2 added

Six changes layered on top of v1's `c_pos × price × matchup × home`
formula:

1. **GK save bonus scaled by opp_xg.** From the Section 09c
   findings, multiplied opponent xG by 0.50 to add expected save points
   for goalkeepers. Replaced v1's contribution of 0 (heuristic GK
   prediction was identical regardless of opponent strength).

2. **DEF clean-sheet probability bonus.** Added `0.3 × exp(-opp_xg)
   × 5` (the clean-sheet reward) to defenders. Encodes the FIFA
   Fantasy fact that defenders earn +5 for a clean sheet at 60+ min.

3. **MID ball-progression bonus.** Added a bonus proportional to a
   midfielder's realised over-performance relative to the position
   median. Captures the FIFA WC 2026 scoring change that rewards
   tackles, chances created, and shots on target.

4. **FWD goal-conversion bonus.** Same as MID but for forwards;
   captures forwards over-performing their position median by
   converting at higher rate.

5. **Premium tier bump.** Added `0.08 × max(0, price - 9.5)` for
   players above $9.5M. Captures the heavy-tailed scoring of elite
   players (Mbappé, Messi, Haaland) that the v1 linear-in-price
   base term misses.

6. **Realised-form anchor.** Blended the model prediction with the
   player's realised average points per match. Weight grows from 0 at
   N=0 games to 0.40 at N=4+ games. Addresses v1's blindness to
   non-EPL stars (Messi: model 5, realised 13).

## 5c.2 The retrospective backtest result

Same protocol as Section 8b: each round, run v2 against the historic
features snapshot, solve the MILP, score against realised round_points.

| Round | v1 squad score | v2 squad score | Δ |
|---|---|---|---|
| MD1 | 46 | 39 | -7 |
| MD2 | 81 | 79 | -2 |
| MD3 | 84 | 72 | -12 |
| **Cumulative** | **211** | **190** | **-21** |

v2 lost on every single round.

## 5c.3 Why v2 failed (diagnosis)

Four candidate reasons:

1. **Realised-form anchor pulls toward noisy data.** With only 1-3
   matches of WC data, the per-player average is dominated by
   variance, not signal. The v1 formula's structural prediction is
   actually more reliable than the realised average when N is small.

2. **Stacked bonuses shift squad composition.** v2's per-position
   bonuses raised certain players' rankings (e.g. DEF with low-xG
   opponents) without those players actually outperforming the v1
   picks in realised data. v2 picks 5+ different players per round
   than v1; the substitutes underperformed.

3. **Premium tier bump is too small.** $0.08/M above $9.5M is a
   trivial nudge that doesn't materially change the optimizer's
   selection. We should have either dialled it up or omitted it.

4. **Combinatorial sensitivity.** With six small additions, the
   model has six places to be slightly wrong. The v1 formula has
   three (position coef, matchup, home). Fewer moving parts means
   fewer sources of misalignment.

## 5c.4 Lessons for the paper

Two reinforcing lessons:

**Lesson 1: theoretical-first feature engineering needs empirical
validation.** This is the same pattern as the GK save bonus
(theoretical 1.13 multiplier vs empirical 0.50 in Section 09c). Each
v2 addition was individually defensible; collectively they degraded
performance. The validation gate is non-negotiable.

**Lesson 2: simpler models can outperform richer models on small
samples.** This is a well-known finding in statistical learning
(bias-variance tradeoff). The richer v2 has more capacity to fit
the training assumptions, but the assumptions themselves are
imperfectly aligned with reality. The simpler v1, even when "wrong"
on individual components, is closer in aggregate.

## 5c.5 Production decision

v2 is **not shipped** as the primary backend.

- `python -m fifa_fantasy.model --backend heuristic` -> v1 (production)
- `python -m fifa_fantasy.model --backend heuristic_v2` -> v2 (research)

Both are kept in the repo for reproducibility. The whitepaper backs
v1+Monte Carlo as the co-best per Section 8b. Future work
(Section 11) lists the v2 component improvements that could survive
validation independently: e.g. just the GK save bonus + form anchor
together, or just the premium tier bump alone, validated on
held-out EPL data before being added to v1 incrementally.

## 5c.6 Decision rule for adding components in the future

After two failed theoretical-first feature additions (GK v2
multiplier, Heuristic v2 stack), we adopt a stricter discipline for
any future model changes:

1. **One component at a time.** Add a single addition, validate, ship
   or revert. No more multi-component shipping.
2. **Held-out EPL RMSE must improve at the position-of-interest** before
   shipping.
3. **Live WC backtest must not regress** for at least one round of
   tournament data.
4. **Document the failed attempts** as we do here. Failed empirics
   are publishable contributions.
