# 11b - Innovation roadmap: prediction markets and beyond

Status: **DRAFT**

The Section 11 future-work proposals cover incremental improvements to
the existing three-backend architecture. This section lays out the more
ambitious research direction we believe is publishable in a top-tier
sports analytics venue: combining live prediction-market data with our
ELO and structural backends, in the spirit of Benter (1994).

## 11b.1 The Benter combining approach applied to fantasy football

Benter (1994) showed that the public's implied probability (from the
parimutuel pool) and a private model's predicted probability are not
substitutes but complements. Each carries signal the other does not.
The optimal combiner is logit-based:

```
logit(p_combined) = β_0 + β_1 logit(p_model) + β_2 logit(p_public)
```

with coefficients estimated by maximum likelihood on historical
out-of-sample data.

For our fantasy football problem, the equivalent is:

```
predicted_points_combined = β_0 + β_1 predicted_points_model
                                + β_2 implied_points_market
```

where `implied_points_market` is derived from prediction-market
contract prices on:

- Match outcome (win/draw/loss probabilities)
- Total goals (Poisson rate priors)
- First-goalscorer markets (per-player implied probability of scoring)
- Top-scorer-of-tournament markets (priors on premium players' scoring rates)

The combiner's β coefficients are estimated on historical World Cup
2022 data (Polymarket has market history; we can scrape it). The
β-validated combined predictor should beat both the model alone and
the market alone on out-of-sample WC 2026 data.

## 11b.2 Specific Polymarket and Kalshi contracts to ingest

Polymarket (CFTC-regulated US app since 2025):

- `Will [Country] win WC 2026?` (one per country, ~32 alive contracts)
- `Will [Country] reach the [stage]?` (multi-stage contracts)
- `Top scorer at WC 2026 will be [Player]?` (~5-10 elite candidates)
- `Will there be more than X goals in [Match]?` (per fixture, ~3-6 contracts each)
- `Will [Match] go to penalties?` (knockout fixtures)

Kalshi (CFTC-regulated):

- Match-outcome contracts at sharper liquidity than Polymarket for major
  fixtures
- Combo contracts (parlays) launched late 2025; useful for player-prop
  combinations
- US-specific player props for selected fixtures

Both platforms expose REST APIs (Polymarket: `clob.polymarket.com`; Kalshi:
`trading-api.kalshi.com`). Rate limits and authentication requirements
are tractable.

## 11b.3 Implied probabilities from contract prices

A binary contract priced at $0.62 implies a market probability of 0.62
(adjusting for the bid-ask spread and platform fee). Aggregating across
related contracts (e.g. all home-win contracts for a fixture) lets us
extract a multinomial outcome distribution.

For our Poisson backend, the market-implied team xG can be derived
from total-goals contracts:

```
For each over/under contract on a fixture:
    P(over k goals) = implied probability from contract price

Find the Poisson(λ) distribution that best matches the implied P(over k)
curve across k. λ is the implied total goal expectation.

Split λ into home and away xG using the home-win contract's implied
probability and the goal-difference distribution from related contracts.
```

This gives us a market-derived (home_xg, away_xg) pair per fixture that
we can substitute for our heuristic-derived (own_xg, opp_xg) in the
Poisson backend. The Benter combiner above is applied at the
fantasy-points level after distributing market-derived team xG across
players using our existing position shares.

## 11b.4 Information asymmetry: where the market does and does not know

The market's strengths:
- Match-outcome probabilities at major fixtures (deep liquidity)
- Total-goals expectations
- Top-tier player markets (Mbappé, Messi)

The market's weaknesses (where our model retains edge):
- Per-player fantasy points scoring under FIFA Fantasy's specific rules
  (the market doesn't price this directly)
- Cheap-tier defenders' clean-sheet expectations
- Auto-sub mechanics and bench-priority decisions
- Captain-x2 and ownership-differential considerations

The combined predictor should weight the market high where the market
has signal (premium attacker scoring, match outcome) and weight our
model high where it does not (low-tier players' fantasy returns,
fantasy-specific scoring components).

## 11b.5 Live market updates during matchdays

Prediction markets update continuously as matches play. A live captain-
switch decision (after the early kickoffs have finished and before the
late kickoffs lock) can incorporate the market's post-early-game
update:

```
Before late kickoff:
    captain_candidate = previously-chosen captain (still has match to play)
    alternative_captain = best XI player who hasn't kicked off

    For each candidate:
        live_market_xg = current Polymarket total-goals contract on their team's match
        update Poisson backend with live_market_xg
        recompute predicted_points
        recompute captain decision

Decision: switch if expected gain > switch cost (which is ~0 if both
candidates still have their match to play)
```

This is a real-time analogue of Benter's combiner applied to a single
decision point.

## 11b.6 Publication target

The combined "Benter for fantasy football with prediction-market data"
contribution is, to our reading of the literature in Section 3b,
publishable as a stand-alone paper. The natural venues:

- **MIT Sloan Sports Analytics Conference**: applied focus, accepts
  industry-quality work, March deadline
- **KDD Sports Analytics Workshop**: ML focus, accepts methodological
  novelty
- **IEEE Big Data**: Sports Analytics track, accepts ML-on-sports work
- **arXiv preprint**: as fast publication path with no review delay

The strongest framing emphasises:

1. The cross-domain transfer problem (club training, international
   inference) which OpenFPL does not address
2. The market-as-feature integration which no published fantasy
   football paper currently does
3. The live decision log with documented model vs human overrides,
   which provides a unique empirical contribution

A preliminary arXiv preprint after WC 2026 final, followed by a
peer-reviewed venue submission, is the realistic path.

## 11b.7 What we need to do during the rest of WC 2026 to support this

To make the Section 11b paper publishable, we need to collect data
**during** the tournament (markets close after each fixture finishes):

1. **Polymarket scraping**: hourly snapshots of all WC 2026 match and
   top-scorer contracts from now through the final. Store as
   `data/external/polymarket_snapshots/<timestamp>.json`.

2. **Kalshi scraping**: same cadence on Kalshi's WC contracts.

3. **Per-fixture market summary**: after each fixture finishes,
   compile (closing market implied probability, our model's prediction,
   realised outcome) into a per-fixture record. This is the training
   data for the Benter combiner.

4. **Per-player implied scoring rate**: derive from match-level total
   goals plus position shares (or per-player first-scorer contracts
   where they exist). Compare to realised fantasy points.

5. **Document the live decisions**: every captain choice (recorded in
   Section 8 of the whitepaper) should also note (a) the market's
   implied probability at decision time, (b) our model's prediction,
   (c) the realised outcome. This becomes the case study for the
   combined-predictor paper.

We have about 4 weeks of tournament remaining at the time of writing
(R32 through final). That is sufficient to collect a meaningful market-
data corpus, build the Benter combiner, and write the preprint.

A `src/fifa_fantasy/external/prediction_markets.py` module is the
first implementation step. Marked as future work for after the WC
final; the live decisions log accumulates the empirical evidence in
the meantime.
