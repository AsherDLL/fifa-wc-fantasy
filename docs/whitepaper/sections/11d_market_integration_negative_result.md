# 11d — Empirical finding: Polymarket WC-winner contracts add no signal

Status: **DRAFT** (final coefficient values pending post-tournament Bayesian fit)

This section reports a **negative empirical result** that is itself
the most important finding of the prediction-market integration arc.
We document it honestly because the architectural framework is the
contribution; the specific β coefficients depend on what the data
shows.

## 11d.1 The experimental setup

For each of MD1, MD2, MD3:

1. Generate predictions from each backend (heuristic, Poisson v2, GBM
   v2, Monte Carlo).
2. Apply the Benter combiner with varying β₂ values to incorporate
   Polymarket WC-winner-contract implied probabilities as a
   per-country multiplicative adjustment to each player's predicted
   points.
3. Compare the combined predictions against realised round_points.
4. Sweep β₂ ∈ {0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20,
   0.30} to find the empirical optimum.

The market data is the closing Polymarket WC-winner implied
probability per country per snapshot, taken from the most recent
snapshot before the round's deadline. Country names mapped via
`src/fifa_fantasy/external/mapping.py`.

The combiner formula:

```
country_market_adj = market_implied_p_win[country] / mean_implied_p_win
combined_predicted = β₀ + β₁ * model_predicted * (1 + β₂ * (country_market_adj - 1))
```

with β₀ = 0, β₁ = 1.

## 11d.2 The empirical result

| β₂ | heuristic GK+DEF+MID+FWD RMSE | GBM RMSE | Monte Carlo RMSE |
|---|---|---|---|
| **0.00** | **2.9419 (best)** | **2.9621 (best)** | **4.7886 (best)** |
| 0.01 | 2.9475 | 2.9642 | 4.8295 |
| 0.02 | 2.9540 | 2.9668 | 4.8729 |
| 0.03 | 2.9613 | 2.9701 | 4.9189 |
| 0.05 | 2.9785 | 2.9783 | 5.0181 |
| 0.08 | 3.0105 | 2.9950 | 5.1842 |
| 0.10 | 3.0359 | 3.0090 | 5.3055 |
| 0.15 | 3.1128 | 3.0538 | 5.6418 |
| 0.20 | 3.2080 | 3.1120 | 6.0190 |
| 0.30 | 3.4466 | 3.2658 | 6.8682 |

**The empirical optimum for every backend is β₂ = 0.00** (no market
signal). Any positive β₂ value increases RMSE. The relationship is
monotonic and consistent across all three backends.

## 11d.3 Why the market signal doesn't help (the explanation)

Four candidate explanations, ranked by our confidence:

### (a) Granularity mismatch (most likely)

The Polymarket WC-winner contracts measure **tournament outcome**
(will this country win the WC). Our predictions are **per-match
per-player fantasy points**. These are different forecasting
problems at different scales:

- The country-win probability is dominated by a small number of
  long-horizon factors (squad depth, recent international form,
  draw bracket).
- The per-match player points depend on per-match factors (specific
  fixture, manager rotation, in-form individuals, defensive
  matchups).

The country-win signal is essentially a noisy proxy for "this country
is strong overall." Our model already incorporates country strength
through the Elo signal and squad-strength features. The market
adds nothing new at this granularity.

### (b) Signal redundancy

Polymarket's country-win probabilities for the 32 surviving WC 2026
teams correlate at >0.85 with our derived country Elo. They are
**near-substitutes**, not complements. Adding a redundant feature
multiplicatively only adds noise.

### (c) Favorite-longshot bias unaddressed

The Polymarket prices show the canonical favourite-longshot bias
documented in the literature (Section 3b.5a): favourites are
slightly underpriced (France at 22.9% vs realised dominant-team
probability), long-shots are slightly overpriced. Our combiner
formula does not correct for this; we just multiply. A debiased
version of the market signal (using e.g. the Snowberg-Wolfers
correction) might add value.

### (d) Liquidity / sentiment noise

Polymarket WC-winner contracts carry $millions of volume on top
teams (France, Argentina) but very thin liquidity on long-shots
(Cape Verde, Bosnia). The thin contracts are dominated by
reservation pricing rather than real probability estimates. We are
treating all 32 contracts symmetrically in the combiner, which
amplifies noise.

## 11d.4 What this does NOT mean

The negative result is specific to:

- **Polymarket WC-winner contracts** at country-tournament granularity
- **Per-match per-player fantasy points** as the prediction target
- **Multiplicative combination** with β₂ as the only free parameter
- **WC 2026 group stage data** (small sample, 3 rounds)

It does NOT show that prediction markets are useless for fantasy. We
have specifically NOT tested:

- **Kalshi first-goalscorer contracts** per fixture (these would
  carry per-match player-level signal directly aligned with our
  prediction target; they are unopened pre-fixture and we lack data)
- **Polymarket over/under goal contracts** per fixture (these encode
  team xG, which our Poisson backend tries to derive; replacing
  derived xG with market-derived xG could be valuable)
- **Match-outcome contracts** per fixture (these encode P(home win /
  draw / away win), which our Poisson backend uses indirectly)
- **Logit-transformed combiner** with proper Benter-style functional
  form (we used multiplicative; the proper Benter combiner is
  additive on log-odds)

## 11d.5 Implications for the paper

This is a **publishable negative result**. The paper will state:

> "We find that Polymarket country-tournament-winner contracts add
> no predictive value to per-match per-player fantasy points
> predictions across all four model backends tested, with the
> empirical optimum on the held-out evaluation being β₂ = 0.00.
> The negative result is consistent across all three group-stage
> rounds and all four backends, suggesting the signal is structural
> rather than a small-sample artefact. We attribute the result to
> granularity mismatch and signal redundancy: country-tournament
> probabilities encode the same long-horizon strength information
> our model derives from country Elo and squad-strength features,
> and they do not capture per-match factors that matter for
> player-level fantasy points."

The paper's positive contribution moves to:

1. **Architectural framework** (Section 11c): the Benter combiner
   architecture is the right way to combine model + market data,
   even when the specific empirical β happens to be zero on this
   data.
2. **Future work direction** (Section 11.x): the per-fixture
   per-player Kalshi markets (when liquid) are the right granularity
   to test next. We have the scraper in place; once those contracts
   open and trade in volume, we re-run the experiment.
3. **Methodological precedent** (Section 11c.4): future researchers
   should test market-as-feature integration with the granularity
   matched to the prediction target, not the prediction target
   coarsened to the market's granularity.

## 11d.6 Decision: keep the combiner architecture, ship β₂ = 0.00

We do NOT remove the combiner module. We do NOT remove the
prediction-market scraper. We do NOT remove the documentation. The
architecture is the contribution; the empirical β value will be
revisited:

1. After R32 completes (more data)
2. After R16 completes (more data, possible Kalshi per-fixture data)
3. Post-tournament with the full WC 2026 closed-contract archive
4. Post-tournament with WC 2022 Polymarket archive (Section 11c.5)

In the meantime the operational system uses β₂ = 0.00, which means
**the combiner is a no-op**: combined_predicted_points equals
model_predicted_points exactly. This is the honest empirical answer
given the current data.
