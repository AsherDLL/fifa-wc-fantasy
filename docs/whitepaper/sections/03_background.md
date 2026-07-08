# 03 - Background and related work

Status: **DRAFT**

## 3.1 Football analytics primitives

**Expected goals (xG).** A scalar in [0, 1] that estimates the
probability a shot becomes a goal given its characteristics: position on
the pitch, body part, defensive pressure, type of pass that led to the
shot. Aggregated per team per match, xG approximates the goals a side
"should have" scored. Public xG models are trained on tracked data from
providers like Opta. We do not use raw xG directly; instead we use
expected team-level goal output as a Poisson rate in our structural
backend.

**Football Elo.** Adapted from chess Elo (Elo, 1978), team-level ratings
that update after each match based on result, expected result given the
pre-match rating gap, and a tournament-importance multiplier. The most
prominent public implementation is the World Football Elo Ratings
maintained at eloratings.net. Elo's scale is calibrated: a 400-point
rating gap corresponds to ten-to-one odds for the higher-rated team.
Hvattum and Arntzen (2010) demonstrate Elo's predictive ability against
betting odds on football matches.

**FIFA Men's World Ranking.** The governing body's official ranking,
published monthly. Less responsive to recent form than Elo, and includes
a "match importance" multiplier that differs from Elo's by tournament.
For our purposes the FIFA ranking serves as a hand-maintained fallback
when our derived Elo is unavailable for a country.

## 3.2 Fantasy sports modelling

Fantasy Premier League (FPL) has a large public community producing
points-per-million ratios, ICT index analyses, and various neural and
gradient-boosting models. The dominant academic line of work (e.g.
Constantinou and Fenton, Robberechts and Davis) applies Bayesian
networks or generalised linear models to FPL with mixed success. The
community consensus on FPL prediction is that:

- Position-specific models materially outperform pooled models
- Recent form is the strongest single feature
- Fixture difficulty (often measured as opponent strength) is the second
- Quantile predictions are useful for differential strategies

We adopt all four of these conventions: one model per position; form
features in the heuristic and as an anchor in our Monte Carlo; an Elo
fixture-difficulty term; LightGBM quantile heads at the 10th, 50th, and
90th percentiles.

## 3.3 Tabular ML for player-level regression

LightGBM (Ke et al., 2017), XGBoost (Chen and Guestrin, 2016), and
CatBoost (Prokhorenkova et al., 2018) all dominate small-to-mid tabular
benchmarks. We use LightGBM specifically because:

- Native handling of NaN values, which our feature table relies on for
  the `rank_diff` and `country_elo_diff` columns that exist at inference
  but not in training
- Deterministic training given seeds, which matters for feature A/B
  testing
- Fast on commodity CPU, no GPU required, no model-serving runtime

We do not use neural networks. Our training set is ~34,000 rows of five
features. At that scale, gradient boosting beats neural networks
consistently on tabular benchmarks (Borisov et al., 2022; Shwartz-Ziv
and Armon, 2022). A neural network would need ten times the data and a
structural prior we cannot provide.

## 3.4 Structural football models

The structural Poisson approach to football match prediction traces to
Maher (1982) and Dixon and Coles (1997). Goals are modelled as Poisson
arrivals at team-level rates derived from offensive and defensive
strength parameters. We borrow the shape but simplify: instead of
estimating per-team latent strengths from match history, we use the
top-11 average squad price and the country Elo gap as observable
proxies. The model is interpretable end-to-end and requires no training,
which makes it useful as a cross-check on the GBM and as a backstop for
players the GBM has not seen.

## 3.5 Why we do not use neural networks

(Expanded reasoning beyond Section 3.3.)

Three constraints disqualify neural networks for our setting:

1. **Data scale.** Hundreds of thousands to millions of rows are the
   neural-network sweet spot. We have around 34,000.
2. **Feature dimensionality.** Five numeric and boolean features per
   row. Neural networks earn their keep on high-dimensional unstructured
   data (images, audio, text); there is nothing hierarchical to learn
   from a five-feature row that a tree split cannot capture.
3. **Interpretability and operational cost.** LightGBM saves as plain
   text, loads in milliseconds, scores on CPU, and can be inspected
   feature-by-feature. Neural networks require a deep-learning runtime
   for any non-trivial size.

The case for a neural network would become real only if we moved to
high-frequency in-match event data (per-second tracking) and an order of
magnitude more rows. The original `docs/algorithms-explained.md` makes
the same argument at a less technical level.
