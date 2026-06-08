# What every term means, plain English

A beginner-friendly tour of the predictors used by this project. No
prior ML knowledge assumed.

## The acronyms

| Term | What it means |
|---|---|
| EPL | English Premier League. The top tier of English club football. Runs August to May. |
| FPL | Fantasy Premier League. The official EPL fantasy game (`fantasy.premierleague.com`). Same scoring shape as the WC fantasy game. |
| GW | Gameweek. One round of EPL matches; an EPL season has 38 GWs. |
| WC | (FIFA) World Cup. The international tournament this project ultimately targets. |
| MD | Matchday. The WC's group-stage rounds (MD1, MD2, MD3). Conceptually the same as EPL's GW, just for the WC. |
| xG | Expected goals. A football analytics number that estimates how many goals a shot or a set of chances should produce. |
| RMSE | Root mean squared error. A standard way to measure how wrong a predictor is. See below. |
| Held-out | Data the model is not allowed to see during training. We score the model on it to estimate how it will do on real future data. |
| Training set | The labelled data the model learns from. |
| Validation set | The labelled data we use to check the model. The model never trains on this. |

## What "predicting fantasy points" actually means

A prediction is a single number: how many fantasy points we expect a
given player to score in a given match. For each (player, round) row
in our feature table, the predictor produces one number.

A good prediction is close to the realised actual points after the
match is played. RMSE measures the average size of the gap, with
larger errors penalised more than small ones:

```
RMSE = sqrt( mean over rows of (predicted - actual)^2 )
```

RMSE of 2.7 means the predictor is typically off by around 2.7 fantasy
points per row. Lower is better.

## What each approach does

### 1. Heuristic (the default in this project)

The word "heuristic" comes from the Greek `heuriskein`, "to find"
(same root as "eureka"). It is the standard name for a hand-written
rule of thumb: a fast, transparent formula built from domain
knowledge rather than from a probabilistic derivation or from
learning on labelled data. Heuristics are the right tool when an
exact answer is unavailable or unaffordable; they trade optimality
for clarity and speed.

The same idea is sometimes called any of: baseline model
(the file is `baseline.py` for this reason), rule-based predictor,
hand-crafted formula, closed-form predictor, parametric formula,
expert-system rule. In Fantasy Premier League community discussion
you may see "points-per-million model"; that is the simplest version
of the same idea.

A formula. Not learnt from data; written down by hand. The full
expression lives in `src/fifa_fantasy/model/baseline.py`:

```
predicted_points
  = points_per_price_unit[position] * price
  * (1 + alpha * tanh(combined_strength_signal))
  * (1 + beta * is_home)
  + premium_boost * max(0, price - $9M)
```

`points_per_price_unit` differs by position (GK 0.50, DEF 0.55, MID
0.60, FWD 0.65). `combined_strength_signal` blends the FIFA World
Ranking gap with a squad-price gap. The multipliers cap how much the
fixture and the home advantage can move the prediction.

Strengths: every number is explainable. No training data required. No
risk of distribution shift. Easy to read off "Lautaro vs Algeria, home,
$8.8M -> roughly 8 expected points".

Weaknesses: cannot learn from data. If a real pattern exists that the
formula does not capture, the heuristic will never find it. The slope
in price is linear within each position; in reality the marginal
return on the last $1M is not linear.

### 2. Poisson goals model (structural)

A statistical model based on the Poisson distribution. The Poisson
distribution describes counts of independent events; it is the
canonical model for goals in a football match. Steps:

1. Estimate expected goals per team for the fixture. The own team's
   expected goals is a base rate (1.30) multiplied by a matchup factor
   `exp(rank_diff/800 + price_diff/6)`, with a small home boost.
2. Distribute those goals across players by position. Forwards take
   ~50 percent of team goals; midfielders ~30 percent; defenders ~10
   percent; goalkeepers ~0.
3. Compute clean sheet probability as `exp(-opponent_xg)`. Under
   Poisson, that is the probability the opponent scores zero.
4. Translate goals, assists, clean sheet probability, appearance and
   the scouting bonus into Fantasy points using the same constants
   `scoring.py` encodes.

Strengths: every number is traceable to a football statement. The
clean sheet term is exact under the Poisson assumption. No training
data needed.

Weaknesses: the per-position goal share is a population average. The
model cannot tell a star striker apart from his backup if they have
the same price and same matchup. It can over-predict cheap players on
high-xG teams (which is what produced the $4.0M Spanish midfielder
captain pick).

### 3. GBM (gradient boosting machine, specifically LightGBM)

The only learnt model in this project. "Gradient boosting" is an
ensemble method: it builds a sequence of small decision trees where
each tree corrects the errors of the previous ones.

How it actually works, in five steps:

1. Start with a flat prediction equal to the average target value.
2. Compute the residual for every training row: `actual - current
   prediction`.
3. Fit a small tree (few splits) that predicts those residuals from
   the features.
4. Add a scaled version of the tree's prediction back into the running
   prediction.
5. Repeat hundreds of times. Each new tree is small; the sum of many
   small trees is a strong predictor.

"LightGBM" is Microsoft's efficient open-source implementation; it is
fast and is what most tabular ML benchmarks use.

What "training" the GBM requires:

- A pile of labelled rows, one per (player, match), with the features
  the model will see at inference time and the realised fantasy points
  as the target.
- We train one separate GBM per position (GK, DEF, MID, FWD) because
  the scoring mechanics differ enough that one model would have to
  waste capacity learning the per-position differences.
- We train four "heads" per position: the mean and three quantile
  predictors (10th, 50th, 90th percentile). Quantile predictors
  estimate the lower and upper bounds of likely outcomes, not just
  the average.

What the GBM is good for: capturing non-linear interactions between
features (e.g. "a 9M-plus forward against a weak defence at home
scores disproportionately well"), which a hand-written formula cannot
express compactly. It is the right tool for tabular data with tens of
thousands of rows.

What the GBM is bad at: extrapolation. If the training data has
nothing that looks like the inference data, the GBM has nothing to
draw on and its predictions can be unhelpful. We addressed this by
training on three full EPL seasons (FPL scoring is very close to FIFA
Fantasy scoring) and validating on a held-out chunk of one season.

## GBM v1 vs GBM v2

Two iterations are documented in this repo:

| Aspect | v1 (earlier) | v2 (current) |
|---|---|---|
| Training data | one EPL season (2024-25 only), ~11,500 rows | three EPL seasons (2022-23, 2023-24, 2024-25), ~34,200 rows |
| Hyperparameters | LightGBM defaults: 31 leaves per tree, 400 trees | tuned by held-out RMSE sweep: 15 leaves per tree, 200 trees |
| RMSE on held-out 2024-25 GW 30-38 | GK 2.710, DEF 3.297, MID 2.898, FWD 3.281 | GK 2.645, DEF 3.195, MID 2.795, FWD 3.160 |
| WC pre-tournament captain pick | Enzo Fernandez ($7.5M MID, E~6.8); cheap defenders dominated the squad | Ousmane Dembele ($10M MID, E~8.2); premium attackers in the squad |

The v1 squad looked aggressive because the model was overfitting: too
many leaves per tree and too many trees, on too little data. The
sweep at `src/fifa_fantasy/training/tune.py` showed that a lighter
tree structure beat v1 at every position. The shipped GBM (v2) uses
those lighter hyperparameters and the multi-season data.

The currently saved models in `data/models/gbm_*.txt` are v2. The
results files in `results/` written before 2026-06-08 06:40 UTC are
v1; everything after that timestamp is v2. The recommendation JSON
records `model_version: v2` for v2 outputs so you can tell them apart.

## Why not neural networks?

Three reasons.

1. **Data size.** Neural networks shine when there are hundreds of
   thousands to millions of training examples. We have around 34,000
   tabular rows. At that scale they overfit badly and have no
   structural prior to fall back on. Gradient boosting tree methods
   (LightGBM, XGBoost, CatBoost) dominate every recent benchmark on
   small to mid-size tabular data.

2. **Tabular features.** Our inputs are five numeric and boolean
   features per row. Neural networks earn their keep on images,
   audio, text and other high-dimensional unstructured data where the
   network can build up a hierarchy of features. There is nothing
   hierarchical to learn from price, is_home and three strength
   numbers; a tree split on each does the job.

3. **Interpretability and operational cost.** LightGBM can be saved
   as plain text, loaded instantly, scored in milliseconds on CPU, and
   inspected feature by feature. Neural networks require a deep
   learning runtime, GPU for any non-trivial size, and are harder to
   debug when they go wrong.

If we ever moved to high-frequency live data (per-second event feeds
of the match, for example) and an order of magnitude more rows, the
case for a small neural network would become real. For our current
data and feature set, LightGBM is the right tool.

## Picking between the three on any given match

Use the held-out RMSE table at the top of the README. Each backend
wins for a specific position:

- Goalkeeper: Poisson.
- Defender: heuristic.
- Midfielder, forward: GBM v2.

In practice the optimizer consumes one prediction column at a time,
so you pick a backend per run and live with that backend's strengths
and weaknesses for the whole squad. The lowest-RMSE single backend
across all positions is the heuristic; it is the safest pre-tournament
default. The GBM is the closest competitor and is the right choice
once we have a few rounds of WC data and can refit on
`--include-wc`.
