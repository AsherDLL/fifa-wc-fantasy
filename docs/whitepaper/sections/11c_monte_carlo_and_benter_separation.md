# 11c - Monte Carlo simulator and the Benter separation

Status: **DRAFT**

This section formalises two architectural commitments for the paper's
methodological contribution:

1. The Monte Carlo per-match simulator as a first-class predictor
   backend, parallel to the heuristic / Poisson / GBM.
2. Prediction-market data (Polymarket, Kalshi) treated as a
   **separate meta-signal** combined with the model output via
   Benter's logit combiner, **not** as a feature inside any soccer-
   analytics model.

## 11c.1 Why Monte Carlo (the architectural commitment)

The retrospective backtest (Section 8b) showed Monte Carlo and the
heuristic tied at 212-211 cumulative points across MD1-MD3 versus
the GBM at 161 and Poisson at 72. The Monte Carlo backend has three
properties the others lack:

1. **Natural uncertainty quantification**. The Monte Carlo simulator
   produces a per-player distribution (mean, p10, p50, p90) rather
   than a single point estimate. This matters more in single-game
   knockout rounds than in multi-game group horizons.
2. **Form-weighted scoring**. The Monte Carlo backend explicitly
   scales position shares by per-player form multipliers
   (`avg_per_match / position_avg`), so a striker performing 1.5x
   his price class gets 1.5x the goal distribution. The heuristic
   and GBM do not have this anchor.
3. **Direct calibration against realised data**. Each tournament
   round produces new realised data; the Monte Carlo backend
   re-weights its priors immediately. The GBM cannot do this without
   retraining on a tiny WC corpus.

The architectural choice: keep all four backends (heuristic, Poisson,
GBM, Monte Carlo) as parallel predictors. The optimiser consumes one
backend's `predicted_points` per run; we choose the backend per stage
based on:

- **Group stage (multi-round horizon, partial information)**:
  heuristic - best held-out RMSE on DEF, robust to distribution
  shift.
- **Knockout rounds (single game, full information from group
  stage)**: Monte Carlo - uses realised group-stage data as anchor,
  produces uncertainty bands useful for captain decisions.
- **All stages**: GBM as a check on the heuristic when both agree
  the pick is solid.
- **Poisson**: not used as primary in production until the
  recalibration pass.

## 11c.2 The Monte Carlo simulation contract

For each `(player, match)` row, simulate `N_SIMULATIONS = 1000`
versions of the match:

```
for s in 1..N:
    # Team xG with multiplicative form/news noise
    own_xg_s = own_xg * lognormal(0, sigma=0.15)
    opp_xg_s = opp_xg * lognormal(0, sigma=0.15)

    # Sample team goal counts
    team_goals = Poisson(own_xg_s)
    opp_goals  = Poisson(opp_xg_s)

    # Distribute to players using form-weighted position share
    player_goal_rate = GOAL_SHARE[pos] * form_multiplier(player)
    player_goals_s   = Poisson(team_goals * player_goal_rate)

    # Same for assists at 0.7x rate
    player_assist_rate = ASSIST_SHARE[pos] * form_multiplier(player) * 0.7
    player_assists_s   = Poisson(team_goals * player_assist_rate)

    # Goalkeepers: shots × save_pct
    shots = Poisson(opp_xg_s * SHOT_PER_XG_CALIBRATED)
    saves = Binomial(shots, SAVE_PCT)

    # Clean sheet
    clean_sheet = (opp_goals == 0)

    # Fantasy points
    pts_s[player] = appearance + goals × GOAL_POINTS[pos]
                  + assists × ASSIST_POINTS
                  + clean_sheet × CS_POINTS[pos]
                  + (saves // 3) × 1   for GK
                  + DEF goals_conceded_penalty

# Output:
mean = pts.mean()
p10  = percentile(pts, 10)
p50  = percentile(pts, 50)
p90  = percentile(pts, 90)
```

The realised-data anchor is `form_multiplier(player) = clip(realised_avg /
position_avg, 0.3, 3.0)`. Clipping prevents extreme outliers from
distorting the simulation; the 0.3-3.0 range covers ~99% of
realised player ratios from the WC group stage data we have.

Implementation: `src/fifa_fantasy/model/monte_carlo.py`. CLI:
`python -m fifa_fantasy.model --backend monte_carlo`.

### What we did NOT include in v1

- **Minutes sampling**: every row assumes the player plays at least
  60 minutes. A future Bayesian extension would model minutes
  explicitly with a Beta distribution.
- **Game state**: a team up 2-0 plays differently. We do not model
  in-match dynamics in v1.
- **Polymarket implied goal-difference**: kept separate (Section
  11c.3) rather than embedded as a feature.

## 11c.3 Why prediction markets are NOT embedded as features

This is a sharp architectural decision driven by feedback from the
project's co-author (D. Guajardo, in conversation):

> "Prediction-market is 'casino' style data, not soccer analyst data
> source of truth."

The objection is legitimate. Polymarket and Kalshi reflect bettor
sentiment composed of:

- Sharp bettors with research (signal)
- Casual bettors with narrative/favoritism bias (noise)
- Liquidity providers arbitraging gross mispricings (signal)
- Reservation-pricing for long-tail outcomes (noise)

Embedding these directly into a feature column of a soccer-analytics
model conflates the soccer signal with the bettor-sentiment noise.
The model cannot tell them apart.

**The correct architectural separation**: prediction markets are an
**independent meta-signal**, combined with the soccer-analytics model
output via Benter's (1994) logit combiner:

```
logit(p_combined) = β₀ + β₁ logit(p_model) + β₂ logit(p_market)
```

The coefficients β are estimated by maximum likelihood on historical
data (Polymarket WC 2022 archive + WC 2026 in-tournament data as it
accumulates). The combiner explicitly **weights the market against
the model**, rather than forcing the model to ingest noisy market
data as if it were soccer truth.

### Why this matters for the paper

The honest reviewer's question is: "if you use Polymarket prices,
aren't you just adding noise?" The Benter-separation architecture
gives the answer: we do not assume Polymarket prices are true; we
**measure how much true signal they carry** by fitting β empirically.
If markets are mostly noise, β₂ → 0 and they get ignored. If they
carry information our model lacks (squad depth, team news, narrative
factors that move bettors with skin in the game), β₂ > 0 and we
benefit.

The architectural commitment: **two separate inferences, one
combiner**.

```
┌────────────────────────────┐
│  Soccer-analytics model    │
│  (heuristic / Poisson /    │── predicted_points (per player)
│   GBM / Monte Carlo)       │
└────────────────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Benter combiner │── combined_predicted_points
        │ (β₀, β₁, β₂)    │
        └─────────────────┘
                 ▲
                 │
┌────────────────────────────┐
│  Prediction-market signal  │
│  (Polymarket + Kalshi)     │── market_implied_probability
│  per fixture, per player   │
└────────────────────────────┘
```

This is the central architectural decision the paper documents.

## 11c.4 What "soccer is more random" means for the precedent

Football is structurally more random than:

- **Horse racing** (Benter's original domain): a horse's quality is
  observable from past races and physical attributes; the favourite
  wins approximately 33% of the time.
- **Tennis**: head-to-head matches with no team composition variance;
  serve-and-rally dynamics are stable per player.
- **American football**: discrete play-by-play structure with stable
  per-position outputs.

Football has:

- Continuous play with goal events at ~0.03 per minute mean rate
- Heavy-tailed distributions (a striker can score 5 goals in a game
  or 0)
- Team-level emergent behaviour (defensive line shape, possession
  patterns) that is hard to attribute to individuals
- Manager-level variance (rotation, tactical changes, substitution
  timing) that is exogenous to the player

This means a soccer fantasy paper that adapts Benter's combiner
must use **larger β confidence intervals** than the horse-racing
literature. The combiner's coefficients are learnt with appropriate
uncertainty quantification (e.g. Bayesian logistic regression with
weakly informative priors).

The precedent the paper sets:

1. **Architectural**: treat market data as a meta-signal in a Benter
   combiner, not as a feature in a soccer-analytics model.
2. **Methodological**: use Monte Carlo simulation for single-game
   knockout rounds where uncertainty is high; use simpler heuristics
   for multi-game group horizons.
3. **Statistical**: report β-confidence intervals for the combiner
   that reflect football's intrinsic variance, not Benter's original
   horse-racing-tight confidence.
4. **Empirical**: cross-validate every architectural choice against
   held-out data (EPL 2024-25 for the predictors; WC 2022 Polymarket
   archive for the combiner).

## 11c.5 Scoping the combiner implementation

The combiner needs historical paired data: `(market_implied,
model_predicted, realised)` per fixture or per player. We have:

- **WC 2026 in-tournament**: in-progress, accumulating per the docker
  snapshot scheduler (Section 6).
- **WC 2022 Polymarket archive**: needs separate scrape; Polymarket
  was active in 2022 but the contracts are now closed. Their
  historical-data API exists but requires authentication.

The plan for the combiner implementation:

1. After WC 2026 final, scrape the closed Polymarket WC 2026
   contracts for the complete time-series.
2. Pair with our model predictions (already in `data/processed/`
   per snapshot) and realised outcomes.
3. Fit β coefficients by Bayesian logistic regression with PyMC.
4. Report β posteriors with 95% credible intervals.
5. Validate on holdout splits within WC 2026 (e.g. group stage as
   training, knockout rounds as test).

Implementation lives in `src/fifa_fantasy/external/benter_combiner.py`
(to be created after the tournament; we collect data now, fit
after).

This is the publishable contribution.
