# 03b — Literature review: prior work and our positioning

Status: **DRAFT**

This section reviews the directly relevant prior art and explicitly
positions our contributions against it. We searched arXiv, IEEE Xplore,
and ResearchGate in June 2026 for papers published in 2023-2025 on FPL
prediction, FIFA WC outcome modelling, prediction-market features for
sports, and parimutuel-style market handicapping (the historical
ancestor of the field). Coverage of the most recent quarter is limited
by publication lag.

## 3b.1 Foundational: Benter (1994) on parimutuel markets

Benter's *Computer Based Horse Race Handicapping and Wagering Systems:
A Report* (1994) is the seminal paper on combining a fundamental model
with public market signals. Benter trained a logit-based handicapping
model on horse-race data and showed that the residual after accounting
for the market's implied probabilities still carried predictive signal
beyond either source alone. The key methodological insight: the
public's implied probability is a strong baseline; a model that beats
the public must combine its own features WITH the public's information,
not replace it.

For our problem, the FIFA Fantasy game's `ownership_fraction` field is
the equivalent of the parimutuel market: it tells us what the field
thinks. Our heuristic and Poisson backends do not currently consume
ownership as a feature; the optimizer uses it only via the scouting
bonus encoding (under-5% threshold). Section 11 proposes adding an
ownership-informed term to the captain selection objective, motivated
directly by Benter's combining approach.

**Key reference:**

> Benter, W. (1994). Computer Based Horse Race Handicapping and
> Wagering Systems: A Report. In W. Ziemba, V. Lo, and D. Haush (Eds.),
> *Efficiency of Racetrack Betting Markets* (pp. 183-198). World
> Scientific.

## 3b.2 Soccer match outcome prediction

**Bunker et al. (2024)**: *Machine Learning for Soccer Match Result
Prediction* (arXiv:2403.07669). Demonstrates that gradient-boosted tree
models (notably CatBoost) on engineered soccer-specific ratings
(pi-ratings, ELO, momentum features) currently set the state of the
art on club-football match outcome prediction. Validation on multiple
European leagues. Their finding that pi-ratings (Constantinou & Fenton)
outperform ELO on goal-difference-aware tasks is relevant to our Elo
backend; we use ELO for tractability and falsifiability, accepting a
small accuracy cost.

**Distributional Forecasting (2025)**: arXiv:2501.05873. Forecasts
match results as full Skellam distributions over goal differences,
allowing uncertainty quantification beyond point estimates. Our
quantile-head GBM (q10/q50/q90) captures part of this signal at the
player level; their team-level distribution would be a useful prior
for our structural Poisson backend.

**Complex Networks Approach (2024)**: arXiv:2409.13098. Uses passing
network structural analysis as a feature input to ML predictors. We do
not have access to passing networks for international fixtures and do
not consider this approach.

**FIFA World Cup-specific (2025)**: *From Players to Champions: A
Generalizable Machine Learning Approach for Match Outcome Prediction
with Insights from the FIFA World Cup* (arXiv:2505.01902). Integrates
team-level and player-specific metrics with dimensionality reduction,
validated on FIFA WC 2022. Reports improvement over baselines but does
not address fantasy-points prediction or the cross-domain transfer
problem we tackle (training on club football, inferring on
international play). They note that incorporating player attributes
plus team composition improves WC outcome accuracy; our heuristic
backend uses analogous signals (top-11 average price as composition
proxy, individual price as ability proxy).

## 3b.3 Fantasy Premier League prediction (the closest prior art)

**OpenFPL (Holt, 2025)**: *OpenFPL: An open-source forecasting method
rivaling state-of-the-art Fantasy Premier League services*
(arXiv:2508.09992). The closest published comparable to our system.
Position-specific ensemble models trained on FPL and Understat data
from 2020-21 through 2023-24, validated prospectively on 2024-25.
Achieves accuracy comparable to a leading commercial service and
outperforms the commercial benchmark for high-return players (more
than 2 points). Models and inference code released on GitHub.

The architectural similarity to our work is striking:
- Position-specific models (they: ensemble; we: three backends per
  position with per-position best choice based on held-out RMSE)
- Multi-horizon forecasting (they: 1-3 gameweeks; we: stage horizons of
  1-3 rounds depending on stage)
- Open-source publication

**Our differentiation from OpenFPL:**
1. International tournament focus, not club season
2. Three-backend ensemble with explicit per-position best choice (their
   ensemble is opaque to which model dominates where)
3. Live Elo integration from a separate data source (martj42); their
   pipeline does not use match-history Elo
4. Stage-aware MILP optimizer with hit accounting; they focus on
   prediction accuracy, not squad optimization
5. Documented user-in-the-loop decision log

**Mishra & Mishra (2025)**: *Data-Driven Team Selection in Fantasy
Premier League Using Integer Programming and Predictive Modeling*
(arXiv:2505.02170). Mixed-integer linear programs for starting XI,
bench, and captain selection under FPL constraints (budget, formation,
3-per-club). Reports that ARIMA performs best for player-point
prediction, with linear regression and exponential smoothing as
baselines. The optimization formulation is very close to ours; the
prediction component is simpler (no ensemble, no quantile heads, no
structural model).

**Sentiment-Enriched Prediction (2025)**: *Players' Performance
Prediction for Fantasy Premier League, Using Transformer-based
Sentiment Analysis on News and Statistical Data* (Research-Gate).
Integrates per-player news article sentiment as an additional feature
into a CNN regressor. Reports MSE reduction from 6.27 to 5.63 over
state of the art. The team-news ingestion roadmap in Section 11 of our
whitepaper is the same idea applied to the international tournament
context.

**Earlier IEEE work (Bangaru et al., 2022)**: *Using ML Models to
Predict Points in Fantasy Premier League*. Linear regression, decision
tree, and random forest baselines on FPL data. Useful historical
reference but superseded by the 2024-2025 gradient-boosting and
ensemble approaches above.

## 3b.4 Systematic reviews

**A Systematic Review of Machine Learning in Sports Betting (2024)**:
arXiv:2410.21484. Surveys ML in sports betting markets including model
choice, feature engineering, and validation challenges. Identifies
distribution shift between training and live data as a recurring
failure mode; our Section 7.4 (rejected GBM v3 due to country-Elo
distribution shift) is an instance of the broader pattern.

## 3b.5 Prediction-market data as input feature (new direction)

We could not find published academic work specifically using Kalshi or
Polymarket data as a feature in fantasy-football prediction. The
existing literature on prediction markets and sports analytics has
focused on:

- Prediction markets as evidence of forecast accuracy (Wolfers and
  Zitzewitz, 2004)
- Comparison of market-implied probabilities to model-derived
  probabilities (multiple papers in finance, fewer in sports)
- Bet-sizing and Kelly criterion derivations from market odds

Kalshi and Polymarket both list:

- Match-outcome contracts (which team wins)
- First-goal-scorer contracts
- Total-goals contracts
- Top-scorer-of-tournament contracts (directly relevant to our captain
  decisions)

Polymarket's WC 2026 markets are public and have non-trivial volume
($100M+ on parlay-style contracts in mid-2025 per coverage).

**The innovation we propose:**

A combined model that ingests prediction-market implied probabilities
as an additional feature on top of the per-(player, round) feature
table. Specifically:

1. Per-match implied probabilities of win/draw/loss (consume Polymarket
   contracts directly)
2. Per-player implied probability of scoring in the match (where
   contracts exist; sparse coverage)
3. Per-tournament implied probability of top-scorer for each elite
   striker (Mbappé, Messi, Haaland)

This is the Benter (1994) approach updated for crypto-era prediction
markets and applied to fantasy football, which to our knowledge has not
appeared in the academic literature.

## 3b.6 Our novel contributions

Drawing the line between prior art and our contributions:

| Contribution | Closest prior art | Our delta |
|---|---|---|
| MILP for squad and captain selection | Mishra & Mishra (2025) | Stage-aware (group vs knockout); transfer-hit accounting; three-backend prediction input |
| Position-specific predictors | OpenFPL (2025); Bunker et al. (2024) | Three explicit backends (heuristic, Poisson, GBM) with per-position best choice from held-out validation |
| Cross-domain transfer | None published | Train on club EPL, infer on WC; identify and document distribution-shift failure modes |
| Live international Elo | Hvattum & Arntzen (2010) for Elo; nothing in fantasy | Roll Elo through martj42 history; integrate as priority signal over static FIFA rank |
| Prediction-market features | Benter (1994) for parimutuel | Polymarket/Kalshi as feature source for FIFA fantasy; new and untested |
| Tournament-specific captain Monte Carlo with ownership | None | Differential-adjusted captain selection that explicitly models the field's likely captain distribution |
| User-in-the-loop decision log | None | Documented every transfer round with model recommendation, human override, and realised outcome |

## 3b.7 Sources for the BibTeX file

To be added to `13_references.bib.md`:

```bibtex
@article{benter1994computer,
  title={Computer Based Horse Race Handicapping and Wagering Systems: A Report},
  author={Benter, William},
  journal={Efficiency of Racetrack Betting Markets},
  pages={183--198},
  year={1994},
  publisher={World Scientific}
}

@article{holt2025openfpl,
  title={{OpenFPL}: An open-source forecasting method rivaling state-of-the-art Fantasy Premier League services},
  author={Holt, [author surname per arxiv]},
  journal={arXiv preprint arXiv:2508.09992},
  year={2025},
  url={https://arxiv.org/abs/2508.09992}
}

@article{mishra2025fpl,
  title={Data-Driven Team Selection in Fantasy Premier League Using Integer Programming and Predictive Modeling Approach},
  author={Mishra and Mishra},
  journal={arXiv preprint arXiv:2505.02170},
  year={2025},
  url={https://arxiv.org/abs/2505.02170}
}

@article{bunker2024soccer,
  title={Machine Learning for Soccer Match Result Prediction},
  author={Bunker, Rory and others},
  journal={arXiv preprint arXiv:2403.07669},
  year={2024},
  url={https://arxiv.org/abs/2403.07669}
}

@article{worldcup2025players,
  title={From Players to Champions: A Generalizable Machine Learning Approach for Match Outcome Prediction with Insights from the FIFA World Cup},
  journal={arXiv preprint arXiv:2505.01902},
  year={2025},
  url={https://arxiv.org/abs/2505.01902}
}

@article{wolfers2004prediction,
  title={Prediction markets},
  author={Wolfers, Justin and Zitzewitz, Eric},
  journal={Journal of Economic Perspectives},
  volume={18},
  number={2},
  pages={107--126},
  year={2004}
}

@article{sportsbetting2024review,
  title={A Systematic Review of Machine Learning in Sports Betting: Techniques, Challenges, and Future Directions},
  journal={arXiv preprint arXiv:2410.21484},
  year={2024},
  url={https://arxiv.org/abs/2410.21484}
}
```
