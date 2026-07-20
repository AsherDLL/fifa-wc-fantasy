# A Multi-Backend Predictive Framework for FIFA World Cup 2026 Fantasy Football: Summary Edition

**Asher Davila** (Independent Researcher) and **Diego Guajardo** (Actuarial Scientist)

*This is the condensed edition of the full paper (target: at most 17 typeset pages including references). Citation markers use the bibliography keys of the full paper (docs/whitepaper/sections/13_references.bib.md); every number is copied from the generated artifacts named beneath each table, never retyped from drafts. Figures referenced by path are the same PDF/SVG assets the full paper uses.*

---

## Abstract

We present an open-source decision-support system for the official FIFA World Cup 2026 fantasy game, built, validated, and revised entirely while the tournament was running. The system combines three complementary expected-points backends (a calibrated price-anchored heuristic, an independent-Poisson structural model, and a gradient-boosted tree model), a per-position ensemble routed by held-out error, a mixed-integer squad optimizer that prices captaincy ceiling, player availability, and the backup-goalkeeper asymmetry, and a live decision layer whose every recommendation and override was logged with its realized outcome. Validation is leak-free throughout, in that each model change faced a walk-forward gate on realized tournament rounds before deployment, and failures are reported alongside successes. Replayed on frozen pre-round data across all eight rounds, the backends scored between 294 and 800 points against a random-squad baseline mean of 173. A skill audit of match-level forecasting found prediction-market prices losing to a simple Elo rating out of sample, earning the market a capped minority weight rather than its customary anchoring role. The live entry closed on 513 net points at global rank 905,675; its highest-scoring round was the final one, in which a pre-committed plan holding six starters from the eventual champion's defence, with the qualification booster active, returned 95 points and improved the entry's global rank by approximately 250,000 places. The complete system, data snapshots, and decision log are released as open source.

## 1. Introduction

Fantasy football research has concentrated on domestic leagues, where seasons are long, rosters are stable, and training data is plentiful [groos2025openfpl; mishra2025fpl]. A World Cup breaks all three assumptions at once: the tournament lasts eight fantasy rounds, squads are national sides that assemble for weeks, and no directly comparable historical data exists for the game being played, because the game itself (scoring rules, boosters, a 48-team format) is new. The 2026 edition adds structural wrinkles of its own: stage-varying budgets and nationality caps, five boosters, and a final round that scores two matches, the third-place play-off and the final, under one deadline.

This paper asks how much of such a tournament can be handled by models built under those constraints, and answers with a deployed system rather than a retrospective. Between 7 June and 19 July 2026 we built and operated a full pipeline: collectors for the official game's public JSON endpoints, feature builders, three predictor backends plus a simulation backend, a per-position ensemble, a MILP optimizer for squads, transfers, lineups, and captaincy, and a live layer that produced a recommendation before every deadline. Every round's predictions were frozen pre-kickoff; every transfer, captaincy, and lineup decision was logged with its rationale at decision time and its realized outcome afterwards, including the cases where a human domain prior overrode the model and whether the override was vindicated. The entry closed the tournament on 513 net points at global rank 905,675, with its highest-scoring round (95 points) being the final one, on a squad restructured by model-recommended transfers into the eventual champion's defensive block.

The contributions, in condensed form:

1. **System.** An end-to-end, reproducible pipeline for a data-poor, short-horizon fantasy tournament, released as open source with frozen data snapshots (https://github.com/AsherDLL/fifa-wc-fantasy).
2. **Leak-free in-tournament validation.** A walk-forward protocol on realized World Cup rounds that gated every deployed model change, including two mid-tournament redesigns (a form-aware retrain and a real-xG feature upgrade) and the rejections of five other candidates.
3. **A negative-results ledger.** Six documented failures, from a six-component heuristic upgrade to a Benter-style market combiner [benter1994computer], each with the measured regression that killed it.
4. **A market-skill audit.** A match-outcome forecaster in which prediction-market prices are one component among four, with influence earned from measured out-of-sample skill. The market lost to a plain Elo rating and earned a weight of 0.25.
5. **A complete decision log.** Eight rounds of human-plus-model decisions with realized outcomes, closing with a final round in which the contributions of model structure and operator correction are cleanly separable.

## 2. Data and system

Six data sources feed the system. The official FIFA Fantasy API (public JSON endpoints: players with prices, ownership, and per-round points; squads; fixtures) is the primary inference source, snapshotted daily. A public archive of every international match since 1872 [martj42] drives a rolling country Elo [elo1978rating] with tournament-weighted K-factors and a goal-margin multiplier, following common practice in Elo-based football forecasting [hvattum2010using; gilch2018elo]. Club-level match and odds data [footballdata] supports training-side enrichment and the goalkeeper formula calibration. Community dumps of English Premier League fantasy scoring [vaastav] supply the only labelled training corpus: 34,221 player-gameweek rows across three seasons after dropping non-appearances. A static copy of the FIFA World Ranking serves as a strength fallback. Finally, from mid-tournament, a community-maintained World Cup 2026 statistics dataset [mominul2026dataset] (CC0; real match expected goals, lineups with minutes, referee card histories) was validated against our own collected fixtures (102 of 102 match scores in agreement) and adopted for feature experiments and match forecasting.

The pipeline design favours simplicity. Collectors write dated Parquet snapshots; a feature builder emits one row per (player, round) with price, strength, Elo, form, rest-day, and (in the final generation) real-xG form columns; each backend maps that table to predicted points; the optimizer consumes whichever backend is selected and writes a recommendation artifact (JSON plus human-readable report) whose filename carries the backend, stage, and timestamp. A four-page static dashboard renders the same artifacts. Everything regenerates from the repository with two commands, and 352 tests pin the scoring rules, solvers, loaders, and report pages.

## 3. Methods

**Backend 1: heuristic.** A price-anchored formula with no training step: predicted points are a per-position coefficient times price, tilted by a strength gap blended from squad price differentials and country Elo, with a small home-nation term. It was deployed before the opening match and remained the transparent baseline throughout the tournament.

**Backend 2: structural Poisson.** Team expected goals are derived from price and Elo gaps around a base rate; player points follow from hand-set per-position goal and assist shares, exact clean-sheet probabilities under the Poisson goal model [maher1982modelling; dixon1997modelling], appearance points, and a save-bonus term for goalkeepers whose multiplier was calibrated empirically (0.50) after the theoretically derived value (1.13) measured worse.

**Backend 3: gradient-boosted trees.** A LightGBM model [ke2017lightgbm] per position with four heads: an L2 mean and pinball-loss quantile heads at the 10th, 50th, and 90th percentiles, reflecting the continued strength of tree ensembles on tabular data [shwartz2022tabular]. The final generation (v4xg) trains on eight features: price, home flag, price-strength gap, own and opponent top-eleven average prices, a leak-free trailing-form feature, and the team's trailing real expected goals for and against from [mominul2026dataset], the last two existing only on tournament rows (league rows carry missing values, which the trees handle natively). The model retrains on EPL data plus all completed World Cup rounds at every data refresh.

**Ensemble.** Per-position routing to the held-out winner: Poisson for goalkeepers, heuristic for defenders, GBM for midfielders and forwards (Section 4). The routing was fixed by the original held-out comparison and never needed revision; the v4xg upgrade improved the GBM at every position without changing the routing winners.

**Optimizer.** Three PuLP MILPs [pulp; cbc] share one constraint core (squad shape 2-5-5-3, stage budget, per-country cap, eliminated-player exclusion): fresh squad selection, transfer planning with the official 3-point hit as a penalized slack variable, and lineup selection over the seven legal formations. Three decision-layer corrections were added during the tournament, each with a measured justification. (i) An availability discount scales a player's effective points by 0.5 + 0.5 times a trailing participation rate; replayed on the heuristic backend it lifted realized squad points from 275 to 283 over the first four rounds. (ii) A ceiling-aware captain selector scores candidates by (1 - c) mean + c q90, where c is the entry's standings percentile; a leak-free captain backtest across three rounds credited the pure-ceiling rule with 21 raw captain points against 11 for the shipped mean-argmax rule (oracle: 42). (iii) A backup-goalkeeper term: only one of the two mandatory goalkeepers can score in a round, so a designated-scorer variable discounts the second goalkeeper to his automatic-substitution value (5 percent), making the optimum one strong starter plus the cheapest legal backup, an insight contributed by the operator and adopted into the objective the same day.

**Match forecaster.** For the closing rounds we built a separate match-outcome forecaster that applies the same treatment of market information operationally. Four components (a penalized maximum-likelihood Dixon-Coles model [dixon1997modelling] with ridge-resolved identifiability and nested walk-forward hyperparameters; an iterative scoring-rates model on a goals/xG blend; leak-free as-of-kickoff Elo; and the prediction market's tournament-winner price ratio) are pooled linearly, with weights learned from out-of-sample skill under proper scoring rules, reported leave-one-round-out, and with the market weight hard-capped at 0.25.

## 4. Validation

**Held-out league benchmark.** With no World Cup data in existence pre-tournament, backends were compared on the last nine gameweeks of the 2024-25 EPL season, held out from training; all backends score identical rows. No backend wins everywhere, and the split defines the ensemble routing:

| Position | n | Heuristic | Poisson | GBM |
|---|---|---|---|---|
| GK | 180 | 2.698 | **2.491** | 2.650 |
| DEF | 910 | **3.141** | 3.400 | 3.187 |
| MID | 1228 | 2.874 | 4.371 | **2.767** |
| FWD | 368 | 3.515 | 4.560 | **3.153** |

*Held-out EPL RMSE at the routing decision (v2-era GBM); source data/training/validation_report.json via paper/generated/tab_holdout_rmse.tex. Re-run under the final v4xg feature set the GBM improves at every position (2.566 / 3.147 / 2.687 / 3.081) with the routing winners unchanged.*

**Leak-free walk-forward on tournament rounds.** The league benchmark cannot detect distribution shift or evaluate tournament-only features, so every mid-tournament change was gated by walk-forward validation: for each completed round k of 2 through 7, train strictly on earlier data, predict round k, pool the errors. Six configurations summarize the tournament's feature history:

| Config | GK | DEF | MID | FWD | All |
|---|---|---|---|---|---|
| A: EPL, no form | 3.498 | 3.082 | 3.050 | 3.175 | 3.142 |
| B: EPL + form | 3.314 | 2.920 | 2.764 | 2.970 | 2.926 |
| C: EPL+WC + form | 2.404 | 2.850 | 2.621 | 2.759 | 2.699 |
| D: EPL+WC, proxy minutes | 2.403 | 2.840 | 2.636 | 2.803 | 2.710 |
| E: EPL+WC, real xG form | **2.275** | 2.836 | **2.530** | **2.739** | **2.645** |
| F: EPL+WC, real minutes | 2.383 | **2.832** | 2.588 | 2.766 | 2.681 |

*Pooled leak-free RMSE, holdout rounds 2-7; source data/evaluation/wc_forward_validation.json via paper/generated/tab_walkforward.tex; figure results/figures/fig_walkforward.pdf.*

The ordering summarizes the tournament's feature history. Form awareness (B) and tournament retraining (C) each improve every position; C was the deployed model for most of the knockouts. Configuration E, real team xG form, is the only feature candidate of the tournament to improve every position, and was deployed as the fourth-generation model two days before the final under a pre-registered rule (pooled improvement of at least 0.01, at most one position regressing). Configurations D and F concern the minutes signal, proxied and real respectively, and are discussed in Section 6. Training is exactly reproducible: seeds are pinned after run-to-run RMSE wobble of roughly plus-minus 0.05 was found masking feature comparisons.

## 5. Live deployment

The tournament record interleaves two series: the actual entry (human plus model, real transfer costs, boosters) and a retrospective cross-model backtest that replays every backend on frozen pre-round snapshots with unlimited free transfers (an upper bound, not a like-for-like comparison).

| Round | Ensemble (replay) | Entry (actual, net) | Random baseline |
|---|---|---|---|
| Group Matchday 1 | 107 | 33 | 22 |
| Group Matchday 2 | 116 | 84 | 23 |
| Group Matchday 3 | 116 | 42 | 22 |
| Round of 32 | 89 | 83 | 22 |
| Round of 16 | 70 | 66 | 20 |
| Quarter-final | 64 | 73 | 22 |
| Semi-final | 37 | 37 | 22 |
| Final | 83 | 95 | 26 |

*Source data/evaluation/backtest_summary.json via paper/generated/tab_season_summary.tex; cumulative trajectories in results/figures/fig_backtest_cumulative.pdf. Closing totals across all backends: GBM 800, ensemble 682, heuristic v2 544, entry 513 net, Monte Carlo 481, heuristic 480, Poisson 294, random mean 173.*

The round narratives compress as follows. The opening round was the entry's lowest-scoring (33): the captain blanked, a France triple-stack shared output, and no defender kept a clean sheet. Matchday 2 (84) was the group-stage peak, built on a domain override the model opposed (retaining and captaining an in-form premium the optimizer wanted to sell). The knockouts alternated: a wildcard rebuild delivered 83, the quarter-final's 73 beat every backend replay except the GBM's 76, and the semi-final collapsed to 37 when France lost 0-2 and the entry's France stack returned four raw points, leaving the entry at global rank 1,153,720.

The final round illustrates the division of labour between model and operator. Round 8 scored both closing matches under one deadline (the bronze play-off counts, a rule the system verified against the live game data after its own documentation claimed otherwise). The locked plan moved five transfers into the eventual champion's block, six Spain starters in all, with the Qualification Booster active and Mbappe captaining the bronze match. A planned mid-round armband switch had to be abandoned when live play established that the app locks the captaincy at the captain's own kickoff, contrary to the published help text; the operator committed a static 5-3-2, benching Kane on a rotation read. Both operator decisions were correct: Kane never played, and Spain's 1-0 win converted six clean sheets plus twelve booster points. The model's one clear miss was leaving Olise (10 points, started and scored in the bronze) on the bench of the model-guided eleven. Mbappe's hat-trick in the 4-6 bronze loss returned 17 raw points, doubled to 34, the entry's largest single return of the tournament. The round's 95 net points were the entry's best of all eight rounds and lifted the closing rank to 905,675, an improvement of approximately 250,000 places in one round.

Three cross-model findings survived the whole tournament. First, the Poisson backend's held-out goalkeeper win did not save it as a standalone squad picker (294 points, last): it systematically over-predicts mean points, and its deficit is a calibration finding, not noise. Second, captaincy is the highest-leverage single decision in a round and resists prospective optimization; the ceiling-aware selector narrows the gap but the within-squad optimal captain was identified in retrospect in every group round. Third, the entry's best rounds (84 and 95) were exactly the rounds where model recommendations and domain judgement pointed the same way, and its worst were rounds where one signal overrode the other.

## 6. Negative results and the market-skill audit

We maintain a structured ledger of failures, each with the measurement that killed it; the six entries plus the minutes coda are summarized here.

1. **Heuristic v2 (six-component stack).** Lost every group round to the one-line v1 it meant to replace; reverted after a three-round gate. Reported honestly: the replay ordering later reversed, and v2 finished the eight-round replay at 544 against v1's 480, underlining how noisy a three-round gate is in both directions.
2. **Team Elo difference as a GBM feature.** A deterministic A/B produced offsetting per-position deltas (net wash) with added distribution-shift risk at inference; not shipped.
3. **Theoretical goalkeeper save-bonus multiplier.** The value derived from the scoring rules (1.13) measured worse than an empirically calibrated 0.50 on both league and tournament data.
4. **Proxy minutes as model features (config D).** Participation proxied by positive scores regressed the walk-forward pool (2.710 vs 2.699); the signal moved to the optimizer's availability discount, where it measurably helps.
5. **A Benter-style market combiner** [benter1994computer]. Stacking tournament-winner prices onto every backend's predictions made every backend worse on realized rounds; the log implied win probabilities correlate at 0.89 with our own country Elo, making the market term a noisy near-substitute rather than a complement, with favorite-longshot bias [thaler1988favorite; snowberg2010explaining] a plausible contributor.
6. **Goalkeeper defensive form.** No realized-form signal beat the structural Poisson for goalkeepers at tournament sample sizes, as either a model feature or a formula adjustment.

**The minutes coda (config F).** When real lineup sheets arrived [mominul2026dataset], we replaced the proxy with true lagged start rates and minutes shares. Real minutes beat the proxy and the deployed config C, but only marginally (2.681), regressed forwards, and lost to the real-xG config E in the same run. Even the true minutes signal earns no place in the point model; minutes information belongs in the selection layer.

**The market-skill audit.** The forecaster of Section 3 makes the market question quantitative. Every component was scored out of sample under the same protocol (ranked probability score on the 90-minute outcome for held-out rounds 2-7; log-loss on knockout advance for rounds 4-7, the rounds with pre-kickoff market prices):

| Component | 1X2 RPS (n=78) | Advance log-loss (n=29-30) |
|---|---|---|
| Elo | **0.148** | **0.437** |
| Dixon-Coles | 0.204 | 0.600 |
| xG-Poisson | 0.205 | 0.618 |
| Market (trophy ratio) | -- | 0.535 |
| Uniform baseline | 0.237 | -- |

*Source data/evaluation/match_prediction_skill.json via paper/generated/tab_forecaster_skill.tex; figure results/figures/fig_forecast_skill.pdf.*

Two results are notable. A rating computed from a century and a half of international history (Elo) outperforms every component fit on a single tournament's data, including the market. And the market proxy loses to plain Elo on knockout advance despite pricing the same matches; the unconstrained ensemble weight fit assigns it 0.25, at which our precautionary cap never binds. Randomized probability-integral-transform checks reject no component, so the differences are skill, not calibration artifacts. In deployment the forecaster called the final on the modal outcome (Spain, 54 percent) and missed the bronze (a 59 percent favourite lost 4-6), a one-hit-one-miss record entirely consistent with its stated probabilities. The practical conclusion joins ledger entry 5 from the opposite direction: market sentiment earned a minority voice in our planning, not a veto, and the half-and-half market anchoring we began the knockouts with had overweighted it by roughly a factor of two [wolfers2004prediction; croxson2014information].

## 7. Lessons

Condensed from the full paper's lessons chapter, in rough order of transferability. (i) Validate on data the model has never seen, every time; both mid-tournament upgrades and all six ledger entries came from one walk-forward harness. (ii) Inspect downstream decisions, not just aggregate error: the overfit first-generation GBM was caught by implausible squads before any metric moved. (iii) Data contracts fail without warning; the API's round-points truncation silently corrupted form features for 49 percent of the player pool until padded, and the fantasy app's real locking behavior contradicted its own help text twice (the bronze-match scoring rule, in the system's favor; the captaincy lock, against it). Live verification against the running game beats documentation. (iv) Structural knowledge of the game yields measurable value: the one-goalkeeper insight, the bronze-round double-match structure, and booster timing each contributed measurable value that no player-level model could see. (v) For a trailing entry, variance is an asset: realized round totals swing by a factor of two with selection held constant (predicted round-total standard deviation 5.8 against 19.4 realized), so a chaser should captain the ceiling and play for spikes, which is what the final round did.

## 8. Conclusion

We asked how much of a data-poor, short-horizon fantasy tournament can be handled by models built, validated, and revised inside the tournament itself. On the evidence of this record, structural decisions, most point predictions, and round-level planning can be delegated to models behind leak-free validation gates. The corrections of highest value came from operator observation of the live game, and the entry's highest-scoring rounds coincided with agreement between model and operator. Every backend exceeded the random baseline by a factor of three or more; the entry closed on 513 net points at global rank 905,675 with its highest-scoring round last; and the negative-results ledger, from the six-component stack to the market audit, is the component of this record we consider most likely to generalize beyond this tournament. The complete system, the frozen snapshots underlying every table, and the round-by-round decision log are publicly released to support replication and extension.

## References

*Keys resolve in docs/whitepaper/sections/13_references.bib.md (extracted to paper/references.bib); the LaTeX edition renders these as a plainnat bibliography.*

- [benter1994computer] Benter, W. Computer based horse race handicapping and wagering systems: a report. In *Efficiency of Racetrack Betting Markets*, 1994.
- [croxson2014information] Croxson, K., and Reade, J. Information and efficiency: goal arrival in soccer betting. *The Economic Journal*, 2014.
- [dixon1997modelling] Dixon, M. J., and Coles, S. G. Modelling association football scores and inefficiencies in the football betting market. *JRSS C*, 1997.
- [eager2023football] Eager, E. A., and Erickson, R. A. *Football Analytics with Python and R*. O'Reilly, 2023.
- [elo1978rating] Elo, A. *The Rating of Chessplayers, Past and Present*. Arco, 1978.
- [footballdata] Football-Data.co.uk. Football results, statistics and odds. 2024.
- [gilch2018elo] Gilch, L., and Muller, S. On Elo-based prediction models for the FIFA World Cup. 2018.
- [groos2025openfpl] Groos, D. OpenFPL: an open-source forecasting method rivaling state-of-the-art Fantasy Premier League services. arXiv:2508.09992, 2025.
- [hvattum2010using] Hvattum, L. M., and Arntzen, H. Using ELO ratings for match result prediction in association football. *IJF*, 2010.
- [ke2017lightgbm] Ke, G., et al. LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS*, 2017.
- [maher1982modelling] Maher, M. J. Modelling association football scores. *Statistica Neerlandica*, 1982.
- [martj42] Martj42. International football results from 1872 to 2024. github.com/martj42/international_results, 2024.
- [mishra2025fpl] Mishra, S., et al. Machine-learning approaches to Fantasy Premier League team selection. 2025.
- [mominul2026dataset] Islam, MD Mominul. FIFA World Cup 2026 Dataset: Live and Updated Stats. github.com/mominullptr/FIFA-World-Cup-2026-Dataset, 2026.
- [pulp] Mitchell, S., et al. PuLP: a linear programming toolkit for Python.
- [cbc] Forrest, J., et al. COIN-OR CBC mixed-integer programming solver.
- [shwartz2022tabular] Shwartz-Ziv, R., and Armon, A. Tabular data: deep learning is not all you need. *Information Fusion*, 2022.
- [snowberg2010explaining] Snowberg, E., and Wolfers, J. Explaining the favorite-longshot bias. *JPE*, 2010.
- [thaler1988favorite] Thaler, R. H., and Ziemba, W. T. Anomalies: parimutuel betting markets. *JEP*, 1988.
- [vaastav] Anand, V. Fantasy Premier League historical data. github.com/vaastav/Fantasy-Premier-League.
- [winston2022mathletics] Winston, W. L., Nestler, S., and Pelechrinis, K. *Mathletics*. 2nd ed., Princeton, 2022.
- [wolfers2004prediction] Wolfers, J., and Zitzewitz, E. Prediction markets. *JEP*, 2004.
