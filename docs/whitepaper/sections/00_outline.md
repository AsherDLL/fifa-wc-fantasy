# 00 — Outline and section list

Working title:

> **A Multi-Backend Predictive Framework for FIFA World Cup Fantasy Football: Heuristic, Poisson, and Gradient Boosting Approaches with Live Elo Integration**

Target length: 25-35 pages, single column, including figures and references.

## Per-section scope

### 1. Abstract (~250 words)
One paragraph each: problem, approach, key contributions, headline
results, lessons.

### 2. Introduction
- The FIFA Fantasy WC 2026 prediction problem
- Cross-domain transfer challenge (club-football models to international play)
- Resource constraints: ~34,000 EPL rows, 0 WC training rows pre-tournament
- Contributions:
  1. Three-backend ensemble (heuristic, Poisson, GBM v2) with held-out validation
  2. Live international Elo signal from martj42, calibrated to 400-point log-odds scale
  3. MILP optimizer with stage-aware constraints (budget, country cap, transfer hits)
  4. Honest empirical study: model EV vs human domain priors over six matchdays

### 3. Background and related work
- Football analytics primer: xG, Elo, FIFA rankings, FBref-style stats
- Fantasy sports modelling history (FPL community, "points-per-million")
- Tabular ML: why LightGBM dominates small-to-mid datasets
- Why neural networks are wrong for this problem at our scale (the textbook answer)
- Relevant prior work in fantasy points prediction and football modelling

### 4. Data
- FIFA Fantasy API: endpoints, schemas, drift history (the dict-vs-list MD1 incident)
- martj42 international_results: history depth, Elo derivation
- football-data.co.uk: club leagues, odds, club Elo
- Vaastav community FPL dumps: three full EPL seasons as training proxy
- Static FIFA Men's World Ranking snapshot as fallback
- Data cleaning and normalization (Pydantic two-stage validation)
- Country-name harmonization across sources (martj42 vs FIFA Fantasy vs FBref)

### 5. Methodology
- Per-(player, round) feature table contract
- Three predictor backends:
  - Heuristic: position coefficient times price times matchup factor times home factor plus premium boost
  - Poisson: structural team xG model with positional goal/assist shares
  - GBM v2: LightGBM mean + three quantile heads per position, trained on EPL FPL data
- Elo integration: priority hierarchy (country_elo > rank_diff > price-only)
- MILP optimizer (PuLP/CBC): squad selection, transfer planning with hits, lineup choice
- Scouting bonus encoding (more than 4 pts AND under 5% ownership equals plus 2)

### 6. Implementation and architecture
- Pipeline: collector then features then model then optimizer (then live tools)
- Repository layout, package structure
- Reproducibility: deterministic LightGBM seed, frozen requirements, Parquet artefacts
- Test strategy: scoring rules pinned to exhaustive parametrized tests

### 7. Held-out validation (EPL 2024-25 GW 30-38)
- Validation strategy: train on full prior seasons plus 2024-25 GW 1-29, predict GW 30-38
- Per-position RMSE per backend (the canonical table)
- v1 vs v2 GBM comparison: why we shipped lighter hyperparameters and three-season data
- The team_elo_diff feature A/B (rejected) and what it tells us about distribution shift

### 8. Live tournament results (MD1 through Final)
This section grows match-by-match through the tournament.
- Per-round actual score versus model prediction
- Captain choices, ownership snapshots, model rank vs realised rank
- Transfer log: every move with rationale and outcome
- Personal-league position tracker

### 9. Critical analysis
- Where the model dominated human guess (cite specific matchdays)
- Where the human domain prior dominated model (team-news cases, minutes risk)
- Failure modes: GBM blindness to non-EPL stars (Messi, Mbappé), rotation in clinched-group games
- Ownership and differential: how it interacts with the points objective

### 10. Lessons learned
- Realised points must anchor model predictions for elite players (the "Messi total = 33, model says 5" problem)
- Rotation risk per group-stage matchday is a first-class feature, not a fudge factor
- Captain selection is the single highest-leverage decision; the model is rarely contradicted on the ranking but is sometimes contradicted on the magnitude
- Differential strategy at the bottom of a league has different optimal moves than at the top
- The cost of an unforced transfer hit is consistently underestimated by users (us)

### 11. Better approaches and future work
- Player-level form weighting that updates continuously
- Pre-game team-news scraper (press conferences, predicted XIs from major outlets)
- Bayesian update per match (posterior on player skill, opponent strength, manager rotation propensity)
- Tournament-specific GBM trained on prior WC and Euros data
- Captain Monte Carlo that explicitly models the league-ownership distribution
- Stage-aware horizon weighting (R32 and beyond are single-game; group stage was multi-game)
- Auto-sub mechanic modelling (currently we rely on the FIFA game; we could derive a richer bench-priority objective)

### 12. Conclusion
- Three-paragraph summary
- Honest grade: where did our season finish
- Open call to action: open-sourcing the code and data

### 13. References
- BibTeX entries for: LightGBM paper (Ke et al.), Poisson-football models (Dixon-Coles, Maher),
  Elo's chess ratings adapted to football (Hvattum, Arntzen), FPL community work,
  PuLP/COIN-OR documentation, FIFA Fantasy WC 2026 official rules

### Appendix
- Full feature schema
- Model artefact format
- Reproducibility checklist
