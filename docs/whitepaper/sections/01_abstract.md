# 01 - Abstract

Status: **TODO** (write last, after results section is final)

## Notes for drafting

A typical academic abstract has four moves:

1. **Problem** - fantasy football point prediction for FIFA World Cup
   2026, a cross-domain transfer challenge (club-game training data, international play at inference).
2. **Approach** - three predictor backends combined with a MILP
   optimiser; Elo derived from full international match history; live
   decision tools driven by post-match snapshots.
3. **Results** - held-out RMSE per backend per position on EPL 2024-25 GW
   30-38; live tournament finish position; per-round MAE on the personal
   league.
4. **Lessons** - distribution shift between club and international play
   makes a single model wrong; ensembling with structural and
   formula-based backends recovers ground; user domain priors on minutes
   and rotation outperformed the model when team news landed.

## Stub (replace before submission)

> *We present a three-backend predictive framework for FIFA World Cup
> 2026 fantasy football: a hand-tuned heuristic, a structural Poisson
> goals model, and a LightGBM ensemble trained on three seasons of
> Premier League fantasy data. The system ingests live international
> match history via martj42's open dataset, derives an Elo signal
> calibrated to the natural 400-point log-odds scale, and integrates it
> through a NaN-safe priority hierarchy that falls back to the static
> FIFA ranking on absence. An MILP optimiser handles squad selection,
> stage-aware transfer planning with hit accounting, and starting XI
> choice. Held-out RMSE on EPL 2024-25 gameweeks 30-38 establishes the
> per-position best backend: Poisson for goalkeepers, heuristic for
> defenders, GBM for midfielders and forwards. Across the WC 2026 group
> stage and knockouts, the system's recommendations beat a random-pick
> baseline by [X] points per round on average; failure cases cluster
> around team-news events the model does not see (manager rotation in
> clinched matches) and around premium international stars
> (Messi, Mbappé) absent from the club-football training corpus. We
> open-source the code, data pipeline, and decision log.*
