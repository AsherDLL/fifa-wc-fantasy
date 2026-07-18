# 12 - Conclusion

Status: written in the LaTeX paper (`paper/sections/12_conclusion.tex`), which is the authoritative text; only the final-round numbers remain open there. This md file is retained as the drafting record.

## Draft notes (superseded by paper/sections/12_conclusion.tex)


Three paragraphs:

1. **Recap of contributions.** Three-backend ensemble, live international
   Elo signal, MILP optimiser with stage-aware constraints, honest
   six-matchday empirical study. Held-out RMSE per-position best
   backend (Poisson GK, heuristic DEF, GBM MID/FWD). The system's
   recommendations beat a random-pick baseline by [X] points per round
   on average; the user's domain priors beat the system on captain
   choice in [Y of Z] observed cases.

2. **Where we landed.** Personal-league final position: [TBD]. The
   model-driven entry's strongest contribution was preventing illegal
   squad choices (budget, country cap, transfer-quota violations) and
   surfacing differential candidates the user would have missed without
   the candidate table. The weakest contribution was the GBM's
   under-prediction of premium international stars, which the heuristic
   and Poisson backends compensated for but did not eliminate.

3. **Call to action.** Open-sourcing the code, data pipeline, and
   decision log; future work on player-level rotation risk, team-news
   ingestion, Bayesian player-skill posteriors, and differential-aware
   MILP objectives. The system is a reasonable starting point for the
   next major tournament (Euros 2028).
