# 03d - Verified source and method gap analysis (July 2026 audit)

Status: **DRAFT**

This section records the results of a systematic gap analysis run on
2026-07-13/14: which data sources, modeling methods, and literature
could be added to or cited by the system, checked against what already
exists in the stack. Every claim below survived adversarial
verification (three independent verifiers per claim, each instructed to
refute; claims killed on a 2-of-3 refute vote). Live-source checks are
as of 2026-07-14. Items already covered in Section 3b (Benter, OpenFPL,
Mishra & Mishra, Bunker et al., Levitt, Wolfers & Zitzewitz) are not
repeated here.

## 3d.1 Data sources worth adding

### soccerdata (Python)

One library wrapping eight sources: Club Elo, ESPN, FBref,
Football-Data.co.uk, Sofascore, SoFIFA, Understat, WhoScored. Adds
over the current collector: Understat shot-level xG
(`read_shot_events`), WhoScored event data, ESPN/FBref lineups,
player season stats, ClubElo history, and Football-Data.co.uk
historical results plus bookmaker odds usable for backtest
calibration. Apache-2.0, one-way compatible as a dependency of this
GPL-3.0 release. Active (v1.9.0, April 2026). Caveats: Understat
covers big-5 club leagues only, not internationals; the package
license does not cover scraped-data terms (WhoScored/Opta ToS are
restrictive).

- https://github.com/probberechts/soccerdata

### worldfootballR (R)

GPL-3 R package; the only verified route to FBref international data
(World Cup, Euros, Copa America, friendlies): shot-level events per
international match (`fb_match_shooting`: shooter, timing, body part,
distance), seven categories of advanced per-match player and team
stats (summary, passing, passing_types, defense, possession, misc,
keeper), match lineups, Transfermarkt injury histories
(`tm_player_injury_history`), and player market values. Hard caveat:
the repository was archived 2025-09-18 and is unmaintained; CRAN
carries a stale version; FotMob support was removed. Use for
historical feature building and paper validation, not live 2026
ingestion.

- https://worldfootballr.sportsdataverse.org/

### StatsBomb Open Data (validation set)

Free event data with per-shot `statsbomb_xg` covering WC 2018 and
2022, Euro 2020/2024, Copa America 2024, and historical WCs. The
natural independent validation set for our expected-points and
shot-rate proxies. License flag: the Public Data User Agreement
requires attribution and is not an OSI license; fetch at runtime,
do not redistribute the raw data inside this repository.

- https://github.com/statsbomb/open-data

### Wyscout / Pappalardo dataset (open-licensed validation set)

Pappalardo et al., Scientific Data 6:236 (2019); figshare collection
4415000. Spatio-temporal event data for WC 2018, Euro 2016, and the
2017/18 big-5 leagues under CC BY 4.0, the only WC event dataset with
a genuinely open license. Loadable via socceraction
(`PublicWyscoutLoader`) or kloppy. Concrete use: derive historical
fantasy points from events and backtest the expected-points pipeline
on a real World Cup.

Note: the broader "SportsDataverse ecosystem" enumeration largely
failed verification (claims conflated org-hosted with ecosystem-listed
packages); only worldfootballR and soccerdata survived. Treat other
SportsDataverse soccer-package claims as unvetted.

## 3d.2 Method upgrades (cheap, testable on the existing backtester)

### Nested Poisson regression (Gilch & Muller)

Gilch & Muller, "On Elo based prediction models for the FIFA Worldcup
2018" (arXiv:1806.01930): the weaker team's goal rate conditions on
the stronger team's realized goals,
log lambda_B = gamma_0 + gamma_1 * Elo_A + gamma_2 * G_A. Backtested
best of four Poisson variants (independent, bivariate,
diagonal-inflated bivariate, nested) on WC 2010 and WC 2014 by Brier
and RPS. A few-line change to our Poisson backend. The same paper is
a direct citable precedent for the whole Elo + Poisson + Monte Carlo
architecture (eloratings.net Elo covariates, neutral-ground training
window, 100,000 tournament simulations); the player-level fantasy
layer is the novelty this paper can claim over it. Caveats: preprint,
two tournaments, self-benchmarked; frame as upgrade candidate, not
proven superiority.

### Dynamic Elo inside tournament simulations

Same source: simulations that freeze Elo for the whole tournament
shift stage probabilities by up to 5 percentage points and
systematically favor stronger teams (they miss underdog runs).
Applies only if we add multi-round chained simulation; the current
`monte_carlo.py` is per-match with a fixed `country_elo_diff`.

### Dixon-Coles dependence adjustment (with its documented limit)

The independence assumption in our joint score model is empirically
misspecified: across 9,130 big-5 matches (2014-15 to 2018-19) the
home/away goal correlation is -0.085 (p < 0.001; Petretta, Schiavon
& Diquigiovanni, arXiv:2103.07272, published in Statistical
Modelling), and `monte_carlo.py` draws team and opponent goals as
independent Poisson variates. If adding the Dixon-Coles rho
correction, cite its structural limit: rho only redistributes mass
among the (0,0), (1,0), (0,1), (1,1) cells, so Under/Over 2.5
probabilities are mathematically identical to the independence model.
Open question: whether any of this moves fantasy expected points (as
opposed to match-outcome probabilities); a walk-forward A/B on the
existing backtester answers it cheaply. Either result is paper
material, as a justified upgrade or as a justified deliberate
simplification.

### Pi-ratings (primary source)

Constantinou & Fenton, JQAS 9(1), 2013, DOI 10.1515/jqas-2012-0036.
Dynamic team rating with separate home/away components, updated from
predicted-vs-observed goal-difference error diminished by
psi(e) = 3 * log10(1 + e). Outperformed both Hvattum & Arntzen (2010)
football Elo variants and was profitable against best bookmaker odds
over five EPL seasons. Python implementation exists in penaltyblog.
Section 3b.2 already cites pi-ratings secondhand via Bunker et al.;
this is the primary source. Caveats: the home/away split matters less
at mostly neutral-venue WC matches; profitability is partly
in-sample; the QMUL PDF link is dead, cite the DOI or the
constantinou.info mirror.

## 3d.3 Book lineage (verified, with two corrections)

### Mathletics, 2nd edition (Winston, Nestler, Pelechrinis, 2022)

Verified soccer-relevant content (Google Books search-inside plus the
official chapter code repos at github.com/mathletics-book):

- Ch 39 Soccer Analytics: logistic-regression xG on 2016 MLS shot
  data; Markov-chain possession-value player model following Sarah
  Rudd; penalty-kick game theory (Chiappori, Levitt & Groseclose,
  AER 2002).
- Ch 46 Rating Sports Teams: least-squares ratings with home edge,
  applied to the Premier League; Poisson goals with Skellam
  goal-difference for home/draw/away probabilities, citing Karlis &
  Ntzoufras (2003).
- Ch 47 From Point Ratings to Probabilities: ratings-to-probability
  conversion plus Monte Carlo tournament simulation.
- Ch 55 Elo Ratings: Elo with draws as half-win.
- Ch 44-45 Sports Gambling: de-vigged implied probabilities, soccer
  1X2 arbitrage worked example, market efficiency via Levitt (2004).
- Ch 60 Daily Fantasy Sports: salary-cap lineup construction
  formulated explicitly as a knapsack integer program. This is the
  direct textbook citation for our MILP squad optimizer.
- Ch 49 Kelly criterion (optional, only if captain/differential picks
  are framed as bankroll-style risk allocation).

Correction 1: Mathletics 2e does NOT cite Maher (1982), Dixon & Coles
(1997), or Benter (1994); verified zero hits including the
bibliography. It also contains no World Cup model. Cite Maher,
Dixon & Coles, and Benter directly (Benter already in 3b.1); Winston
gets us Skellam-based outcome probabilities, Karlis & Ntzoufras, and
the knapsack-DFS formulation.

Correction 2: Eager & Erickson, "Football Analytics with Python and
R" (O'Reilly, 2023) is exclusively American football; its entire data
pipeline is nflfastR / nfl_data_py and the TOC has no soccer content.
Cite only for transferable methodology: Poisson regression for count
outcomes, player-stat stability and regression to the mean,
simulation-based projections, and open-data reproducibility practice.

Canonical citation chain for the Poisson lineage: Maher (1982)
independent double-Poisson; Dixon & Coles (1997) low-score dependence
and time-decay weighting, explicitly motivated by exploiting
betting-market inefficiency (the standard football precedent for our
market-vs-model framing); Karlis & Ntzoufras (2003) bivariate Poisson.

## 3d.4 Industry and newsletter benchmarks (all URLs live 2026-07-14)

Useful as independent forecast benchmarks for calibration comparisons
and as citable methodology write-ups:

- Silver Bulletin (Nate Silver): PELE model methodology (Predictive
  Elo with Lineup Equilibria; 100k sims, player market values),
  posting WC 2026 knockout forecasts now.
  https://www.natesilver.net/p/pele-methodology
- Expecting Goals (Michael Caley): WC 2026 team-strength projections,
  updated every 2-3 days; adjusted-xG lineage.
  https://www.expectinggoals.com
- Neil Paine: 2026 World Cup odds tracker built on Polymarket prices,
  directly parallel to our `data/external/prediction_markets`
  snapshots; ideal market-vs-model comparison.
  https://neilpaine.substack.com
- The Analyst (Opta): live WC 2026 supercomputer predictions and
  power rankings. https://theanalyst.com
- Jakob Sanderson: ownership-leverage / EV frameworks for tournament
  fantasy (NFL context, transfers to captain and differential
  decisions). https://jakobsanderson.substack.com
- robotjames (ex-Betfair trader): 2026 series on edge, EV, and
  model-vs-market pricing including in-play soccer models in Python.
  https://robotjames.substack.com
- Also active: Get Goalside (Mark Thompson), The Transfer Flow (Ted
  Knutson et al.), No Grass in the Clouds (Ryan O'Hanlon), Grace on
  Soccer. Dormant but citable archives: John Muller's newsletter,
  Tony ElHabr's blog (expected-points season simulations).

## 3d.5 Open questions from the audit

1. Does a Dixon-Coles or nested-Poisson correction measurably change
   fantasy expected points and captain/transfer decisions, given the
   small -0.085 correlation? (Walk-forward A/B.)
2. Will FBref/soccerdata cover WC 2026 matches in near-real-time, and
   do FBref bot policy and rate limits permit that use in a public
   GPL-3.0 repository?
3. Pi-ratings vs our international Elo as a team-strength feature:
   worth one ablation run given the neutral-venue caveat.

## 3d.6 Additions for the BibTeX file

Added to `13_references.bib.md` (Maher, Dixon & Coles, and Hvattum &
Arntzen were already present as `maher1982modelling`,
`dixon1997modelling`, `hvattum2010using`). New entries:

```bibtex
@article{karlis2003analysis,
  title={Analysis of sports data by using bivariate Poisson models},
  author={Karlis, Dimitris and Ntzoufras, Ioannis},
  journal={Journal of the Royal Statistical Society: Series D (The Statistician)},
  volume={52},
  number={3},
  pages={381--393},
  year={2003}
}

@article{constantinou2013pi,
  title={Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adversaries},
  author={Constantinou, Anthony C. and Fenton, Norman E.},
  journal={Journal of Quantitative Analysis in Sports},
  volume={9},
  number={1},
  pages={37--50},
  year={2013},
  doi={10.1515/jqas-2012-0036}
}

@article{gilch2018elo,
  title={On Elo based prediction models for the FIFA Worldcup 2018},
  author={Gilch, Lorenz A. and M{\"u}ller, Sebastian},
  journal={arXiv preprint arXiv:1806.01930},
  year={2018},
  url={https://arxiv.org/abs/1806.01930}
}

@article{petretta2022dependence,
  title={On the dependence in football match outcomes},
  author={Petretta, Marco and Schiavon, Lorenzo and Diquigiovanni, Jacopo},
  journal={Statistical Modelling},
  year={2025},
  doi={10.1177/1471082X241238802},
  note={arXiv:2103.07272}
}

@article{pappalardo2019public,
  title={A public data set of spatio-temporal match events in soccer competitions},
  author={Pappalardo, Luca and Cintia, Paolo and Rossi, Alessio and Massucco, Emanuele and Ferragina, Paolo and Pedreschi, Dino and Giannotti, Fosca},
  journal={Scientific Data},
  volume={6},
  number={236},
  year={2019},
  doi={10.1038/s41597-019-0247-7}
}

@book{winston2022mathletics,
  title={Mathletics: How Gamblers, Managers, and Fans Use Mathematics in Sports},
  author={Winston, Wayne L. and Nestler, Scott and Pelechrinis, Konstantinos},
  edition={2},
  year={2022},
  publisher={Princeton University Press}
}

@book{eager2023football,
  title={Football Analytics with Python and R},
  author={Eager, Eric A. and Erickson, Richard A.},
  year={2023},
  publisher={O'Reilly Media},
  note={American football; cited for transferable methodology only}
}
```
