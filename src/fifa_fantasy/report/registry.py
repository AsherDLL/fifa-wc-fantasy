"""Canonical registry of every predictive model this project has run.

One place that answers, without digging through git history or the
whitepaper, three questions the dashboard and the notebook both need:
which backends exist, what does each one actually compute, and what
happened to the ideas that did not survive validation.

Everything here is declarative. No I/O, no pandas; the record fields are
plain strings and tuples so templates and notebooks can render them
directly. Dates come from the commit history (git log --date=short).
Formulas are stored twice: `formula_text` is a copyable plain-text block,
`formula_mathtext` is a tuple of matplotlib-mathtext lines rendered to
SVG by report.figures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRecord:
    """A backend (or backend generation) and its documented life story."""

    key: str                    # backend name as used in --backend / files
    label: str                  # display name
    status: str                 # shipped | superseded | rejected | experimental
    generation: str             # short lineage tag, e.g. "GBM v3form"
    introduced: str             # ISO date of the introducing commit
    summary: str                # what the model does, in plain prose
    formula_text: str           # copyable plain-text formula
    formula_mathtext: tuple[str, ...] = ()   # matplotlib mathtext lines
    routing_role: str | None = None          # where the ensemble uses it
    evidence: tuple[str, ...] = ()           # repo-relative artifact paths
    whitepaper_refs: tuple[str, ...] = ()    # docs/whitepaper section files
    in_production_tick: bool = False         # run by the 12h daemon tick


@dataclass(frozen=True)
class NegativeResult:
    """An idea that was built, measured, and rejected on the evidence."""

    key: str
    label: str
    tested: str                 # ISO date
    summary: str                # what was tried and why it was rejected
    measured: str               # the number that killed it
    evidence: tuple[str, ...] = ()
    whitepaper_refs: tuple[str, ...] = ()


ALLOWED_STATUSES = ("shipped", "superseded", "rejected", "experimental")

# The five backends an analyst can select today, one record each. The GBM
# record describes the current generation; earlier generations appear in
# GENERATIONS below for the evolution timeline.
MODEL_REGISTRY: tuple[ModelRecord, ...] = (
    ModelRecord(
        key="heuristic",
        label="Heuristic",
        status="shipped",
        generation="Heuristic v1",
        introduced="2026-06-07",
        summary=(
            "A price-anchored formula with no training step. Each player's "
            "expectation is a position coefficient times price, adjusted by a "
            "saturating matchup term that blends squad-price gap with "
            "national-team strength (live Elo when available, FIFA ranking as "
            "the fallback), plus a small home boost. It is transparent, "
            "immune to distribution shift, and has been the defender "
            "workhorse since matchday 1."
        ),
        formula_text=(
            "predicted = c[pos] * price * (1 + 0.40*tanh(z)) * (1 + 0.05*home)\n"
            "z = 0.35*(strength_diff / 2.0) + 0.65*(elo_diff / 400)\n"
            "    (rank_diff / 250 when Elo is missing; price-only when both are)\n"
            "c = {GK 0.50, DEF 0.55, MID 0.60, FWD 0.65}\n"
            "zeroed when status != playing, squad eliminated, or news says benched"
        ),
        formula_mathtext=(
            r"$\hat{y} = c_{pos}\, p\,\left(1 + 0.40\,\tanh z\right)"
            r"\left(1 + 0.05\,h\right)$",
            r"$z = 0.35\,\frac{\Delta_{price}}{2.0} + "
            r"0.65\,\frac{\Delta_{elo}}{400}$",
        ),
        routing_role="Routed the DEF position by the ensemble",
        evidence=("data/training/validation_report.json",
                  "src/fifa_fantasy/model/baseline.py"),
        whitepaper_refs=("docs/whitepaper/sections/05_methodology.md",
                         "docs/whitepaper/sections/07_validation.md"),
        in_production_tick=True,
    ),
    ModelRecord(
        key="poisson",
        label="Poisson",
        status="shipped",
        generation="Poisson (GK formula v2)",
        introduced="2026-06-07",
        summary=(
            "A structural expected-goals model with no training step. Team xG "
            "is built from strength signals, distributed across positions with "
            "fixed goal and assist shares, then mapped to fantasy points "
            "through the official scoring rules, including a Poisson "
            "clean-sheet probability. The goalkeeper save bonus scales with "
            "opponent xG at an empirically calibrated 0.50 per unit, the v2 "
            "formula that replaced the flat constant on 2026-06-28. It owns "
            "goalkeepers in the ensemble."
        ),
        formula_text=(
            "own_xg = 1.30 * exp(clip(m, +-1.2)) * (1 + 0.10*home)\n"
            "m = elo_diff/400 (or rank_diff/800) + strength_diff/6\n"
            "predicted = 2 + own_xg*goal_share[pos]*goal_pts[pos]\n"
            "          + 0.7*own_xg*assist_share[pos]*3\n"
            "          + exp(-opp_xg)*cs_pts[pos] + gk_and_def_terms\n"
            "gk: +0.50*opp_xg - E[max(0,k-1)];  def: -0.5*E[max(0,k-1)]\n"
            "E[max(0,k-1)] = opp_xg - 1 + exp(-opp_xg)   (k ~ Poisson(opp_xg))"
        ),
        formula_mathtext=(
            r"$\lambda_{own} = 1.30\,e^{\,\mathrm{clip}(m,\,\pm 1.2)}"
            r"\left(1 + 0.10\,h\right),\quad "
            r"m = \frac{\Delta_{elo}}{400} + \frac{\Delta_{price}}{6}$",
            r"$\hat{y} = 2 + \lambda_{own}\,s^{G}_{pos}\,G_{pos} + "
            r"2.1\,\lambda_{own}\,s^{A}_{pos} + e^{-\lambda_{opp}}\,C_{pos} + "
            r"\mathrm{GK{:}}\ 0.50\,\lambda_{opp} - "
            r"E[\max(0,k{-}1)]$",
        ),
        routing_role="Routed the GK position by the ensemble",
        evidence=("data/evaluation/gk_formula_ab_2026-06-29.json",
                  "src/fifa_fantasy/model/poisson.py"),
        whitepaper_refs=("docs/whitepaper/sections/09c_gk_formula_evolution.md",
                         "docs/whitepaper/sections/09b_gk_insight.md"),
        in_production_tick=True,
    ),
    ModelRecord(
        key="gbm",
        label="GBM",
        status="shipped",
        generation="GBM v4xg",
        introduced="2026-07-17",
        summary=(
            "A LightGBM model per position with four heads each, a mean "
            "regression plus pinball-loss quantile heads at the 10th, 50th "
            "and 90th percentiles. Trained on three EPL FPL seasons and, "
            "since v3form, retrained every tick on the completed World Cup "
            "rounds as well, with a leak-free trailing-form feature. v4xg "
            "adds real team xG for/against trailing form from the community "
            "WC-2026 match dataset (walk-forward config E), which improved "
            "the leak-free RMSE at every position, pooled 2.699 to 2.645. "
            "It owns midfielders and forwards in the ensemble, and its q90 "
            "head drives the ceiling-seeking captain pick."
        ),
        formula_text=(
            "predicted[pos] = LightGBM_pos(price_millions, is_home, strength_diff,\n"
            "                              squad_top_n_avg_price,\n"
            "                              opp_squad_top_n_avg_price, form_lag,\n"
            "                              team_xg_form_real, team_xga_form_real)\n"
            "heads: mean (L2) and quantiles tau in {0.10, 0.50, 0.90}\n"
            "pinball loss L_tau(y, q) = max(tau*(y-q), (tau-1)*(y-q))\n"
            "15 leaves, 200 trees, lr 0.05, seed 42, deterministic"
        ),
        formula_mathtext=(
            r"$\hat{y}_{pos} = f^{\,\mathrm{LGBM}}_{pos}\!\left(p,\ h,\ "
            r"\Delta_{price},\ \bar{p}_{11},\ \bar{p}^{\,opp}_{11},\ "
            r"\mathrm{form\_lag},\ \mathrm{xG}^{\pm}_{form}\right)$",
            r"$L_{\tau}(y, q) = \max\!\left(\tau\,(y-q),\ "
            r"(\tau - 1)\,(y-q)\right),\quad \tau \in \{0.1,\ 0.5,\ 0.9\}$",
        ),
        routing_role="Routed the MID and FWD positions by the ensemble",
        evidence=("data/evaluation/wc_forward_validation.json",
                  "data/models/",
                  "src/fifa_fantasy/model/gbm.py"),
        whitepaper_refs=(
            "docs/whitepaper/sections/11f_model_improvement_retrospective.md",
            "docs/whitepaper/sections/07_validation.md",
        ),
        in_production_tick=True,
    ),
    ModelRecord(
        key="monte_carlo",
        label="Monte Carlo",
        status="experimental",
        generation="Monte Carlo simulator",
        introduced="2026-06-28",
        summary=(
            "A per-match simulator rather than a point estimator. It samples "
            "team xG around the Poisson backend's estimate, draws goals, "
            "distributes them with form-weighted player multipliers, samples "
            "goalkeeper saves, and scores each simulated match with the "
            "official rules, so the output is a full points distribution. "
            "Its mean has never passed held-out validation, so it is not run "
            "by the production tick; its backtest row appears here for "
            "comparison and its distributional ideas fed the captain work."
        ),
        formula_text=(
            "team_xg ~ own_xg * LogNormal(0, sigma)\n"
            "goals ~ Poisson(team_xg), assigned by position share * form multiplier\n"
            "assists ~ 0.7 * goals, same shape\n"
            "gk saves ~ Binomial(shots(opp_xg), save_pct)\n"
            "predicted = mean(simulated points); p10/p50/p90 from the same draws"
        ),
        formula_mathtext=(
            r"$G_{team} \sim \mathrm{Poisson}\!\left(\lambda_{own}\,"
            r"\varepsilon\right),\quad \varepsilon \sim "
            r"\mathrm{LogNormal}(0, \sigma)$",
            r"$\hat{y} = \mathrm{mean}(Y_{sim}),\quad q_{10},\ q_{50},\ "
            r"q_{90}\ \mathrm{from}\ Y_{sim},\quad N = 1000$",
        ),
        routing_role=None,
        evidence=("data/evaluation/backtest_summary.json",
                  "src/fifa_fantasy/model/monte_carlo.py"),
        whitepaper_refs=(
            "docs/whitepaper/sections/11c_monte_carlo_and_benter_separation.md",
        ),
        in_production_tick=False,
    ),
    ModelRecord(
        key="ensemble",
        label="Ensemble",
        status="shipped",
        generation="Per-position ensemble",
        introduced="2026-07-08",
        summary=(
            "Not a new formula but a routing decision. The held-out EPL "
            "validation showed no single backend wins every position, so the "
            "ensemble runs the component backends and keeps, per position, "
            "the prediction from the documented winner. Poisson takes "
            "goalkeepers, the heuristic takes defenders, the GBM takes "
            "midfielders and forwards and contributes its quantiles. This is "
            "the backend the optimizer consumes and the squad this desk "
            "recommends."
        ),
        formula_text=(
            "predicted(i) = predicted_route(pos_i)(i)\n"
            "route = {GK: poisson, DEF: heuristic, MID: gbm, FWD: gbm}\n"
            "route is re-derived from data/training/validation_report.json\n"
            "(lowest held-out RMSE per position); quantiles carried where gbm"
        ),
        formula_mathtext=(
            r"$\hat{y}(i) = \hat{y}_{r(pos_i)}(i),\quad r = \{\mathrm{GK}\!:"
            r"\mathrm{poisson},\ \mathrm{DEF}\!:\mathrm{heuristic},\ "
            r"\mathrm{MID}\!:\mathrm{gbm},\ \mathrm{FWD}\!:\mathrm{gbm}\}$",
        ),
        routing_role="The production default; the official squad comes from it",
        evidence=("data/training/validation_report.json",
                  "src/fifa_fantasy/model/ensemble.py"),
        whitepaper_refs=(
            "docs/whitepaper/sections/11f_model_improvement_retrospective.md",
        ),
        in_production_tick=True,
    ),
)

# Superseded generations, for the evolution timeline. These are not tabs;
# they are history.
GENERATIONS: tuple[ModelRecord, ...] = (
    ModelRecord(
        key="gbm_v1",
        label="GBM v1",
        status="superseded",
        generation="GBM v1",
        introduced="2026-06-07",
        summary=(
            "Single EPL season (2024-25), LightGBM defaults with 31 leaves "
            "and 400 trees. It overfit league idiosyncrasies, famously "
            "clustering Scandinavian defenders and proposing Chris Wood as "
            "vice-captain of a World Cup squad. Superseded the same day it "
            "was benchmarked."
        ),
        formula_text="LightGBM per position, 5 features, defaults (31 leaves, 400 trees)",
        whitepaper_refs=("docs/whitepaper/sections/07_validation.md",),
        evidence=("docs/gbm.md",),
    ),
    ModelRecord(
        key="gbm_v2",
        label="GBM v2",
        status="superseded",
        generation="GBM v2",
        introduced="2026-06-07",
        summary=(
            "Three EPL seasons (34,221 rows) and tuned hyperparameters, 15 "
            "leaves and 200 trees. It won midfielders and forwards on the "
            "held-out EPL benchmark and shipped as the tournament starter, "
            "but it was form-blind: five features, all price and strength, "
            "none about how the player had actually been scoring. That defect "
            "is what v3form fixed."
        ),
        formula_text="LightGBM per position, 5 features (no form), 15 leaves, 200 trees",
        whitepaper_refs=("docs/whitepaper/sections/07_validation.md",
                         "docs/whitepaper/sections/11f_model_improvement_retrospective.md"),
        evidence=("docs/gbm.md",),
    ),
    ModelRecord(
        key="heuristic_v2",
        label="Heuristic v2",
        status="rejected",
        generation="Heuristic v2",
        introduced="2026-06-28",
        summary=(
            "Six stacked bonuses on top of the v1 formula, including a 0.40 "
            "realized-form anchor. It lost to plain v1 in every backtested "
            "round, cumulative 190 against 211, a textbook case of adding "
            "components faster than they can be validated. Kept in the tree "
            "for reproducibility; never routed."
        ),
        formula_text="v1 formula + 6 additive bonuses (form anchor 0.40, and others)",
        whitepaper_refs=("docs/whitepaper/sections/05c_heuristic_v2_attempt.md",),
        evidence=("src/fifa_fantasy/model/baseline_v2.py",
                  "data/evaluation/backtest_summary.json"),
    ),
)

NEGATIVE_RESULTS: tuple[NegativeResult, ...] = (
    NegativeResult(
        key="heuristic_v2",
        label="Heuristic v2 component stack",
        tested="2026-06-28",
        summary=(
            "Six intuition-driven bonuses added at once to the heuristic. "
            "Every backtested round preferred the unmodified v1. The lesson "
            "written into the whitepaper is procedural, one component at a "
            "time, each behind a held-out gate."
        ),
        measured="Cumulative realized points 190 vs 211 for unmodified v1",
        evidence=("data/evaluation/backtest_summary.json",),
        whitepaper_refs=("docs/whitepaper/sections/05c_heuristic_v2_attempt.md",),
    ),
    NegativeResult(
        key="team_elo_diff_feature",
        label="team_elo_diff as a GBM feature",
        tested="2026-07-08",
        summary=(
            "Adding the Elo gap as a sixth GBM feature was a wash on held-out "
            "EPL and carried distribution-shift risk, since international Elo "
            "gaps dwarf club gaps. The signal is consumed instead by the "
            "heuristic and Poisson backends, where it is used directly."
        ),
        measured="EPL held-out RMSE: GK -0.018, MID -0.019, DEF +0.033, FWD +0.015",
        evidence=("src/fifa_fantasy/model/gbm.py",),
        whitepaper_refs=("docs/whitepaper/sections/07_validation.md",),
    ),
    NegativeResult(
        key="gk_theoretical_multiplier",
        label="GK save bonus, theoretical 1.13 multiplier",
        tested="2026-06-28",
        summary=(
            "The save-bonus multiplier derived from first principles (shots "
            "per xG times save rate over saves per bonus point) overestimated "
            "reality. Not all opponent xG becomes on-target shots and the "
            "median keeper does not save at the elite rate. The empirical "
            "sweep landed on 0.50, which shipped as GK formula v2."
        ),
        measured="Empirical sweep minimum at 0.50, not the derived 1.13",
        evidence=("data/evaluation/gk_formula_ab_2026-06-29.json",
                  "scripts/gk_formula_ab.py"),
        whitepaper_refs=("docs/whitepaper/sections/09c_gk_formula_evolution.md",),
    ),
    NegativeResult(
        key="minutes_features",
        label="start_rate_lag and team_gc_form as GBM features",
        tested="2026-07-08",
        summary=(
            "Config D of the walk-forward validation added participation and "
            "team defensive form to the feature set. Both regressed the "
            "leak-free RMSE at every position; the participation proxy is too "
            "noisy over four rounds and defensive form is collinear with the "
            "strength signals. Each signal is still used where it helps, in "
            "the Poisson clean-sheet term and the rotation-risk flag."
        ),
        measured=("Pooled walk-forward RMSE 3.281 to 3.307 when added "
                  "(July-8 population, before the round-points padding fix; "
                  "the regenerated artifact reproduces the same ordering, "
                  "C 2.699 vs D 2.710)"),
        evidence=("data/evaluation/wc_forward_validation.json",
                  "scripts/wc_forward_validation.py"),
        whitepaper_refs=("docs/whitepaper/sections/11g_minutes_captaincy_variance.md",),
    ),
    NegativeResult(
        key="real_minutes_features",
        label="Real lineup minutes as GBM features (config F)",
        tested="2026-07-16",
        summary=(
            "Config F replaced the noisy points-based participation proxy "
            "with REAL start rates and minutes shares from the community "
            "WC-2026 lineup dataset. It beat the deployed config C but only "
            "marginally, regressed forwards, and lost to the real-xG config "
            "E evaluated in the same run, so E was deployed and F was not. "
            "Real minutes remain in the optimizer's availability discount, "
            "where they help."
        ),
        measured=("Pooled walk-forward RMSE 2.681 vs C 2.699 and E 2.645; "
                  "FWD regressed 2.759 to 2.766"),
        evidence=("data/evaluation/wc_forward_validation.json",
                  "scripts/wc_forward_validation.py"),
        whitepaper_refs=("docs/whitepaper/sections/11f_model_improvement_retrospective.md",),
    ),
    NegativeResult(
        key="benter_market_combiner",
        label="Benter combiner with Polymarket prices",
        tested="2026-06-28",
        summary=(
            "Combining model predictions with prediction-market win "
            "probabilities, after Benter's racetrack architecture, made every "
            "backend worse in every round. Tournament-winner contracts carry "
            "no per-player, per-match information the models lack. Market "
            "data stays on the intelligence page as context and deliberately "
            "does not feed the models."
        ),
        measured="delta RMSE positive for all backend-round pairs (combined worse)",
        evidence=("data/evaluation/with_vs_without_market.json",
                  "scripts/with_vs_without_market.py"),
        whitepaper_refs=(
            "docs/whitepaper/sections/11d_market_integration_negative_result.md",
        ),
    ),
    NegativeResult(
        key="gk_defensive_form",
        label="GK defensive-form adjustment",
        tested="2026-07-08",
        summary=(
            "Scaling goalkeeper predictions by the team's recent goals "
            "conceded looked sensible and measured worse. Four rounds of "
            "conceded goals is mostly schedule, not skill, and the strength "
            "signals already carry the opponent."
        ),
        measured="Walk-forward GK RMSE regressed when applied",
        evidence=("scripts/wc_forward_validation.py",),
        whitepaper_refs=("docs/whitepaper/sections/11g_minutes_captaincy_variance.md",),
    ),
)

# Dated milestones for the research-page timeline, oldest first. Each entry
# is (date, title, one-sentence description).
TIMELINE: tuple[tuple[str, str, str], ...] = (
    ("2026-06-07", "Heuristic v1",
     "Price-anchored baseline ships so the optimizer has predictions before a ball is kicked."),
    ("2026-06-07", "GBM v1 and v2",
     "First learned model overfits a single EPL season; the same day, three seasons and tuned "
     "hyperparameters take midfield and forward on held-out EPL."),
    ("2026-06-07", "Poisson backend",
     "A structural xG model joins as the interpretable cross-check and the goalkeeper specialist."),
    ("2026-06-28", "GK formula v2",
     "The save bonus becomes opponent-xG scaled at an empirically calibrated 0.50, "
     "after the theoretical 1.13 measured worse."),
    ("2026-06-28", "Monte Carlo and the backtest harness",
     "A distribution-producing simulator arrives with the cross-model squad backtest "
     "that scores every backend on realized rounds."),
    ("2026-06-28", "Two negative results recorded",
     "Heuristic v2 loses every round to v1, and the Benter market combiner makes "
     "every backend worse; both are documented and shelved."),
    ("2026-07-08", "GBM v3form and the ensemble",
     "A leak-free trailing-form feature plus retraining on completed World Cup rounds "
     "fixes the form-blindness defect; per-position routing becomes the production backend."),
    ("2026-07-17", "GBM v4xg: real xG form",
     "Real team xG for/against trailing form from the community WC-2026 dataset "
     "(walk-forward config E) improves the leak-free RMSE at every position, pooled "
     "2.699 to 2.645; the real-minutes config F passes only marginally and is not shipped."),
    ("2026-07-17", "Match forecaster and the market-skill audit",
     "A Dixon-Coles/xG/Elo/market ensemble with nested walk-forward weight learning "
     "forecasts the closing matches; the prediction market's trophy-ratio proxy loses "
     "to plain Elo out of sample and its weight is capped at 0.25."),
    ("2026-07-18", "Backup-goalkeeper autosub discount",
     "Only one of the two mandatory goalkeepers can score in a round, so the squad "
     "solvers discount the backup to autosub value: one strong starter plus the "
     "cheapest legal backup, savings spent outfield."),
)


def registry_by_key() -> dict[str, ModelRecord]:
    """Backend key to record, tabs only."""
    return {r.key: r for r in MODEL_REGISTRY}
