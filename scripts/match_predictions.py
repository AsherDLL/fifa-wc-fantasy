"""Round-8 match forecaster: winner, score, goals, corners, cards, scorers.

Research-grade rewrite. Four probabilistic components are fit and audited
under a leak-free walk-forward protocol on the tournament's completed
rounds, then combined by a linear opinion pool whose weights are EARNED
from out-of-sample skill (ranked probability score / log loss), never
hand-set:

  dixon_coles  penalized MLE bivariate-adjusted Poisson (Dixon & Coles
               1997): per-team attack/defence on the log scale, low-score
               dependence rho, exponential round decay, L2 ridge (also
               resolves identifiability). Hyperparameters (decay, ridge)
               tuned by NESTED walk-forward, never in-sample.
  xg_poisson   iterative attack/defence scoring rates on a 50/50 blend of
               goals and real match xG (external.wc2026_dataset), recency
               and shrink tuned by the same nested protocol.
  elo          rates implied by leak-free as-of-kickoff country Elo
               (recomputed per match from the martj42 history; the cached
               country_elo.csv snapshot is NOT used - it already contains
               future results and is therefore leaky in a backtest).
  market       Polymarket trophy-price ratio p(A advances) =
               price_A / (price_A + price_B), last snapshot strictly
               before kickoff. POLICY: the market is one capped signal,
               never an anchor. Its ensemble weight is learned from
               rounds 4-7 skill and hard-capped at MARKET_CAP = 0.25; the
               artifact records both the learned and the applied weight.
               No model constant is calibrated against bookmaker prices.

Two pools, because a trophy ratio carries no draw measure:
  pool_1x2      DC + xG + Elo on 90-minute H/D/A, holdout rounds 2-7.
  pool_advance  DC + xG + Elo + market on knockout advance, rounds 4-7.
Ensemble weights are reported leave-one-round-out (LORO) and fit on all
rounds for production. Correct-score/totals outputs mix only the two
grid-producing components (DC, xG).

Uncertainty: case-resampling bootstrap (B configurable) over the
completed matches, refitting DC and xG with frozen hyperparameters and
weights; 90% percentile CIs on win probabilities and expected goals.

Fixed assumptions the data cannot identify (documented, not tuned):
extra-time intensity factor ET_ETA = 0.9 (the 90-minute scorelines of
AET matches are unrecoverable from the dataset); penalty shootouts 0.5;
cards dispersion (no per-match card counts exist) - cards are Poisson
with an assumed NB var/mean = 1.15 sensitivity entry. AET/Pens matches
enter the likelihood with exposure 4/3 and a forced 90-minute Draw label.

Corners: negative binomial on match totals (dispersion by method of
moments), opponent-adjusted multiplicative means. Scorers: Gamma-Poisson
empirical Bayes per position over all 1,248 players (marginal
negative-binomial MLE), posterior scoring rates weighted by expected
minutes from recent lineups.

Usage:
    python scripts/match_predictions.py                 # predict (auto-fit)
    python scripts/match_predictions.py --fit           # re-learn everything
    python scripts/match_predictions.py --bootstrap 500
    python scripts/match_predictions.py --skip-bootstrap --match final
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fifa_fantasy.external import wc2026_dataset as wcd
from fifa_fantasy.external.international_elo import compute_elo, fetch_results
from fifa_fantasy.external.mapping import to_fifa_country
from fifa_fantasy.model.poisson import GOAL_SHARE

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data/evaluation/match_predictions_round8.json"
SKILL_OUT = REPO_ROOT / "data/evaluation/match_prediction_skill.json"
SNAP_DIR = REPO_ROOT / "data/external/prediction_markets"

SEED = 20260716
MAX_GOALS = 8
ET_ETA = 0.9            # ET intensity vs 90'; not estimable from this data
MARKET_CAP = 0.25       # hard cap on the market's applied ensemble weight
WEIGHT_STEP = 0.05      # simplex grid step for weight learning
MINUTES_WINDOW = 3
AET_EXPOSURE = 4.0 / 3.0

DC_XI_GRID = (0.6, 0.7, 0.8, 0.9, 1.0)
DC_RIDGE_GRID = (0.5, 1.0, 2.0, 5.0, 10.0)
DC_DEFAULT = (0.9, 2.0)
XG_XI_GRID = (0.6, 0.7, 0.8, 0.9, 1.0)
XG_SHRINK_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
XG_DEFAULT = (0.85, 0.6)

CORNER_LINES = (8.5, 9.5, 10.5)
CARD_LINES = (2.5, 3.5, 4.5)
CARDS_NB_SENSITIVITY_VMR = 1.15


# --------------------------------------------------------------------------
# data layer
# --------------------------------------------------------------------------

@dataclass
class Bundle:
    spine: pd.DataFrame        # + kickoff_utc
    labels: pd.DataFrame       # completed matches with outcomes/exposure
    tmap: pd.DataFrame
    elo: dict[int, tuple[float, float]]   # match_id -> (elo_h, elo_a) before
    market_ts: list[tuple[datetime, dict[str, float]]]
    team_stats: pd.DataFrame
    player_stats: pd.DataFrame
    players: pd.DataFrame
    lineups: pd.DataFrame
    referees: pd.DataFrame


def _pm_norm(name: str) -> str:
    """Accent-strip, lowercase, letters only - for Polymarket title joins."""
    d = unicodedata.normalize("NFD", str(name))
    return re.sub(r"[^a-z]", "",
                  "".join(c for c in d
                          if unicodedata.category(c) != "Mn").lower())


# FIFA squad name -> Polymarket contract name, where letter-squashing alone
# cannot bridge them.
_FIFA_TO_PM = {
    "korearepublic": "southkorea",
    "cotedivoire": "ivorycoast",
    "caboverde": "capeverde",
    "iriran": "iran",
    "bosniaandherzegovina": "bosniaherzegovina",
}


def _pm_key(fifa_country: str) -> str:
    k = _pm_norm(fifa_country)
    return _FIFA_TO_PM.get(k, k)


def build_labels(spine: pd.DataFrame) -> pd.DataFrame:
    """One row per completed match with outcome labels and exposure.

    r90: 0 home / 1 draw / 2 away in 90 minutes. The dataset's AET scores
    include extra time and the 90' scoreline is unrecoverable, so AET and
    Penalties matches are labeled Draw (exact) and enter goal likelihoods
    with exposure 4/3 (first-order rate correction; xG spans ET too).
    adv: 0 home / 1 away, knockout rounds only.
    """
    done = spine[spine["completed"]].copy()
    rt = done["result_type"].fillna("Regular")
    done["exposure"] = np.where(rt.isin(("AET", "Penalties")),
                                AET_EXPOSURE, 1.0)
    x = done["home_score_ext"].astype(float)
    y = done["away_score_ext"].astype(float)
    r90 = np.where(x > y, 0, np.where(x < y, 2, 1))
    done["r90"] = np.where(rt.isin(("AET", "Penalties")), 1, r90)

    pen_h = pd.to_numeric(done.get("home_penalty_score"), errors="coerce")
    pen_a = pd.to_numeric(done.get("away_penalty_score"), errors="coerce")
    adv = np.where(x > y, 0, np.where(x < y, 1, np.nan))
    pens = rt.eq("Penalties")
    adv = np.where(pens, np.where(pen_h > pen_a, 0, 1), adv)
    done["adv"] = np.where(done["round_id"] >= 4, adv, np.nan)
    return done


def elo_before_by_match(spine: pd.DataFrame) -> dict:
    """Leak-free as-of-kickoff Elo per WC match from the martj42 history.

    country_elo.csv is deliberately NOT used: it is a current snapshot
    (contains all WC results -> leaks into a walk-forward backtest) and
    its builder lets NaN-scored future fixtures nudge ratings. Here NaN
    scores are dropped BEFORE the Elo pass, and each match reads the
    ratings as they stood before kickoff.
    """
    hist = fetch_results(refresh=False)
    hist = hist[hist["home_score"].notna() & hist["away_score"].notna()]
    withelo = compute_elo(hist)

    wc = withelo[(withelo["date"] >= "2026-06-01")
                 & (withelo["tournament"].str.contains("FIFA World Cup",
                                                       case=False, na=False))]
    lookup: dict[tuple, tuple[str, float, str, float]] = {}
    for r in wc.itertuples(index=False):
        h = to_fifa_country(str(r.home_team))
        a = to_fifa_country(str(r.away_team))
        key = (r.date.date(), frozenset((_pm_norm(h), _pm_norm(a))))
        lookup[key] = (_pm_norm(h), float(r.home_elo_before),
                       _pm_norm(a), float(r.away_elo_before))

    latest: dict[str, float] = {}
    for r in withelo.itertuples(index=False):
        latest[_pm_norm(to_fifa_country(str(r.home_team)))] = float(r.home_elo_after)
        latest[_pm_norm(to_fifa_country(str(r.away_team)))] = float(r.away_elo_after)

    by_squad = {}
    tmap = wcd.team_id_map()
    for r in tmap.itertuples(index=False):
        by_squad[int(r.squad_id)] = _pm_norm(str(r.country))

    out: dict[int, tuple[float, float]] = {}
    missing = []
    for r in spine.itertuples(index=False):
        h_n = by_squad[int(r.home_squad_id)]
        a_n = by_squad[int(r.away_squad_id)]
        date = pd.to_datetime(r.date_ext).date()
        hit = None
        for delta in (0, 1, -1):
            key = (date + pd.Timedelta(days=delta), frozenset((h_n, a_n)))
            if key in lookup:
                hit = lookup[key]
                break
        if hit is not None:
            n1, e1, n2, e2 = hit
            out[int(r.match_id)] = (e1, e2) if n1 == h_n else (e2, e1)
        elif not bool(r.completed):
            out[int(r.match_id)] = (latest.get(h_n, 1500.0),
                                    latest.get(a_n, 1500.0))
        else:
            missing.append(int(r.match_id))
    if missing:
        # tolerate a small tail (cache slightly behind); fall back to the
        # latest rating, which for recent matches is nearly the pre-match one
        for mid in missing:
            row = spine[spine["match_id"] == mid].iloc[0]
            out[mid] = (latest.get(by_squad[int(row["home_squad_id"])], 1500.0),
                        latest.get(by_squad[int(row["away_squad_id"])], 1500.0))
        print(f"elo: {len(missing)} matches missing from the history cache; "
              f"used latest ratings for match_ids {missing[:6]}")
    return out


_SNAP_TS = re.compile(r"snapshot_(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})Z")
_TROPHY = re.compile(r"^Will (.+) win the 2026 FIFA World Cup\?$")


def market_series() -> list[tuple[datetime, dict[str, float]]]:
    """Chronological [(ts_utc, {pm_norm_name: price>0})] trophy series."""
    series = []
    for path in sorted(SNAP_DIR.glob("snapshot_*.jsonl")):
        m = _SNAP_TS.search(path.name)
        if not m:
            continue
        ts = datetime.fromisoformat(
            f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}+00:00")
        prices = {}
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = _TROPHY.match(d.get("title") or "")
            if t and d.get("yes_price"):
                prices[_pm_norm(t.group(1))] = float(d["yes_price"])
        if prices:
            series.append((ts, prices))
    return series


def market_advance(home_country: str, away_country: str,
                   kickoff_utc: datetime,
                   series: list[tuple[datetime, dict[str, float]]]
                   ) -> float | None:
    """Clipped trophy-ratio p(home advances) from the last pre-kickoff
    snapshot; None when either side is unpriced (e.g. the bronze match)."""
    h_k, a_k = _pm_key(home_country), _pm_key(away_country)
    for ts, prices in reversed(series):
        if ts >= kickoff_utc:
            continue
        h, a = prices.get(h_k), prices.get(a_k)
        if h and a:
            return float(np.clip(h / (h + a), 0.01, 0.99))
        return None
    return None


def load_bundle() -> Bundle:
    spine = wcd.match_spine()
    fixtures = wcd._latest(wcd.DEFAULT_RAW_DIR, "fixtures")
    spine = spine.merge(fixtures[["fixture_id", "kickoff"]],
                        on="fixture_id", how="left")
    spine["kickoff_utc"] = pd.to_datetime(spine["kickoff"], utc=True)

    n_done = int(spine["completed"].sum())
    assert n_done >= 102, f"expected >=102 completed matches, got {n_done}"
    assert (~spine["completed"]).sum() == 2, "expected exactly 2 upcoming"
    labels = build_labels(spine)
    n_special = int((labels["exposure"] > 1).sum())
    assert n_special == 8, f"expected 8 AET/Pens matches, got {n_special}"

    return Bundle(
        spine=spine,
        labels=labels,
        tmap=wcd.team_id_map(),
        elo=elo_before_by_match(spine),
        market_ts=market_series(),
        team_stats=wcd.load("match_team_stats"),
        player_stats=wcd.load("player_stats"),
        players=wcd.load("squads_and_players"),
        lineups=wcd.load("match_lineups"),
        referees=wcd.load("referees"),
    )


# --------------------------------------------------------------------------
# shared prediction machinery
# --------------------------------------------------------------------------

def _dc_tau_grid(grid: np.ndarray, lam: float, mu: float,
                 rho: float) -> np.ndarray:
    g = grid.copy()
    g[0, 0] *= max(1 - lam * mu * rho, 1e-10)
    g[0, 1] *= max(1 + lam * rho, 1e-10)
    g[1, 0] *= max(1 + mu * rho, 1e-10)
    g[1, 1] *= max(1 - rho, 1e-10)
    return g / g.sum()


def score_grid(lam_h: float, lam_a: float, rho: float = 0.0,
               max_goals: int = MAX_GOALS) -> np.ndarray:
    k = np.arange(max_goals + 1)
    grid = np.outer(stats.poisson.pmf(k, lam_h), stats.poisson.pmf(k, lam_a))
    grid /= grid.sum()
    if rho:
        grid = _dc_tau_grid(grid, lam_h, lam_a, rho)
    return grid


def grid_1x2(grid: np.ndarray) -> np.ndarray:
    p = np.array([np.tril(grid, -1).sum(), np.trace(grid),
                  np.triu(grid, 1).sum()])
    return p / p.sum()


def p_advance_from(lam_h: float, lam_a: float, rho: float = 0.0) -> float:
    """P(home advances): 90' win + draw -> ET (reduced rate) -> pens 0.5."""
    p90 = grid_1x2(score_grid(lam_h, lam_a, rho))
    et = grid_1x2(score_grid(lam_h * ET_ETA / 3.0, lam_a * ET_ETA / 3.0,
                             rho, max_goals=5))
    return float(p90[0] + p90[1] * (et[0] + 0.5 * et[1]))


# --------------------------------------------------------------------------
# component: Dixon-Coles penalized MLE
# --------------------------------------------------------------------------

@dataclass
class DCFit:
    c: float
    rho: float
    att: dict[int, float]
    deff: dict[int, float]
    xi: float
    ridge: float

    def lambdas(self, home_sq: int, away_sq: int) -> tuple[float, float]:
        lam = math.exp(self.c + self.att.get(home_sq, 0.0)
                       - self.deff.get(away_sq, 0.0))
        mu = math.exp(self.c + self.att.get(away_sq, 0.0)
                      - self.deff.get(home_sq, 0.0))
        return lam, mu


def dc_nll_grad(theta: np.ndarray, h_idx: np.ndarray, a_idx: np.ndarray,
                x: np.ndarray, y: np.ndarray, t: np.ndarray, w: np.ndarray,
                n_teams: int, ridge: float) -> tuple[float, np.ndarray]:
    c, rho = theta[0], theta[1]
    att = theta[2:2 + n_teams]
    deff = theta[2 + n_teams:]
    lam = t * np.exp(c + att[h_idx] - deff[a_idx])
    mu = t * np.exp(c + att[a_idx] - deff[h_idx])

    nll = -np.sum(w * (x * np.log(lam) - lam + y * np.log(mu) - mu))
    grad = np.zeros_like(theta)
    rx = w * (x - lam)     # d(loglik)/d(log lam) per match
    ry = w * (y - mu)
    grad[0] = -(rx.sum() + ry.sum())
    g_att = np.zeros(n_teams)
    g_def = np.zeros(n_teams)
    np.add.at(g_att, h_idx, -rx)
    np.add.at(g_att, a_idx, -ry)
    np.add.at(g_def, a_idx, rx)
    np.add.at(g_def, h_idx, ry)

    # tau low-score correction
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau = np.ones_like(lam)
    tau[m00] = np.maximum(1 - lam[m00] * mu[m00] * rho, 1e-10)
    tau[m01] = np.maximum(1 + lam[m01] * rho, 1e-10)
    tau[m10] = np.maximum(1 + mu[m10] * rho, 1e-10)
    tau[m11] = np.maximum(1 - rho, 1e-10)
    nll -= np.sum(w * np.log(tau))

    # d(log tau)/d(log lam), d/d(log mu), d/d(rho)
    dl = np.zeros_like(lam)
    dm = np.zeros_like(lam)
    dr = np.zeros_like(lam)
    lm = lam[m00] * mu[m00]
    dl[m00] = -rho * lm / tau[m00]
    dm[m00] = -rho * lm / tau[m00]
    dr[m00] = -lm / tau[m00]
    dl[m01] = rho * lam[m01] / tau[m01]
    dr[m01] = lam[m01] / tau[m01]
    dm[m10] = rho * mu[m10] / tau[m10]
    dr[m10] = mu[m10] / tau[m10]
    dr[m11] = -1.0 / tau[m11]

    grad[0] -= np.sum(w * (dl + dm))
    grad[1] = -np.sum(w * dr)
    np.add.at(g_att, h_idx, -w * dl)
    np.add.at(g_att, a_idx, -w * dm)
    np.add.at(g_def, a_idx, w * dl)
    np.add.at(g_def, h_idx, w * dm)

    nll += ridge * (att @ att + deff @ deff)
    g_att += 2 * ridge * att
    g_def += 2 * ridge * deff
    grad[2:2 + n_teams] = g_att
    grad[2 + n_teams:] = g_def
    return float(nll), grad


def fit_dc(train: pd.DataFrame, xi: float, ridge: float) -> DCFit:
    teams = sorted(set(train["home_squad_id"]) | set(train["away_squad_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h_idx = train["home_squad_id"].map(idx).to_numpy()
    a_idx = train["away_squad_id"].map(idx).to_numpy()
    x = train["home_score_ext"].to_numpy(dtype=float)
    y = train["away_score_ext"].to_numpy(dtype=float)
    t = train["exposure"].to_numpy(dtype=float)
    r_max = int(train["round_id"].max())
    w = xi ** (r_max - train["round_id"].to_numpy(dtype=float))

    theta0 = np.zeros(2 + 2 * n)
    theta0[0] = math.log(max(float(np.average(
        np.concatenate([x / t, y / t]), weights=np.concatenate([w, w]))),
        0.2))
    bounds = ([(-3, 3), (-0.2, 0.2)] + [(-3, 3)] * (2 * n))
    res = optimize.minimize(
        dc_nll_grad, theta0, args=(h_idx, a_idx, x, y, t, w, n, ridge),
        jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500})
    theta = res.x
    return DCFit(
        c=float(theta[0]), rho=float(theta[1]),
        att={t_: float(theta[2 + i]) for t_, i in idx.items()},
        deff={t_: float(theta[2 + n + i]) for t_, i in idx.items()},
        xi=xi, ridge=ridge)


# --------------------------------------------------------------------------
# component: xG-Poisson iterative rates
# --------------------------------------------------------------------------

@dataclass
class XGFit:
    att: dict[int, float]
    deff: dict[int, float]
    mu: float
    shrink: float
    xi: float

    def lambdas(self, home_sq: int, away_sq: int) -> tuple[float, float]:
        lam = self.mu * (self.att.get(home_sq, 1.0)
                         * self.deff.get(away_sq, 1.0)) ** self.shrink
        mu_ = self.mu * (self.att.get(away_sq, 1.0)
                         * self.deff.get(home_sq, 1.0)) ** self.shrink
        return lam, mu_


def fit_xg(train: pd.DataFrame, xi: float, shrink: float,
           n_iter: int = 10) -> XGFit:
    d = train.copy()
    r_max = int(d["round_id"].max())
    d["w"] = xi ** (r_max - d["round_id"])
    has_xg = d["home_xg"].notna() & d["away_xg"].notna()
    d["g_home"] = np.where(
        has_xg, 0.5 * d["home_score_ext"] + 0.5 * d["home_xg"],
        d["home_score_ext"]) / d["exposure"]
    d["g_away"] = np.where(
        has_xg, 0.5 * d["away_score_ext"] + 0.5 * d["away_xg"],
        d["away_score_ext"]) / d["exposure"]
    long = pd.concat([
        d.rename(columns={"home_squad_id": "team", "away_squad_id": "opp",
                          "g_home": "gf", "g_away": "ga"})[
            ["team", "opp", "gf", "ga", "w"]],
        d.rename(columns={"away_squad_id": "team", "home_squad_id": "opp",
                          "g_away": "gf", "g_home": "ga"})[
            ["team", "opp", "gf", "ga", "w"]],
    ], ignore_index=True)
    mu = float((long["gf"] * long["w"]).sum() / long["w"].sum())
    teams = sorted(set(long["team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    ti = long["team"].map(idx).to_numpy()
    oi = long["opp"].map(idx).to_numpy()
    gf = long["gf"].to_numpy(dtype=float)
    ga = long["ga"].to_numpy(dtype=float)
    w = long["w"].to_numpy(dtype=float)
    att = np.ones(n)
    deff = np.ones(n)
    num_att = np.bincount(ti, weights=w * gf, minlength=n)
    num_def = np.bincount(ti, weights=w * ga, minlength=n)
    for _ in range(n_iter):
        att = num_att / np.maximum(
            np.bincount(ti, weights=w * mu * deff[oi], minlength=n), 1e-9)
        deff = num_def / np.maximum(
            np.bincount(ti, weights=w * mu * att[oi], minlength=n), 1e-9)
        att /= att.mean()
        deff /= deff.mean()
    return XGFit(att={t: float(att[i]) for t, i in idx.items()},
                 deff={t: float(deff[i]) for t, i in idx.items()},
                 mu=mu, shrink=shrink, xi=xi)


# --------------------------------------------------------------------------
# component: Elo-implied rates
# --------------------------------------------------------------------------

def elo_lambdas(elo_h: float, elo_a: float, mu: float) -> tuple[float, float]:
    diff = float(np.clip((elo_h - elo_a) / 400.0, -1.2, 1.2))
    return mu * math.exp(diff), mu * math.exp(-diff)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def rps(p: np.ndarray, outcome: int) -> float:
    """Ranked probability score for ordered H/D/A (lower is better)."""
    obs = np.zeros(3)
    obs[outcome] = 1.0
    cf, co = np.cumsum(p), np.cumsum(obs)
    return float(np.sum((cf[:2] - co[:2]) ** 2) / 2)


def log_loss_multi(p: np.ndarray, outcome: int) -> float:
    return float(-np.log(max(p[outcome], 1e-12)))


def brier_multi(p: np.ndarray, outcome: int) -> float:
    obs = np.zeros(len(p))
    obs[outcome] = 1.0
    return float(np.sum((p - obs) ** 2))


def log_loss_bin(p_home: float, adv: int) -> float:
    p = np.clip(p_home, 1e-12, 1 - 1e-12)
    return float(-math.log(p if adv == 0 else 1 - p))


def brier_bin(p_home: float, adv: int) -> float:
    return float((p_home - (1.0 if adv == 0 else 0.0)) ** 2)


# --------------------------------------------------------------------------
# walk-forward engine
# --------------------------------------------------------------------------

RATE_COMPONENTS = ("dixon_coles", "xg_poisson", "elo")
ADV_COMPONENTS = RATE_COMPONENTS + ("market",)


def _mu_tournament(train: pd.DataFrame) -> float:
    g = pd.concat([train["home_score_ext"] / train["exposure"],
                   train["away_score_ext"] / train["exposure"]])
    return float(g.mean())


def _predict_components(train: pd.DataFrame, matches: pd.DataFrame,
                        bundle: Bundle, dc_hp: tuple, xg_hp: tuple) -> list[dict]:
    """Fit on `train`, predict each row of `matches`. Returns row dicts."""
    dc = fit_dc(train, *dc_hp)
    xg = fit_xg(train, *xg_hp)
    mu_t = _mu_tournament(train)
    rows = []
    for r in matches.itertuples(index=False):
        h, a = int(r.home_squad_id), int(r.away_squad_id)
        lam = {
            "dixon_coles": dc.lambdas(h, a),
            "xg_poisson": xg.lambdas(h, a),
            "elo": elo_lambdas(*bundle.elo[int(r.match_id)], mu_t),
        }
        row = {"match_id": int(r.match_id), "round_id": int(r.round_id)}
        for name, (lh, la) in lam.items():
            rho = dc.rho if name == "dixon_coles" else 0.0
            row[f"{name}_lam"] = (lh, la)
            row[f"{name}_1x2"] = grid_1x2(score_grid(lh, la, rho))
            row[f"{name}_adv"] = p_advance_from(lh, la, rho)
        row["dc_rho"] = dc.rho
        rows.append(row)
    return rows


def _inner_select(labels: pd.DataFrame, k: int, component: str,
                  bundle: Bundle) -> tuple:
    """Nested hyperparameter choice for holdout round k: inner walk-forward
    over rounds 2..k-1, each trained on rounds < j only. RPS objective."""
    inner_rounds = [j for j in range(2, k)]
    if not inner_rounds:
        return DC_DEFAULT if component == "dixon_coles" else XG_DEFAULT
    grid = (list(product(DC_XI_GRID, DC_RIDGE_GRID))
            if component == "dixon_coles"
            else list(product(XG_XI_GRID, XG_SHRINK_GRID)))
    best, best_score = None, np.inf
    for hp in grid:
        scores = []
        for j in inner_rounds:
            train = labels[labels["round_id"] < j]
            hold = labels[labels["round_id"] == j]
            if component == "dixon_coles":
                fit = fit_dc(train, *hp)
            else:
                fit = fit_xg(train, *hp)
            rho = fit.rho if component == "dixon_coles" else 0.0
            for r in hold.itertuples(index=False):
                lh, la = fit.lambdas(int(r.home_squad_id),
                                     int(r.away_squad_id))
                scores.append(rps(grid_1x2(score_grid(lh, la, rho)),
                                  int(r.r90)))
        m = float(np.mean(scores))
        if m < best_score - 1e-12:
            best, best_score = hp, m
    return best


def run_walkforward(bundle: Bundle) -> tuple[list[dict], dict]:
    labels = bundle.labels
    rounds = sorted(int(r) for r in labels["round_id"].unique())
    holdouts = [k for k in rounds if k >= 2]
    preds: list[dict] = []
    chosen: dict[int, dict] = {}
    by_country = bundle.tmap.set_index("squad_id")["country"]
    for k in holdouts:
        train = labels[labels["round_id"] < k]
        hold = labels[labels["round_id"] == k]
        dc_hp = _inner_select(labels, k, "dixon_coles", bundle)
        xg_hp = _inner_select(labels, k, "xg_poisson", bundle)
        chosen[k] = {"dc": dc_hp, "xg": xg_hp}
        rows = _predict_components(train, hold, bundle, dc_hp, xg_hp)
        hold_ix = hold.set_index("match_id")
        for row in rows:
            lab = hold_ix.loc[row["match_id"]]
            row["r90"] = int(lab["r90"])
            row["adv"] = None if pd.isna(lab["adv"]) else int(lab["adv"])
            if row["round_id"] >= 4:
                row["market_adv"] = market_advance(
                    str(by_country[int(lab["home_squad_id"])]),
                    str(by_country[int(lab["away_squad_id"])]),
                    lab["kickoff_utc"], bundle.market_ts)
            else:
                row["market_adv"] = None
        preds.extend(rows)
        print(f"walk-forward round {k}: n={len(rows)} dc_hp={dc_hp} "
              f"xg_hp={xg_hp}")
    return preds, chosen


# --------------------------------------------------------------------------
# ensemble weight learning
# --------------------------------------------------------------------------

def simplex_grid(n: int, step: float = WEIGHT_STEP) -> np.ndarray:
    ticks = int(round(1.0 / step))
    combos = []
    def rec(prefix, remaining, slots):
        if slots == 1:
            combos.append(prefix + [remaining])
            return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v, slots - 1)
    rec([], ticks, n)
    return np.array(combos, dtype=float) * step


def _pool_1x2(row: dict, w: np.ndarray, log_pool: bool) -> np.ndarray:
    ps = [np.asarray(row[f"{c}_1x2"]) for c in RATE_COMPONENTS]
    if log_pool:
        logp = sum(wi * np.log(np.clip(p, 1e-12, 1)) for wi, p in zip(w, ps))
        p = np.exp(logp - logp.max())
        return p / p.sum()
    p = sum(wi * p_ for wi, p_ in zip(w, ps))
    return p / p.sum()


def _pool_adv(row: dict, w: np.ndarray, log_pool: bool) -> float:
    comps, weights = [], []
    for c, wi in zip(ADV_COMPONENTS, w):
        v = row["market_adv"] if c == "market" else row[f"{c}_adv"]
        if v is not None:
            comps.append(float(v))
            weights.append(wi)
    weights = np.array(weights)
    if weights.sum() <= 0:
        return float(np.mean(comps))
    weights = weights / weights.sum()
    if log_pool:
        logit = sum(wi * (math.log(p / (1 - p)))
                    for wi, p in zip(weights, comps))
        return 1.0 / (1.0 + math.exp(-logit))
    return float(np.dot(weights, comps))


def fit_weights(preds: list[dict], pool: str, log_pool: bool,
                cap_market: float | None = None,
                exclude_round: int | None = None) -> tuple[np.ndarray, float]:
    if pool == "1x2":
        rows = [r for r in preds if exclude_round is None
                or r["round_id"] != exclude_round]
        grid = simplex_grid(len(RATE_COMPONENTS))
        best_w, best = None, np.inf
        for w in grid:
            s = float(np.mean([rps(_pool_1x2(r, w, log_pool), r["r90"])
                               for r in rows]))
            if s < best - 1e-12:
                best_w, best = w, s
        return best_w, best
    rows = [r for r in preds
            if r["adv"] is not None
            and (exclude_round is None or r["round_id"] != exclude_round)]
    grid = simplex_grid(len(ADV_COMPONENTS))
    if cap_market is not None:
        grid = grid[grid[:, ADV_COMPONENTS.index("market")]
                    <= cap_market + 1e-9]
    best_w, best = None, np.inf
    for w in grid:
        s = float(np.mean([log_loss_bin(_pool_adv(r, w, log_pool), r["adv"])
                           for r in rows]))
        if s < best - 1e-12:
            best_w, best = w, s
    return best_w, best


def fit_grid_pool(preds: list[dict]) -> tuple[float, float]:
    """Weight for the score-grid pool, DC vs xG only (the two components
    that produce a score distribution). Fit on 1X2 RPS like the main pool;
    kept separate because the main pool may put all weight on Elo, which
    has no team-specific score information."""
    best_w, best = 0.5, np.inf
    for w in np.arange(0.0, 1.0 + 1e-9, WEIGHT_STEP):
        s = float(np.mean([
            rps(w * np.asarray(r["dixon_coles_1x2"])
                + (1 - w) * np.asarray(r["xg_poisson_1x2"]), r["r90"])
            for r in preds]))
        if s < best - 1e-12:
            best_w, best = float(w), s
    return best_w, best


def loro_skill(preds: list[dict], pool: str, log_pool: bool,
               cap_market: float | None) -> dict:
    """Leave-one-round-out ensemble skill: weights fit excluding round k,
    scored on round k."""
    rounds = sorted({r["round_id"] for r in preds
                     if pool == "1x2" or r["adv"] is not None})
    per_round, all_scores = {}, []
    for k in rounds:
        w, _ = fit_weights(preds, pool, log_pool, cap_market,
                           exclude_round=k)
        rows = [r for r in preds if r["round_id"] == k
                and (pool == "1x2" or r["adv"] is not None)]
        if pool == "1x2":
            scores = [rps(_pool_1x2(r, w, log_pool), r["r90"]) for r in rows]
        else:
            scores = [log_loss_bin(_pool_adv(r, w, log_pool), r["adv"])
                      for r in rows]
        per_round[k] = round(float(np.mean(scores)), 4)
        all_scores.extend(scores)
    return {"per_round": per_round,
            "pooled": round(float(np.mean(all_scores)), 4)}


def component_skill(preds: list[dict]) -> dict:
    out: dict = {"skill_1x2": {}, "skill_advance": {}}
    for c in RATE_COMPONENTS:
        p1 = [(r[f"{c}_1x2"], r["r90"]) for r in preds]
        out["skill_1x2"][c] = {
            "rps": round(float(np.mean([rps(p, o) for p, o in p1])), 4),
            "log_loss": round(float(np.mean(
                [log_loss_multi(p, o) for p, o in p1])), 4),
            "brier": round(float(np.mean(
                [brier_multi(p, o) for p, o in p1])), 4),
            "n": len(p1),
        }
    adv_rows = [r for r in preds if r["adv"] is not None]
    for c in ADV_COMPONENTS:
        vals = [(r["market_adv"] if c == "market" else r[f"{c}_adv"],
                 r["adv"]) for r in adv_rows]
        vals = [(p, o) for p, o in vals if p is not None]
        out["skill_advance"][c] = {
            "log_loss": round(float(np.mean(
                [log_loss_bin(p, o) for p, o in vals])), 4),
            "brier": round(float(np.mean(
                [brier_bin(p, o) for p, o in vals])), 4),
            "n": len(vals),
        }
    uniform = np.array([1 / 3, 1 / 3, 1 / 3])
    out["baselines"] = {
        "uniform_rps": round(float(np.mean(
            [rps(uniform, r["r90"]) for r in preds])), 4),
    }
    return out


def calibration_block(preds: list[dict], w_1x2: np.ndarray,
                      log_pool: bool, rng: np.random.Generator) -> dict:
    """Randomized PIT on goals under the DC+xG mixture grid + sharpness."""
    w_dc, w_xg = w_1x2[0], w_1x2[1]
    tot = max(w_dc + w_xg, 1e-9)
    w_dc, w_xg = w_dc / tot, w_xg / tot
    pits_h, pits_a, pits_t = [], [], []
    for r in preds:
        g = (w_dc * score_grid(*r["dixon_coles_lam"], r["dc_rho"])
             + w_xg * score_grid(*r["xg_poisson_lam"]))
        g /= g.sum()
        # PIT uses the recorded scores; AET matches carry ET goals, a small
        # known contamination affecting 8/78 rows (flagged in the artifact)
        ph = g.sum(axis=1)
        pa = g.sum(axis=0)
        n = g.shape[0]
        pt = np.zeros(2 * n - 1)
        for i in range(n):
            pt[i:i + n] += g[i, :]
        pits_h.append(_rand_pit(ph, r["x_obs"], rng))
        pits_a.append(_rand_pit(pa, r["y_obs"], rng))
        pits_t.append(_rand_pit(pt, r["x_obs"] + r["y_obs"], rng))
    def _hist(p):
        h, _ = np.histogram(p, bins=10, range=(0, 1))
        chi = stats.chisquare(h)
        return {"bins": h.tolist(), "chi2_p": round(float(chi.pvalue), 3)}
    ent = [float(-np.sum(_pool_1x2(r, w_1x2, log_pool)
                         * np.log(np.clip(_pool_1x2(r, w_1x2, log_pool),
                                          1e-12, 1)))) for r in preds]
    mx = [float(np.max(_pool_1x2(r, w_1x2, log_pool))) for r in preds]
    # 3-bin reliability on ensemble max-prob
    bins = [(1 / 3, 0.5), (0.5, 0.65), (0.65, 1.01)]
    rel = []
    for lo, hi in bins:
        rows = [r for r, m in zip(preds, mx) if lo <= m < hi]
        if not rows:
            rel.append({"bin": [lo, hi], "n": 0})
            continue
        hits = [int(np.argmax(_pool_1x2(r, w_1x2, log_pool)) == r["r90"])
                for r in rows]
        conf = [float(np.max(_pool_1x2(r, w_1x2, log_pool))) for r in rows]
        rel.append({"bin": [round(lo, 2), round(hi, 2)], "n": len(rows),
                    "mean_conf": round(float(np.mean(conf)), 3),
                    "accuracy": round(float(np.mean(hits)), 3)})
    return {
        "pit_home_goals": _hist(pits_h),
        "pit_away_goals": _hist(pits_a),
        "pit_total_goals": _hist(pits_t),
        "pit_note": "AET matches contribute ET-inclusive scores (8/78 rows)",
        "sharpness": {"mean_entropy": round(float(np.mean(ent)), 3),
                      "mean_max_prob": round(float(np.mean(mx)), 3)},
        "reliability_3bin": rel,
    }


def _rand_pit(pmf: np.ndarray, obs: int, rng: np.random.Generator) -> float:
    obs = int(min(obs, len(pmf) - 1))
    below = float(pmf[:obs].sum())
    return below + float(rng.uniform()) * float(pmf[obs])


# --------------------------------------------------------------------------
# fit: learn everything, write the skill artifact
# --------------------------------------------------------------------------

def run_fit(bundle: Bundle) -> dict:
    rng = np.random.default_rng(SEED)
    preds, chosen = run_walkforward(bundle)
    lab_ix = bundle.labels.set_index("match_id")
    for r in preds:
        r["x_obs"] = int(lab_ix.loc[r["match_id"], "home_score_ext"])
        r["y_obs"] = int(lab_ix.loc[r["match_id"], "away_score_ext"])

    skill = component_skill(preds)

    pools = {}
    for pool in ("1x2", "advance"):
        cap = MARKET_CAP if pool == "advance" else None
        _, lin_s = fit_weights(preds, pool, log_pool=False, cap_market=cap)
        _, log_s = fit_weights(preds, pool, log_pool=True, cap_market=cap)
        loro_lin = loro_skill(preds, pool, False, cap)
        loro_log = loro_skill(preds, pool, True, cap)
        # log pool goes to production only if it wins BOTH in-sample and LORO
        use_log = loro_log["pooled"] < loro_lin["pooled"] and log_s < lin_s
        learned_w, _ = fit_weights(preds, pool, use_log, cap_market=None)
        applied_w, applied_s = fit_weights(preds, pool, use_log,
                                           cap_market=cap)
        comp_names = RATE_COMPONENTS if pool == "1x2" else ADV_COMPONENTS
        pools[f"pool_{pool}"] = {
            "components": list(comp_names),
            "learned": dict(zip(comp_names, np.round(learned_w, 2))),
            "applied": dict(zip(comp_names, np.round(applied_w, 2))),
            "cap": cap,
            "pool_type": "log" if use_log else "linear",
            "in_sample_score": round(applied_s, 4),
            "loro": loro_lin if not use_log else loro_log,
            "loro_linear": loro_lin, "loro_log": loro_log,
        }

    w_grid, s_grid = fit_grid_pool(preds)
    pools["pool_grid"] = {
        "components": ["dixon_coles", "xg_poisson"],
        "applied": {"dixon_coles": round(w_grid, 2),
                    "xg_poisson": round(1 - w_grid, 2)},
        "rps": round(s_grid, 4),
        "note": "score-distribution pool; separate fit because the 1X2 "
                "pool may exclude the grid components entirely",
    }

    # production hyperparams: nested selection over ALL rounds (k = max+1)
    max_round = int(bundle.labels["round_id"].max())
    dc_hp = _inner_select(bundle.labels, max_round + 1, "dixon_coles", bundle)
    xg_hp = _inner_select(bundle.labels, max_round + 1, "xg_poisson", bundle)

    calib = calibration_block(
        preds, np.array([w_grid, 1 - w_grid, 0.0]),
        pools["pool_1x2"]["pool_type"] == "log", rng)

    per_round = {}
    for k in sorted({r["round_id"] for r in preds}):
        rows = [r for r in preds if r["round_id"] == k]
        per_round[k] = {
            c: round(float(np.mean([rps(np.asarray(r[f"{c}_1x2"]), r["r90"])
                                    for r in rows])), 4)
            for c in RATE_COMPONENTS}
        per_round[k]["n"] = len(rows)

    eval_rows = [{
        "round_id": r["round_id"], "r90": r["r90"], "adv": r["adv"],
        "market_adv": r["market_adv"],
        **{f"{c}_1x2": [round(float(v), 4) for v in r[f"{c}_1x2"]]
           for c in RATE_COMPONENTS},
        **{f"{c}_adv": round(float(r[f"{c}_adv"]), 4)
           for c in RATE_COMPONENTS},
    } for r in preds]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED,
        "n_completed": int(bundle.labels.shape[0]),
        "eval_rows": eval_rows,
        "hyperparams": {
            "production": {"dc": {"xi": dc_hp[0], "ridge": dc_hp[1]},
                           "xg": {"recency": xg_hp[0], "shrink": xg_hp[1]}},
            "per_round_selected": {str(k): {"dc": list(v["dc"]),
                                            "xg": list(v["xg"])}
                                   for k, v in chosen.items()},
        },
        "weights": pools,
        **skill,
        "skill_1x2_per_round": per_round,
        "calibration": calib,
        "notes": [
            "market = capped skill-audited component, never an anchor",
            f"market cap {MARKET_CAP}; learned vs applied recorded",
            "ET eta 0.9 and pens 0.5 are fixed, not estimable",
            "cards dispersion assumed (no per-match card data)",
        ],
    }
    SKILL_OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nskill artifact: {SKILL_OUT}")
    return payload


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------

def _weights_bootstrap(skill: dict, round8_rows: dict[int, dict],
                       B: int, rng: np.random.Generator) -> dict:
    """CI on the headline probabilities from WEIGHT-selection uncertainty:
    resample the walk-forward evaluation rows, refit the (capped) advance
    weights per replicate, re-pool the frozen round-8 component predictions.
    This is the dominant uncertainty when the learned weights are near a
    simplex corner. Fully vectorized."""
    rows = [r for r in skill["eval_rows"] if r["adv"] is not None]
    def _val(r, c):
        v = r["market_adv"] if c == "market" else r[f"{c}_adv"]
        return np.nan if v is None else float(v)
    A = np.array([[_val(r, c) for c in ADV_COMPONENTS] for r in rows])
    has_mkt = ~np.isnan(A[:, ADV_COMPONENTS.index("market")])
    o = np.array([r["adv"] for r in rows], dtype=float)  # 0 home / 1 away

    W = simplex_grid(len(ADV_COMPONENTS))
    W = W[W[:, ADV_COMPONENTS.index("market")] <= MARKET_CAP + 1e-9]
    W3 = W[:, :3] / np.maximum(W[:, :3].sum(axis=1, keepdims=True), 1e-9)

    draws: dict[int, list[float]] = {mid: [] for mid in round8_rows}
    n = len(rows)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        Ab, ob, hb = A[idx], o[idx], has_mkt[idx]
        # pooled p(home advances) per combo x row
        P = np.where(
            hb[None, :],
            W @ np.nan_to_num(Ab).T,
            W3 @ Ab[:, :3].T)
        P = np.clip(P, 1e-9, 1 - 1e-9)
        ll = -(np.log(P) * (ob == 0)[None, :]
               + np.log(1 - P) * (ob == 1)[None, :]).mean(axis=1)
        w = W[int(np.argmin(ll))]
        for mid, row in round8_rows.items():
            draws[mid].append(_pool_adv(row, w, log_pool=False))
    return {mid: [round(float(np.percentile(v, q)), 3) for q in (5, 95)]
            for mid, v in draws.items()}


def _lambda_bootstrap(bundle: Bundle, skill: dict, upcoming: pd.DataFrame,
                      B: int, rng: np.random.Generator) -> dict:
    """CI on expected goals from goal-model estimation uncertainty:
    case-resample the completed matches, refit DC and xG with frozen
    hyperparameters, mix with the grid-pool weights."""
    hp = skill["hyperparams"]["production"]
    dc_hp = (hp["dc"]["xi"], hp["dc"]["ridge"])
    xg_hp = (hp["xg"]["recency"], hp["xg"]["shrink"])
    wg = skill["weights"]["pool_grid"]["applied"]["dixon_coles"]
    labels = bundle.labels

    draws: dict[int, dict[str, list]] = {
        int(r.match_id): {"lam_h": [], "lam_a": []}
        for r in upcoming.itertuples(index=False)}
    for _ in range(B):
        sample = labels.sample(n=len(labels), replace=True,
                               random_state=int(rng.integers(2**31)))
        dc = fit_dc(sample, *dc_hp)
        xg = fit_xg(sample, *xg_hp)
        for r in upcoming.itertuples(index=False):
            h, a = int(r.home_squad_id), int(r.away_squad_id)
            d_l, x_l = dc.lambdas(h, a), xg.lambdas(h, a)
            draws[int(r.match_id)]["lam_h"].append(
                wg * d_l[0] + (1 - wg) * x_l[0])
            draws[int(r.match_id)]["lam_a"].append(
                wg * d_l[1] + (1 - wg) * x_l[1])
    return {mid: {
        "lambda_home_ci90": [round(float(np.percentile(d["lam_h"], q)), 2)
                             for q in (5, 95)],
        "lambda_away_ci90": [round(float(np.percentile(d["lam_a"], q)), 2)
                             for q in (5, 95)],
    } for mid, d in draws.items()}


# --------------------------------------------------------------------------
# corners, cards, scorers
# --------------------------------------------------------------------------

def expected_corners(bundle: Bundle, home_sq: int, away_sq: int) -> dict:
    ts = bundle.team_stats.merge(
        bundle.tmap[["team_id", "squad_id"]], on="team_id", how="left")
    done = bundle.spine[bundle.spine["completed"]][
        ["match_id", "home_squad_id", "away_squad_id"]]
    ts = ts.merge(done, on="match_id")
    ts["opp_squad_id"] = np.where(ts["squad_id"] == ts["home_squad_id"],
                                  ts["away_squad_id"], ts["home_squad_id"])
    mean_c = float(ts["corners"].mean())
    cf = ts.groupby("squad_id")["corners"].mean()
    ca = ts.groupby("opp_squad_id")["corners"].mean()
    e_h = float(cf.get(home_sq, mean_c) * ca.get(away_sq, mean_c) / mean_c)
    e_a = float(cf.get(away_sq, mean_c) * ca.get(home_sq, mean_c) / mean_c)
    total = e_h + e_a

    match_tot = ts.groupby("match_id")["corners"].sum()
    m, v = float(match_tot.mean()), float(match_tot.var(ddof=1))
    overs = {}
    if v > m:
        r_disp = m * m / (v - m)
        p_nb = r_disp / (r_disp + total)
        for line in CORNER_LINES:
            overs[str(line)] = round(float(
                1 - stats.nbinom.cdf(math.floor(line), r_disp, p_nb)), 3)
        dist = {"family": "nbinom", "vmr": round(v / m, 2)}
    else:
        for line in CORNER_LINES:
            overs[str(line)] = round(float(
                1 - stats.poisson.cdf(math.floor(line), total)), 3)
        dist = {"family": "poisson", "vmr": 1.0}
    return {"home": round(e_h, 1), "away": round(e_a, 1),
            "total": round(total, 1), "p_over": overs, **dist}


def expected_cards(bundle: Bundle, match_row: pd.Series) -> dict:
    ps = bundle.player_stats.merge(
        bundle.tmap[["team_id", "squad_id"]], on="team_id", how="left")
    team_cards = ps.groupby("squad_id").apply(
        lambda g: float(g["yellow_cards"].sum() + 2 * g["red_cards"].sum()),
        include_groups=False)
    done = bundle.labels
    played = pd.concat([done["home_squad_id"],
                        done["away_squad_id"]]).value_counts()
    rate = (team_cards / played).dropna()

    factor, ref_name = 1.0, None
    ref_id = match_row.get("referee_id")
    refs = bundle.referees
    if pd.notna(ref_id) and len(refs):
        row = refs[refs["referee_id"] == int(ref_id)]
        if len(row):
            ref_name = str(row.iloc[0]["name"])
            factor = float(row.iloc[0]["avg_cards_per_game"]
                           / refs["avg_cards_per_game"].mean())
    h = int(match_row["home_squad_id"])
    a = int(match_row["away_squad_id"])
    e_h = float(rate.get(h, rate.mean())) * factor
    e_a = float(rate.get(a, rate.mean())) * factor
    total = e_h + e_a
    overs_pois, overs_nb = {}, {}
    vmr = CARDS_NB_SENSITIVITY_VMR
    r_disp = total / (vmr - 1)
    p_nb = r_disp / (r_disp + total)
    for line in CARD_LINES:
        overs_pois[str(line)] = round(float(
            1 - stats.poisson.cdf(math.floor(line), total)), 3)
        overs_nb[str(line)] = round(float(
            1 - stats.nbinom.cdf(math.floor(line), r_disp, p_nb)), 3)
    return {"home": round(e_h, 1), "away": round(e_a, 1),
            "total": round(total, 1),
            "referee": ref_name, "referee_factor": round(factor, 2),
            "p_over_poisson": overs_pois,
            "p_over_nb_sensitivity": overs_nb,
            "dispersion_source": "assumed",
            "note": "per-match card counts absent from dataset; Poisson "
                    f"primary, NB var/mean={vmr} sensitivity"}


def fit_position_priors(bundle: Bundle) -> dict[str, tuple[float, float]]:
    """Gamma(alpha, beta) prior on scoring rate per position, via the
    marginal negative-binomial likelihood over all players."""
    ps = bundle.player_stats
    priors = {}
    for pos in ("GK", "DEF", "MID", "FWD"):
        sub = ps[(ps["position"] == pos)
                 & (ps["minutes_played"] / 90.0 >= 0.1)]
        g = sub["goals"].to_numpy(dtype=float)
        e = (sub["minutes_played"] / 90.0).to_numpy(dtype=float)
        # degenerate fallback anchors on the POSITION's own empirical rate
        # (a GK prior must be ~zero, not the outfield pooled rate)
        pos_rate = max(float(g.sum() / max(e.sum(), 1e-9)), 1e-4)
        if len(g) < 10 or g.sum() == 0:
            priors[pos] = (pos_rate * 50.0, 50.0)
            continue

        def nll(params):
            a, b = np.exp(params)
            return -np.sum(gammaln(g + a) - gammaln(a) - gammaln(g + 1)
                           + a * np.log(b / (b + e))
                           + g * np.log(np.clip(e / (b + e), 1e-12, 1)))
        res = optimize.minimize(nll, [0.0, math.log(1.0 / pos_rate)],
                                method="Nelder-Mead")
        a, b = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
        if not np.isfinite(a) or a > 50:
            priors[pos] = (pos_rate * 50.0, 50.0)
        else:
            priors[pos] = (a, b)
    return priors


def scorer_probs(bundle: Bundle, squad_id: int, lam_team: float,
                 priors: dict, top_n: int = 8) -> list[dict]:
    tmap = bundle.tmap
    team_id = int(tmap.loc[tmap["squad_id"] == squad_id, "team_id"].iloc[0])
    roster = bundle.players[bundle.players["team_id"] == team_id].copy()
    ps = bundle.player_stats.set_index("player_id")
    roster["t_goals"] = roster["player_id"].map(ps["goals"]).fillna(0.0)
    roster["exposure"] = (roster["player_id"].map(ps["minutes_played"])
                          .fillna(0.0) / 90.0)

    done = bundle.labels[(bundle.labels["home_squad_id"] == squad_id)
                         | (bundle.labels["away_squad_id"] == squad_id)]
    recent = done.sort_values("round_id").tail(MINUTES_WINDOW)["match_id"]
    lu = bundle.lineups[bundle.lineups["match_id"].isin(recent)]
    minutes = lu.groupby("player_id")["minutes_played"].sum() / (
        90.0 * max(len(recent), 1))
    roster["min_share"] = roster["player_id"].map(minutes).fillna(0.0).clip(
        upper=1.0)

    if roster["min_share"].sum() <= 0:
        pos_n = roster.groupby("position")["player_id"].count()
        share = roster["position"].map(
            lambda p: GOAL_SHARE.get(p, 0.0) / max(int(pos_n.get(p, 1)), 1))
        share = share / share.sum()
    else:
        ab = roster["position"].map(lambda p: priors.get(p, (0.05, 1.0)))
        post_rate = np.array([
            (a + g) / (b + e) for (a, b), g, e in
            zip(ab, roster["t_goals"], roster["exposure"])])
        w = post_rate * roster["min_share"].to_numpy()
        share = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
    p_any = 1.0 - np.exp(-lam_team * np.asarray(share))
    assert bool(((p_any >= 0) & (p_any < 1)).all())
    roster["p_anytime"] = p_any
    top = roster.nlargest(top_n, "p_anytime")
    return [{"player": str(r.player_name), "pos": str(r.position),
             "tournament_goals": int(r.t_goals),
             "p_anytime": round(float(r.p_anytime), 3)}
            for r in top.itertuples(index=False)]


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------

def run_predict(bundle: Bundle, skill: dict, which: str,
                bootstrap_n: int, out_path: Path) -> None:
    labels = bundle.labels
    hp = skill["hyperparams"]["production"]
    dc = fit_dc(labels, hp["dc"]["xi"], hp["dc"]["ridge"])
    xg = fit_xg(labels, hp["xg"]["recency"], hp["xg"]["shrink"])
    mu_t = _mu_tournament(labels)
    by_country = bundle.tmap.set_index("squad_id")["country"]
    priors = fit_position_priors(bundle)

    w1 = skill["weights"]["pool_1x2"]["applied"]
    wa = skill["weights"]["pool_advance"]["applied"]
    wg = skill["weights"]["pool_grid"]["applied"]["dixon_coles"]
    log_1x2 = skill["weights"]["pool_1x2"]["pool_type"] == "log"
    log_adv = skill["weights"]["pool_advance"]["pool_type"] == "log"

    upcoming = bundle.spine[~bundle.spine["completed"]].sort_values(
        "match_id")
    if which != "both":
        upcoming = (upcoming.head(1) if which == "third_place"
                    else upcoming.tail(1))

    # component predictions for the upcoming matches (also feeds bootstrap)
    rows8: dict[int, dict] = {}
    for r in upcoming.itertuples(index=False):
        h, a = int(r.home_squad_id), int(r.away_squad_id)
        row = {}
        for name, (lh, la, rho) in {
                "dixon_coles": (*dc.lambdas(h, a), dc.rho),
                "xg_poisson": (*xg.lambdas(h, a), 0.0),
                "elo": (*elo_lambdas(*bundle.elo[int(r.match_id)], mu_t),
                        0.0)}.items():
            row[f"{name}_1x2"] = grid_1x2(score_grid(lh, la, rho))
            row[f"{name}_adv"] = p_advance_from(lh, la, rho)
            row[f"{name}_lam"] = (lh, la)
        row["dc_rho"] = dc.rho
        row["market_adv"] = market_advance(
            str(by_country[h]), str(by_country[a]), r.kickoff_utc,
            bundle.market_ts)
        rows8[int(r.match_id)] = row

    adv_cis, lam_cis = {}, {}
    if bootstrap_n > 0:
        rng = np.random.default_rng(SEED + 1)
        print(f"bootstrap: B={bootstrap_n} (weights + goal models) ...")
        adv_cis = _weights_bootstrap(skill, rows8, bootstrap_n, rng)
        lam_cis = _lambda_bootstrap(bundle, skill, upcoming, bootstrap_n,
                                    rng)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED,
        "inputs": {
            "n_completed": int(len(labels)),
            "hyperparams": hp,
            "weights_applied": {"pool_1x2": w1, "pool_advance": wa},
            "market_cap": MARKET_CAP,
            "market_snapshot_latest": (
                sorted(SNAP_DIR.glob("snapshot_*.jsonl"))[-1].name
                if list(SNAP_DIR.glob("snapshot_*.jsonl")) else None),
        },
        "matches": [],
    }

    for r in upcoming.itertuples(index=False):
        h, a = int(r.home_squad_id), int(r.away_squad_id)
        h_name, a_name = str(by_country[h]), str(by_country[a])
        row = rows8[int(r.match_id)]

        w1_arr = np.array([w1[c] for c in RATE_COMPONENTS])
        wa_arr = np.array([wa[c] for c in ADV_COMPONENTS])
        ens_1x2 = _pool_1x2(row, w1_arr, log_1x2)
        ens_adv = _pool_adv(row, wa_arr, log_adv)

        # score-grid pool: DC + xG at the dedicated grid-pool weights
        grid = (wg * score_grid(*row["dixon_coles_lam"], dc.rho)
                + (1 - wg) * score_grid(*row["xg_poisson_lam"]))
        grid /= grid.sum()
        lam_mix = (wg * np.array(row["dixon_coles_lam"])
                   + (1 - wg) * np.array(row["xg_poisson_lam"]))
        flat = sorted(
            (((i, j), float(grid[i, j]))
             for i in range(grid.shape[0]) for j in range(grid.shape[1])),
            key=lambda z: -z[1])
        total_lam = float(lam_mix.sum())

        is_final = bool(r.match_id == bundle.spine["match_id"].max())
        block = {
            "match": f"{h_name} v {a_name}",
            "kind": "final" if is_final else "third_place",
            "kickoff_utc": str(r.kickoff_utc),
            "winner": {
                "home": h_name, "away": a_name,
                "p_home_lifts": round(float(ens_adv), 3),
                "p_away_lifts": round(float(1 - ens_adv), 3),
                "market_component_present": row["market_adv"] is not None,
            },
            "p_1x2_90min": {"home": round(float(ens_1x2[0]), 3),
                            "draw": round(float(ens_1x2[1]), 3),
                            "away": round(float(ens_1x2[2]), 3)},
            "expected_goals": {"home": round(float(lam_mix[0]), 2),
                               "away": round(float(lam_mix[1]), 2),
                               "total": round(total_lam, 2)},
            "p_over_2_5": round(float(1 - stats.poisson.cdf(2, total_lam)), 3),
            "p_btts": round(float((1 - math.exp(-lam_mix[0]))
                                  * (1 - math.exp(-lam_mix[1]))), 3),
            "top_scores": [{"score": f"{i}-{j}", "p": round(p, 3)}
                           for (i, j), p in flat[:8]],
            "components": {
                name: {"lam": [round(v, 2) for v in row[f"{name}_lam"]],
                       "p_1x2": [round(float(v), 3)
                                 for v in row[f"{name}_1x2"]],
                       "p_adv": round(float(row[f"{name}_adv"]), 3)}
                for name in RATE_COMPONENTS},
            "market_proxy_p_home": (round(row["market_adv"], 3)
                                    if row["market_adv"] is not None
                                    else None),
            "uncertainty": ({
                "p_advance_ci90": adv_cis.get(int(r.match_id)),
                **lam_cis.get(int(r.match_id), {}),
                "component_range_p_adv": [
                    round(min(float(row[f"{c}_adv"])
                              for c in RATE_COMPONENTS), 3),
                    round(max(float(row[f"{c}_adv"])
                              for c in RATE_COMPONENTS), 3)],
                "B": bootstrap_n,
                "sources": "p_advance CI: weight-selection bootstrap (can "
                           "be narrow when the learned weights are stable); "
                           "component_range: structural disagreement across "
                           "models; lambdas: match-resampling refit of the "
                           "goal models",
            } if bootstrap_n > 0 else None),
            "corners": expected_corners(bundle, h, a),
            "cards": expected_cards(bundle, pd.Series(r._asdict())),
            "scorers": {
                h_name: scorer_probs(bundle, h, float(lam_mix[0]), priors),
                a_name: scorer_probs(bundle, a, float(lam_mix[1]), priors),
            },
        }
        payload["matches"].append(block)
        _print_block(block)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"\nwritten: {out_path}")


def _print_block(b: dict) -> None:
    w = b["winner"]
    u = b.get("uncertainty") or {}
    ci = u.get("p_advance_ci90")
    cr = u.get("component_range_p_adv")
    print(f"\n=== {b['match']} ({b['kind']}) ===")
    print(f"win: {w['home']} {w['p_home_lifts']:.1%} / "
          f"{w['away']} {w['p_away_lifts']:.1%}"
          + (f"   90% CI [{ci[0]:.0%}, {ci[1]:.0%}]" if ci else "")
          + (f"   model range [{cr[0]:.0%}, {cr[1]:.0%}]" if cr else "")
          + ("   [market in pool]" if w["market_component_present"]
             else "   [model only]"))
    p = b["p_1x2_90min"]
    eg = b["expected_goals"]
    print(f"90 min: {p['home']:.1%} / {p['draw']:.1%} / {p['away']:.1%}"
          f"   xG {eg['home']}-{eg['away']}"
          f"   O2.5 {b['p_over_2_5']:.1%}  BTTS {b['p_btts']:.1%}")
    print("top scores: " + ", ".join(
        f"{s['score']} ({s['p']:.1%})" for s in b["top_scores"][:5]))
    for name, c in b["components"].items():
        print(f"  {name:<12} lam {c['lam'][0]:.2f}-{c['lam'][1]:.2f}  "
              f"1x2 {c['p_1x2'][0]:.2f}/{c['p_1x2'][1]:.2f}/"
              f"{c['p_1x2'][2]:.2f}  adv {c['p_adv']:.2f}")
    if b["market_proxy_p_home"] is not None:
        print(f"  {'market':<12} p_home {b['market_proxy_p_home']:.2f} "
              f"(capped weight)")
    c = b["corners"]
    print(f"corners: {c['home']} + {c['away']} = {c['total']} "
          f"({c['family']}); P(over) "
          + " ".join(f"{k}:{v:.0%}" for k, v in c["p_over"].items()))
    k = b["cards"]
    print(f"cards: {k['total']} total (ref {k['referee']}, factor "
          f"{k['referee_factor']}); P(over, Poisson) "
          + " ".join(f"{kk}:{vv:.0%}"
                     for kk, vv in k["p_over_poisson"].items()))
    for team, scorers in b["scorers"].items():
        tops = ", ".join(f"{s['player']} {s['p_anytime']:.0%}"
                         for s in scorers[:5])
        print(f"scorers {team}: {tops}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research-grade round-8 match forecaster.")
    parser.add_argument("--fit", action="store_true",
                        help="re-run walk-forward, tuning and weights")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--match", choices=("third_place", "final", "both"),
                        default="both")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    bundle = load_bundle()
    skill = None
    if not args.fit and SKILL_OUT.exists():
        skill = json.loads(SKILL_OUT.read_text())
        if skill.get("n_completed") != int(len(bundle.labels)):
            print("skill artifact stale (completed-match count changed); "
                  "re-fitting")
            skill = None
    if skill is None:
        skill = run_fit(bundle)

    run_predict(bundle, skill, args.match,
                0 if args.skip_bootstrap else args.bootstrap, args.out)


if __name__ == "__main__":
    main()
