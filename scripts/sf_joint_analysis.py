"""Joint squad+XI+captain optimization and booster analysis for a knockout round.

Motivation (whitepaper section 05d). The production transfer solver maximizes
the summed expected points of the full 15-player squad, but only the starting
XI plus a doubled captain scores. During semifinal planning a user-imposed
keep-this-player constraint produced a HIGHER-scoring XI than the
unconstrained solve, which is impossible under a correctly specified
objective and exposed the mismatch. This script optimizes what actually
scores, in one MILP:

    max  sum(eff_p * xi_p) + sum(eff_p * cap_p)
         - 3 * extra_transfers + 0.1 * bench_eff
         (+ booster payoff when a booster is part of the decision)

subject to squad shape (2/5/5/3), budget, per-country cap, XI nested in
squad, one legal formation, captain in XI, and the stage transfer quota.

Two deliberate upgrades over the production pipeline, both strictly more
honest:

  * scouting bonus enters as 2 * P(points > 4), evaluated on the player's
    predicted quantile distribution, instead of a deterministic +2
    (pipeline.py documents its own version as coarse);
  * team advance probabilities blend the poisson backend's match model with
    prediction-market prices 50/50 instead of being hand-set.

A Monte Carlo layer (Gaussian copula, shared team factor) then prices every
candidate plan, captain choice, and booster against a template-manager XI
built from ownership, because expected value alone cannot rank chase
strategies. Eliminated players still in the user's squad are modeled as
retainable zero-point assets: keeping a dead bench player is legal and free,
and with few transfers left it is often correct.

Usage:
    python scripts/sf_joint_analysis.py                       # from-scratch solve
    python scripts/sf_joint_analysis.py --squad my15.json --free 2
    python scripts/sf_joint_analysis.py --compare             # sim the stored picks

--squad JSON shape: {"squad": [15 ids], "xi": [11 ids], "captain": id}
(xi/captain optional; solved when omitted). --compare reads
data/evaluation/sf_pick_comparison.json and re-simulates both stored picks.

The fantasy round's matches are read from the newest fixtures parquet, so the
script works unchanged for the FINAL round (the third-place match does not
award fantasy points and is excluded by the fixtures round grouping).
Deterministic under --seed (default 7).
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
from scipy import stats

from fifa_fantasy.collector.schemas import Stage
from fifa_fantasy.model.poisson import _team_xg
from fifa_fantasy.optimizer.pipeline import (
    apply_availability_discount, apply_scouting_bonus,
)
from fifa_fantasy.optimizer.solvers import (
    SQUAD_POSITION_COUNTS, TRANSFER_HIT_POINTS, VALID_FORMATIONS,
)
from fifa_fantasy.optimizer.stage_config import STAGE_CONFIGS

REPO_ROOT = Path(__file__).resolve().parents[1]

BENCH_WEIGHT = 0.1
RHO_TEAM = 0.3
Z90 = 1.2816
CS_POINTS = {"GK": 5.0, "DEF": 5.0, "MID": 1.0, "FWD": 0.0}
# fallback per-position residual sigma; overridden by the walk-forward
# artifact's pooled C_eplwc_form row when present
SIGMA_POS_FALLBACK = {"GK": 2.415, "DEF": 2.851, "MID": 2.631, "FWD": 2.761}
ROUND_STAGE = {4: Stage.R32, 5: Stage.R16, 6: Stage.QF, 7: Stage.SF,
               8: Stage.FINAL}
# Round-8 booster values used in the pairing table when the analyzed round is
# the SF. Qualification in the final assumes an 8+3 split on a 55 percent
# favorite (2 * (8p + 3(1-p))) with a small execution haircut; the others are
# scenario-weighted estimates derived in the whitepaper section.
FINAL_ROUND_BOOSTER_EV = {"qualification": 11.0, "shield": 7.0,
                          "twelfth": 6.5, "wildcard": 5.7}


def _latest(pattern: str) -> Path:
    matches = sorted(REPO_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no match for {pattern}")
    return matches[-1]


def load_sigma_pos() -> dict[str, float]:
    path = REPO_ROOT / "data/evaluation/wc_forward_validation.json"
    if not path.exists():
        return dict(SIGMA_POS_FALLBACK)
    pooled = json.loads(path.read_text()).get("pooled", {})
    row = pooled.get("C_eplwc_form", {})
    return {p: float(row.get(p, SIGMA_POS_FALLBACK[p]))
            for p in SIGMA_POS_FALLBACK}


def market_probs(snapshot_path: Path, teams: list[str],
                 abbr_to_name: dict[str, str]) -> dict[str, float] | None:
    """Implied tournament-winner shares for the given teams, renormalized."""
    raw: dict[str, float] = {}
    for line in snapshot_path.read_text().splitlines():
        d = json.loads(line)
        m = re.match(r"^Will (.+) win the 2026 FIFA World Cup\?$",
                     d.get("title", ""))
        if m and d.get("yes_price"):
            raw[m.group(1)] = float(d["yes_price"])
    named = {t: raw.get(abbr_to_name[t]) for t in teams}
    if any(v is None for v in named.values()):
        return None
    return named  # per-match renormalization happens in blend_advance_probs


def poisson_match_prob(lam_a: float, lam_b: float, kmax: int = 12) -> float:
    """P(A beats B) incl. extra time, tilted by strength on draws."""
    pa = stats.poisson.pmf(np.arange(kmax + 1), lam_a)
    pb = stats.poisson.pmf(np.arange(kmax + 1), lam_b)
    grid = np.outer(pa, pb)
    win = float(np.tril(grid, -1).sum())
    draw = float(np.trace(grid))
    return win + draw * (lam_a / (lam_a + lam_b))


def blend_advance_probs(matches: list[tuple[str, str]], lam: dict[str, float],
                        mkt: dict[str, float] | None) -> dict[str, float]:
    p_adv: dict[str, float] = {}
    for a, b in matches:
        model = poisson_match_prob(lam[a], lam[b])
        if mkt and mkt.get(a) and mkt.get(b):
            market = mkt[a] / (mkt[a] + mkt[b])
            p = 0.5 * model + 0.5 * market
        else:
            p = model
        p_adv[a], p_adv[b] = p, 1.0 - p
    return p_adv


def build_pool(predictions_path: Path, fixtures_path: Path, round_id: int,
               market_snapshot: Path | None,
               current_ids: list[int] | None) -> tuple[pd.DataFrame, dict]:
    """One row per player available for the round, scored and annotated.

    Members of `current_ids` whose teams are eliminated are appended as
    retainable zero-point rows.
    """
    pred = pd.read_parquet(predictions_path)
    pred = apply_scouting_bonus(pred)
    pred = apply_availability_discount(pred)
    pool = pred[(pred["round_id"] == round_id)
                & (~pred["is_eliminated"].astype(bool))].copy()
    if pool.empty:
        raise SystemExit(f"no round-{round_id} rows in {predictions_path}")

    sigma_pos = load_sigma_pos()
    mu = pool["predicted_points"].astype(float) * pool["availability_factor"]
    sig = pool["position"].map(sigma_pos)
    q10 = (pool["predicted_q10"].astype(float) * pool["availability_factor"]
           ).fillna((mu - Z90 * sig).clip(lower=0.0))
    q50 = (pool["predicted_q50"].astype(float) * pool["availability_factor"]
           ).fillna(mu)
    q90 = (pool["predicted_q90"].astype(float) * pool["availability_factor"]
           ).fillna(mu + Z90 * sig)
    q50 = np.maximum(q50, q10)
    q90 = np.maximum(q90, q50 + 1e-6)
    pool["s_q10"], pool["s_q50"], pool["s_q90"] = q10, q50, q90

    spread = ((pool["s_q90"] - pool["s_q10"]) / (2 * Z90)).clip(lower=0.5)
    p_gt4 = 1.0 - stats.norm.cdf(4.0, loc=pool["s_q50"], scale=spread)
    pool["scout_ev"] = 2.0 * p_gt4 * (pool["ownership_fraction"] < 0.05)
    pool["eff"] = mu + pool["scout_ev"]
    pool["eff_q90"] = pool["s_q90"] + 2.0 * (
        (pool["s_q90"] > 4.0) & (pool["ownership_fraction"] < 0.05))

    own_xg, _ = _team_xg(pool)
    pool["own_xg"] = own_xg
    team_rows = pool.drop_duplicates("country_abbr").set_index("country_abbr")
    lam = {t: float(team_rows.loc[t, "own_xg"]) for t in team_rows.index}

    fixtures = pd.read_parquet(fixtures_path)
    rnd_fx = fixtures[fixtures["round_id"] == round_id]
    matches = [(r.home_squad_abbr, r.away_squad_abbr)
               for r in rnd_fx.itertuples()]
    opp = {}
    for a, b in matches:
        opp[a], opp[b] = b, a

    mkt = None
    if market_snapshot is not None:
        abbr_to_name = {t: str(team_rows.loc[t, "country"])
                        for t in team_rows.index}
        mkt = market_probs(market_snapshot, list(team_rows.index),
                           abbr_to_name)
    p_adv = blend_advance_probs(matches, lam, mkt)

    pool["p_adv"] = pool["country_abbr"].map(p_adv)
    pool["p_concede_one"] = pool["country_abbr"].map(
        lambda t: float(stats.poisson.pmf(1, lam[opp[t]])))
    pool["cs_pts"] = pool["position"].map(CS_POINTS)

    if current_ids:
        missing = set(current_ids) - set(int(p) for p in pool["player_id"])
        if missing:
            meta = pred.drop_duplicates("player_id").set_index("player_id")
            rows = []
            for pid in sorted(missing):
                rows.append({
                    "player_id": pid,
                    "full_name": str(meta.loc[pid, "full_name"]),
                    "position": str(meta.loc[pid, "position"]),
                    "country": str(meta.loc[pid, "country"]),
                    "country_abbr": str(meta.loc[pid, "country_abbr"]),
                    "price_millions": float(meta.loc[pid, "price_millions"]),
                })
            dead = pd.DataFrame(rows)
            for col in ("eff", "eff_q90", "scout_ev", "availability_factor",
                        "p_adv", "p_concede_one", "cs_pts", "s_q10", "s_q50",
                        "s_q90", "ownership_fraction"):
                dead[col] = 0.0
            pool = pd.concat([pool, dead], ignore_index=True)

    pool = pool.reset_index(drop=True)
    ctx = {"lam": lam, "matches": matches, "opp": opp, "p_adv": p_adv}
    return pool, ctx


def solve_joint(pool: pd.DataFrame, config, current_ids: list[int] | None,
                free: int, booster: str = "none", alpha: float = 0.0,
                force_in: tuple[int, ...] = (),
                force_out: tuple[int, ...] = ()) -> dict:
    """One MILP for squad, XI, formation, captain, and transfers."""
    n = len(pool)
    current = set(current_ids or [])
    prob = pulp.LpProblem("joint", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]
    y = [pulp.LpVariable(f"y{i}", cat="Binary") for i in range(n)]
    c = [pulp.LpVariable(f"c{i}", cat="Binary") for i in range(n)]
    f = {k: pulp.LpVariable(f"f_{k}", cat="Binary") for k in VALID_FORMATIONS}
    extra = pulp.LpVariable("extra", lowBound=0)

    score = (1 - alpha) * pool["eff"] + alpha * pool["eff_q90"]
    obj = (
        pulp.lpSum(float(score[i]) * (y[i] + c[i]) for i in range(n))
        + BENCH_WEIGHT * pulp.lpSum(
            float(pool["eff"][i]) * (x[i] - y[i]) for i in range(n))
        - TRANSFER_HIT_POINTS * extra
    )
    if booster == "qualification":
        obj += pulp.lpSum(
            2.0 * float(pool["p_adv"][i] * pool["availability_factor"][i])
            * y[i] for i in range(n))
    elif booster == "shield":
        obj += pulp.lpSum(
            float(pool["cs_pts"][i] * pool["p_concede_one"][i]) * y[i]
            for i in range(n))
    elif booster == "twelfth":
        m = [pulp.LpVariable(f"m{i}", cat="Binary") for i in range(n)]
        for i in range(n):
            prob += m[i] + x[i] <= 1
        prob += pulp.lpSum(m) == 1
        obj += pulp.lpSum(float(pool["eff"][i]) * m[i] for i in range(n))
    prob += obj

    prob += pulp.lpSum(x) == 15
    for pos, cnt in SQUAD_POSITION_COUNTS.items():
        ids = [i for i in range(n) if pool["position"][i] == pos.value]
        prob += pulp.lpSum(x[i] for i in ids) == cnt
    prob += pulp.lpSum(
        float(pool["price_millions"][i]) * x[i] for i in range(n)
    ) <= config.budget_millions
    for _, grp in pool.groupby("country_abbr"):
        prob += pulp.lpSum(x[i] for i in grp.index) <= config.max_per_country
    for i in range(n):
        prob += y[i] <= x[i]
        prob += c[i] <= y[i]
    prob += pulp.lpSum(y) == 11
    prob += pulp.lpSum(c) == 1
    prob += pulp.lpSum(f.values()) == 1
    gk = [i for i in range(n) if pool["position"][i] == "GK"]
    prob += pulp.lpSum(y[i] for i in gk) == 1
    for pos, k in (("DEF", 0), ("MID", 1), ("FWD", 2)):
        ids = [i for i in range(n) if pool["position"][i] == pos]
        prob += (pulp.lpSum(y[i] for i in ids)
                 == pulp.lpSum(f[name] * cnt[k]
                               for name, cnt in VALID_FORMATIONS.items()))
    if booster != "wildcard" and current:
        new = pulp.lpSum(x[i] for i in range(n)
                         if int(pool["player_id"][i]) not in current)
        prob += extra >= new - free
    for pid in force_in:
        i = int(pool.index[pool["player_id"] == pid][0])
        prob += x[i] == 1
    for pid in force_out:
        i = int(pool.index[pool["player_id"] == pid][0])
        prob += x[i] == 0

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"joint solver: {pulp.LpStatus[status]}")

    squad = [i for i in range(n) if x[i].value() > 0.5]
    xi = [i for i in range(n) if y[i].value() > 0.5]
    cap = [i for i in range(n) if c[i].value() > 0.5][0]
    n_new = sum(1 for i in squad
                if int(pool["player_id"][i]) not in current) if current else 0
    hits = 0.0 if booster == "wildcard" else float(
        TRANSFER_HIT_POINTS * max(0, n_new - free))
    return dict(squad=squad, xi=xi, cap=cap, n_new=n_new, hits=hits,
                booster=booster, alpha=alpha)


class RoundSimulation:
    """Correlated Monte Carlo of one knockout round."""

    def __init__(self, pool: pd.DataFrame, ctx: dict, n_sims: int, seed: int,
                 rho: float = RHO_TEAM):
        rng = np.random.default_rng(seed)
        self.pool, self.n_sims = pool, n_sims
        teams = sorted(ctx["lam"])
        team_col = pool["country_abbr"].to_numpy()
        z_team = {t: rng.standard_normal(n_sims) for t in teams}
        z_ind = rng.standard_normal((n_sims, len(pool)))
        z = z_ind.copy()
        for t in teams:
            ix = np.where(team_col == t)[0]
            z[:, ix] = (np.sqrt(rho) * z_team[t][:, None]
                        + np.sqrt(1 - rho) * z_ind[:, ix])
        u = stats.norm.cdf(z)

        q10 = pool["s_q10"].to_numpy()
        q50 = pool["s_q50"].to_numpy()
        q90 = pool["s_q90"].to_numpy()
        pts = np.empty_like(u)
        lo = u <= 0.10
        pts[lo] = (u[lo] / 0.10) * np.broadcast_to(q10, u.shape)[lo]
        mid = (u > 0.10) & (u <= 0.50)
        pts[mid] = (np.broadcast_to(q10, u.shape)[mid]
                    + (u[mid] - 0.10) / 0.40
                    * np.broadcast_to(q50 - q10, u.shape)[mid])
        hi = (u > 0.50) & (u <= 0.90)
        pts[hi] = (np.broadcast_to(q50, u.shape)[hi]
                   + (u[hi] - 0.50) / 0.40
                   * np.broadcast_to(q90 - q50, u.shape)[hi])
        tail = u > 0.90
        slope = (q90 - q50) / 0.40
        pts[tail] = (np.broadcast_to(q90, u.shape)[tail]
                     + (np.minimum(u[tail], 0.995) - 0.90)
                     * np.broadcast_to(slope, u.shape)[tail])
        pts = np.clip(pts, 0, None)
        own = pool["ownership_fraction"].to_numpy()
        pts += 2.0 * ((pts > 4.0) & (own < 0.05))
        self.pts = pts

        goals = {t: stats.poisson.ppf(stats.norm.cdf(z_team[t]),
                                      ctx["lam"][t]).astype(int)
                 for t in teams}
        adv_team: dict[str, np.ndarray] = {}
        for a, b in ctx["matches"]:
            win_a = goals[a] > goals[b]
            draw = goals[a] == goals[b]
            tilt = rng.random(n_sims) < (
                ctx["lam"][a] / (ctx["lam"][a] + ctx["lam"][b]))
            adv_team[a] = win_a | (draw & tilt)
            adv_team[b] = ~adv_team[a]
        self.adv = np.zeros((n_sims, len(pool)), dtype=bool)
        self.conc_one = np.zeros((n_sims, len(pool)), dtype=bool)
        for t in teams:
            ix = np.where(team_col == t)[0]
            self.adv[:, ix] = adv_team[t][:, None]
            self.conc_one[:, ix] = (goals[ctx["opp"][t]] == 1)[:, None]
        self.cs = pool["cs_pts"].to_numpy()
        self.field = self._template_score()

    def _template_score(self) -> np.ndarray:
        top = self.pool.sort_values("ownership_fraction", ascending=False)
        xi = (top[top.position == "GK"].index[:1].tolist()
              + top[top.position == "DEF"].index[:3].tolist()
              + top[top.position == "MID"].index[:5].tolist()
              + top[top.position == "FWD"].index[:2].tolist())
        own = self.pool["ownership_fraction"].to_numpy()
        cap = max(xi, key=lambda i: own[i])
        return self.pts[:, xi].sum(axis=1) + self.pts[:, cap]

    def score(self, plan: dict) -> np.ndarray:
        xi, cap = plan["xi"], plan["cap"]
        total = (self.pts[:, xi].sum(axis=1) + self.pts[:, cap]
                 - plan.get("hits", 0.0))
        booster = plan.get("booster", "none")
        if booster == "qualification":
            total = total + 2.0 * self.adv[:, xi].sum(axis=1)
        elif booster == "shield":
            total = total + (self.cs[xi][None, :]
                             * self.conc_one[:, xi]).sum(axis=1)
        elif booster == "twelfth":
            outside = np.setdiff1d(np.arange(len(self.pool)), plan["squad"])
            best = outside[np.argmax(self.pool["eff"].to_numpy()[outside])]
            plan["twelfth_pick"] = str(self.pool["full_name"][best])
            total = total + self.pts[:, best]
        return total

    def summarize(self, plan: dict) -> dict:
        total = self.score(plan)
        diff = total - self.field
        return {
            "ev": round(float(total.mean()), 2),
            "p90": round(float(np.percentile(total, 90)), 2),
            "p_beat_field": round(float((diff >= 0).mean()), 3),
            "p_beat_field_by_15": round(float((diff >= 15).mean()), 3),
        }


def plan_payload(pool: pd.DataFrame, plan: dict, summary: dict) -> dict:
    def named(ids):
        return [{"id": int(pool["player_id"][i]),
                 "name": str(pool["full_name"][i]),
                 "pos": str(pool["position"][i]),
                 "team": str(pool["country_abbr"][i])} for i in ids]
    bench = [i for i in plan["squad"] if i not in plan["xi"]]
    return {
        "booster": plan.get("booster", "none"),
        "alpha": plan.get("alpha", 0.0),
        "transfers": plan.get("n_new", 0),
        "hit_points": plan.get("hits", 0.0),
        "captain": named([plan["cap"]])[0],
        "xi": named(plan["xi"]),
        "bench": named(bench),
        "twelfth_pick": plan.get("twelfth_pick"),
        "simulation": summary,
    }


def plan_from_ids(pool: pd.DataFrame, squad_ids: list[int],
                  xi_ids: list[int], captain_id: int, booster: str) -> dict:
    idx = {int(p): i for i, p in enumerate(pool["player_id"])}
    return dict(squad=[idx[p] for p in squad_ids],
                xi=[idx[p] for p in xi_ids], cap=idx[captain_id],
                hits=0.0, booster=booster, n_new=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint XI-aware round optimization and booster analysis.")
    parser.add_argument("--round", type=int, default=7)
    parser.add_argument("--squad", type=Path, default=None,
                        help="JSON {squad: [15 ids], xi: [...], captain: id}")
    parser.add_argument("--free", type=int, default=None,
                        help="free transfers left (default: stage quota)")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--market-snapshot", type=Path, default=None)
    parser.add_argument("--sims", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--compare", action="store_true",
                        help="re-simulate the stored SF pick comparison")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "data/evaluation/sf_joint_analysis.json")
    args = parser.parse_args()

    predictions = args.predictions or _latest("data/processed/predictions_*.parquet")
    fixtures = args.fixtures or _latest("data/raw/fixtures_*.parquet")
    market = args.market_snapshot
    if market is None:
        try:
            market = _latest("data/external/prediction_markets/snapshot_*.jsonl")
        except FileNotFoundError:
            market = None

    current_ids: list[int] | None = None
    given_xi: list[int] | None = None
    given_cap: int | None = None
    if args.squad:
        given = json.loads(args.squad.read_text())
        current_ids = [int(p) for p in given["squad"]]
        given_xi = [int(p) for p in given.get("xi", [])] or None
        given_cap = int(given["captain"]) if "captain" in given else None

    config = STAGE_CONFIGS[ROUND_STAGE[args.round]]
    free = args.free if args.free is not None else (config.free_transfers or 0)
    pool, ctx = build_pool(predictions, fixtures, args.round, market,
                           current_ids)
    sim = RoundSimulation(pool, ctx, args.sims, args.seed)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "round_id": args.round,
        "inputs": {"predictions": str(predictions), "fixtures": str(fixtures),
                   "market_snapshot": str(market) if market else None,
                   "sims": args.sims, "seed": args.seed,
                   "free_transfers": free},
        "advance_probs": {t: round(p, 3) for t, p in ctx["p_adv"].items()},
        "plans": [],
    }

    if args.compare:
        stored_path = REPO_ROOT / "data/evaluation/sf_pick_comparison.json"
        stored = json.loads(stored_path.read_text())
        for label in ("user_pick", "model_pick"):
            pick = stored[label]
            squad_ids = [e["id"] for e in pick["xi"] + pick["bench"]]
            cmp_pool, _ = build_pool(predictions, fixtures, args.round,
                                     market, squad_ids)
            cmp_sim = RoundSimulation(cmp_pool, ctx, args.sims, args.seed)
            for booster in ("none", "shield", "qualification"):
                plan = plan_from_ids(cmp_pool, squad_ids,
                                     [e["id"] for e in pick["xi"]],
                                     pick["captain"], booster)
                summary = cmp_sim.summarize(plan)
                payload["plans"].append(
                    {"label": f"{label}/{booster}",
                     **plan_payload(cmp_pool, plan, summary)})
                print(f"{label:<11} {booster:<14} EV={summary['ev']:6.1f} "
                      f"p90={summary['p90']:6.1f}")
    else:
        for booster in ("none", "qualification", "shield", "twelfth",
                        "wildcard"):
            plan = solve_joint(pool, config, current_ids, free, booster)
            summary = sim.summarize(plan)
            payload["plans"].append(
                {"label": f"solve/{booster}",
                 **plan_payload(pool, plan, summary)})
            print(f"{booster:<14} transfers={plan['n_new']} "
                  f"hit=-{plan['hits']:.0f} EV={summary['ev']:6.1f} "
                  f"p90={summary['p90']:6.1f} "
                  f"P>field={summary['p_beat_field']:.3f}")

        best = max(payload["plans"], key=lambda p: p["simulation"]["ev"])
        best_plan = solve_joint(pool, config, current_ids, free,
                                best["booster"])
        sweep = []
        for ci in best_plan["xi"]:
            trial = dict(best_plan, cap=ci)
            s = sim.summarize(trial)
            sweep.append({"captain": str(pool["full_name"][ci]), **s})
        sweep.sort(key=lambda r: -r["ev"])
        payload["captain_sweep"] = sweep
        print("\ncaptain sweep (best plan):")
        for row in sweep[:5]:
            print(f"  {row['captain']:<22} EV={row['ev']:6.1f} "
                  f"P>=+15={row['p_beat_field_by_15']:.3f}")

        if args.round == 7:
            gains = {p["booster"]: p["simulation"]["ev"]
                     for p in payload["plans"]}
            base = gains.pop("none")
            pairing = {}
            for now, g in gains.items():
                if now == "wildcard":
                    continue
                for later, ev8 in FINAL_ROUND_BOOSTER_EV.items():
                    if later != now:
                        pairing[f"{now}+{later}"] = round(g - base + ev8, 1)
            payload["booster_pairing"] = dict(
                sorted(pairing.items(), key=lambda kv: -kv[1]))
            print("\nbooster pairing (SF gain + Final estimate):")
            for k, v in list(payload["booster_pairing"].items())[:4]:
                print(f"  {k:<28} {v:5.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
