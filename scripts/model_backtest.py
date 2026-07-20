"""Cross-round model backtest: model squads vs user squads vs random baseline.

For each completed round we:
  1. Run every model backend on that round's historic features file.
  2. Run the optimizer with the appropriate stage configuration.
  3. Score the model's recommended squad using the realised round_points.
  4. Score the user's actual squad (from data/user_squads/round_NN.json).
  5. Generate a random baseline via Monte Carlo over valid random squads.

Output:
  - Per-round table to stdout
  - data/evaluation/backtest_summary.json with the cumulative comparison

Limitations:
  - The historical features files are point-in-time snapshots; we use the
    closest available snapshot per round. For MD1 use features_2026-06-08
    (pre-tournament), MD2 use features_2026-06-18, MD3 use features_2026-06-23,
    R32 use features_2026-06-28.
  - The transfer constraint between rounds is ignored: each round, the
    model optimizer is allowed unlimited transfers from a fresh squad.
    This makes the model "overstate" what it would have actually
    recommended given the transfer-cost penalty, but it gives us an
    honest upper bound on the model's prediction quality.
  - The user's actual squad may have included transfer-cost hits which
    are subtracted from realised_total_pts in the JSON.
  - IN-SAMPLE CAVEAT for WC-trained backends: when the production GBM is
    trained with --include-wc, it has seen the realised labels of every
    completed WC round, including the rounds scored here. The `gbm` and
    `ensemble` columns are therefore in-sample upper bounds on those
    rounds, not out-of-sample estimates. The leak-free measure of the
    form + WC-label change is the per-position walk-forward RMSE in
    scripts/wc_forward_validation.py, which trains only on WC rounds
    strictly before the round it scores. The heuristic, Poisson and
    Monte Carlo columns never train on WC and remain out-of-sample.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

from fifa_fantasy.collector.schemas import Stage
from fifa_fantasy.features.build import _attach_form_lag
from fifa_fantasy.model.baseline import heuristic_predict
from fifa_fantasy.model.baseline_v2 import heuristic_v2_predict
from fifa_fantasy.model.ensemble import ensemble_predict
from fifa_fantasy.model.gbm import DEFAULT_MODELS_DIR, load_models, predict as gbm_predict
from fifa_fantasy.model.monte_carlo import mc_predict
from fifa_fantasy.model.poisson import poisson_predict
from fifa_fantasy.optimizer.solvers import (
    SQUAD_POSITION_COUNTS, SQUAD_SIZE, solve_lineup,
)
from fifa_fantasy.optimizer.stage_config import STAGE_CONFIGS

OUT_DIR = Path("data/evaluation")

# Round -> (features_file, raw_players_file, stage). Pick the snapshot
# closest to but BEFORE the deadline of the listed round.
ROUND_PLAN = [
    (1, "features_2026-06-08.parquet", "players_2026-07-04.parquet", Stage.GROUP_MD1),
    (2, "features_2026-06-18.parquet", "players_2026-07-04.parquet", Stage.GROUP_MD2),
    (3, "features_2026-06-23.parquet", "players_2026-07-04.parquet", Stage.GROUP_MD3),
    # R32 is now complete; realised scores live in round_points index 3.
    (4, "features_2026-06-28.parquet", "players_2026-07-04.parquet", Stage.R32),
    # R16 complete (realised scores at index 4). The 07-03 snapshot is the
    # last strictly pre-round one; it covers 12 of 16 R16 squads (four
    # pairings were decided by late R32 games). The 07-04 snapshot has all
    # 16 but was taken after the first R16 match had recorded points (max
    # round_points length 5), which leaks that fixture's own labels into
    # its stored form_lag. Clean beats complete.
    (5, "features_2026-07-03.parquet", "players_2026-07-08.parquet", Stage.R16),
    # QF complete (realised scores at index 5). The 07-08 snapshot is the
    # last strictly pre-round one: max stored round_points length is 5
    # (nothing from the QF leaked into its form features) and its round-6
    # rows cover all 8 QF squads.
    (6, "features_2026-07-08.parquet", "players_2026-07-17.parquet", Stage.QF),
    # SF complete (realised scores at index 6). Same check: the 07-13
    # snapshot's max round_points length is 6 and it covers all 4 SF squads.
    (7, "features_2026-07-13.parquet", "players_2026-07-17.parquet", Stage.SF),
    # FINAL round (bronze + final, both score). The 07-18 snapshot was
    # built pre-round (max round_points length 7, both round-8 fixtures
    # scheduled); realised scores at index 7 from the post-final pull.
    # Note: raw API round points exclude booster bonuses.
    (8, "features_2026-07-18.parquet", "players_2026-07-19.parquet", Stage.FINAL),
]

BACKENDS = ("heuristic", "heuristic_v2", "poisson", "gbm", "monte_carlo", "ensemble")


class _SkipAB(Exception):
    """Availability A/B not applicable for this round's archived features."""


@dataclass(frozen=True)
class RoundResult:
    round_id: int
    stage: str
    backend: str
    squad_player_ids: list[int]
    starter_ids: list[int]
    captain_id: int
    formation: str
    realised_starter_pts: int
    realised_captain_raw: int
    realised_total: int
    predicted_total: float  # solver-expected XI + captain points, pre-round


def _round_points(raw_players: pd.DataFrame, round_id: int) -> dict[int, int]:
    out = {}
    for r in raw_players.itertuples():
        rp = list(r.round_points) if r.round_points is not None else []
        idx = round_id - 1
        out[int(r.player_id)] = int(rp[idx]) if 0 <= idx < len(rp) else 0
    return out


def _predict(features: pd.DataFrame, backend: str) -> pd.DataFrame:
    if backend == "heuristic":
        return heuristic_predict(features)
    if backend == "heuristic_v2":
        return heuristic_v2_predict(features)
    if backend == "poisson":
        return poisson_predict(features)
    if backend == "gbm":
        models = load_models(DEFAULT_MODELS_DIR)
        return gbm_predict(features, models)
    if backend == "monte_carlo":
        return mc_predict(features)
    if backend == "ensemble":
        return ensemble_predict(features)
    raise ValueError(backend)


def _solve_squad(predictions: pd.DataFrame, stage: Stage, round_id: int) -> tuple[list[int], str, int, list[int]]:
    """Solve a fresh squad against this round's predictions."""
    config = STAGE_CONFIGS[stage]
    p = predictions.copy()
    if "predicted_points" not in p.columns:
        raise ValueError("predictions table needs predicted_points column")
    # Aggregate to per-player effective points for this round.
    in_scope = p[p["round_id"] == round_id]
    if in_scope.empty:
        # Maybe features file doesn't cover this round; take any rows we have.
        in_scope = p
    agg = in_scope.groupby("player_id", sort=False).agg(
        eff=("predicted_points", "sum"),
    ).reset_index()
    meta_cols = ["player_id", "position", "country", "price_millions", "is_eliminated"]
    meta = in_scope[meta_cols].drop_duplicates("player_id")
    table = meta.merge(agg, on="player_id", how="inner")
    table = table[~table["is_eliminated"].astype(bool)].reset_index(drop=True)

    prob = pulp.LpProblem("backtest", pulp.LpMaximize)
    x = {int(r.player_id): pulp.LpVariable(f"x_{int(r.player_id)}", cat="Binary")
         for r in table.itertuples()}
    prob += pulp.lpSum(x[int(r.player_id)] * float(r.eff) for r in table.itertuples())
    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    for pos, count in SQUAD_POSITION_COUNTS.items():
        ids = [int(r.player_id) for r in table.itertuples() if r.position == pos.value]
        prob += pulp.lpSum(x[i] for i in ids) == count
    prob += pulp.lpSum(x[int(r.player_id)] * float(r.price_millions)
                       for r in table.itertuples()) <= config.budget_millions
    for c, gp in table.groupby("country", sort=False):
        ids = [int(pid) for pid in gp["player_id"]]
        prob += pulp.lpSum(x[i] for i in ids) <= config.max_per_country
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    chosen = sorted(int(pid) for pid, v in x.items() if v.value() > 0.5)
    # Run lineup solver for this round.
    md = in_scope[in_scope["player_id"].isin(chosen)][
        ["player_id", "position", "predicted_points"]
    ].copy()
    lineup = solve_lineup(md)
    return chosen, lineup.formation, lineup.captain_id, lineup.starter_ids


def _score_squad(starter_ids: list[int], captain_id: int,
                 realised: dict[int, int]) -> tuple[int, int, int]:
    starter_pts = sum(realised.get(pid, 0) for pid in starter_ids)
    cap_pts = realised.get(captain_id, 0)
    total = starter_pts + cap_pts  # captain counted twice
    return starter_pts, cap_pts, total


def _random_baseline(predictions: pd.DataFrame, stage: Stage, round_id: int,
                     realised: dict[int, int], n_trials: int = 500,
                     seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    config = STAGE_CONFIGS[stage]
    eligible = predictions.drop_duplicates("player_id")
    eligible = eligible[~eligible["is_eliminated"].astype(bool)]
    eligible = eligible[eligible["status"] == "playing"]
    totals = []
    attempts = 0
    while len(totals) < n_trials and attempts < n_trials * 20:
        attempts += 1
        squad: list[int] = []
        ok = True
        for pos, n in SQUAD_POSITION_COUNTS.items():
            cand = eligible[eligible["position"] == pos.value]
            if len(cand) < n: ok = False; break
            picked = rng.choice(cand["player_id"].values, size=n, replace=False)
            squad.extend(int(x) for x in picked)
        if not ok: continue
        sq = eligible[eligible["player_id"].isin(squad)]
        if float(sq["price_millions"].sum()) > config.budget_millions: continue
        if (sq.groupby("country").size() > config.max_per_country).any(): continue
        # Random 3-4-3 XI from the squad.
        starters = []
        for pos, n in {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3}.items():
            cand = sq[sq["position"] == pos]
            picked = rng.choice(cand["player_id"].values, size=n, replace=False)
            starters.extend(int(x) for x in picked)
        # Random captain.
        cap = int(rng.choice(starters))
        _, _, total = _score_squad(starters, cap, realised)
        totals.append(total)
    arr = np.array(totals)
    return {
        "n_valid_trials": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "p10": float(np.percentile(arr, 10)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
    }


def run() -> dict:
    rows: list[RoundResult] = []
    summary = {"rounds": [], "totals_by_source": {}}

    for round_id, feat_name, raw_name, stage in ROUND_PLAN:
        feat_path = Path("data/processed") / feat_name
        raw_path = Path("data/raw") / raw_name
        if not feat_path.exists() or not raw_path.exists():
            print(f"\nROUND {round_id}: skipping (missing files)")
            continue
        feat = pd.read_parquet(feat_path)
        # These snapshots predate the form feature. Reconstruct form_lag from
        # the point-in-time round_points already stored in the snapshot; it
        # reflects only rounds completed before the snapshot, so it is
        # leak-free for the round being predicted.
        if "form_lag" not in feat.columns:
            feat = _attach_form_lag(feat)
        raw = pd.read_parquet(raw_path)
        realised = _round_points(raw, round_id)
        print(f"\n=== ROUND {round_id} ({stage.value}) ===")

        round_summary: dict = {"round_id": round_id, "stage": stage.value,
                                "backends": []}

        # Per-backend.
        for backend in BACKENDS:
            try:
                preds = _predict(feat.copy(), backend)
                chosen, formation, cap_id, starter_ids = _solve_squad(preds, stage, round_id)
                starter_pts, cap_pts, total = _score_squad(starter_ids, cap_id, realised)
                in_round = preds[preds["round_id"] == round_id]
                pred_map = dict(zip(in_round["player_id"].astype(int),
                                    in_round["predicted_points"].astype(float)))
                predicted = (sum(pred_map.get(pid, 0.0) for pid in starter_ids)
                             + pred_map.get(cap_id, 0.0))
            except Exception as e:
                print(f"  {backend}: FAILED ({type(e).__name__}: {e})")
                continue
            r = RoundResult(
                round_id=round_id, stage=stage.value, backend=backend,
                squad_player_ids=list(chosen), starter_ids=list(starter_ids),
                captain_id=cap_id, formation=formation,
                realised_starter_pts=starter_pts,
                realised_captain_raw=cap_pts,
                realised_total=total,
                predicted_total=round(predicted, 2),
            )
            rows.append(r)
            round_summary["backends"].append(asdict(r))
            print(f"  {backend:<10} squad scored {total} pts "
                  f"(starters {starter_pts} + captain raw {cap_pts}, formation {formation})")

        # User actual.
        user_path = Path("data/user_squads") / f"round_{round_id:02d}.json"
        if user_path.exists():
            us = json.loads(user_path.read_text())
            recorded = us.get("realised_total_pts")
            rescored = None
            if us.get("starter_ids"):
                sp, cp, ut = _score_squad(us["starter_ids"], us["captain_id"],
                                          realised)
                rescored = ut - us.get("transfer_cost_points", 0)
            # The recorded in-app score is the primary record and is
            # already net of transfer hits; the id-based rescore (which
            # cannot see boosters or in-round captain switches) is kept
            # only as a cross-check.
            if recorded is not None:
                net = int(recorded)
            elif rescored is not None:
                net = rescored
            else:
                net = None
            if net is not None:
                round_summary["user_actual"] = {
                    "net_total": net,
                    "transfer_cost_points": us.get("transfer_cost_points", 0),
                    "recorded_total_in_json": recorded,
                    "rescored_total": rescored,
                }
                print(f"  user_actual net {net} "
                      f"(recorded: {recorded}, rescored: {rescored})")

        # Availability-discount A/B on the heuristic backend (MD1-R32):
        # squads solved with and without the discount, both scored on
        # realised points. Feeds the whitepaper/paper availability numbers.
        if round_id <= 4:
            try:
                from fifa_fantasy.optimizer.pipeline import (
                    apply_availability_discount, apply_scouting_bonus,
                )
                base_preds = _predict(feat.copy(), "heuristic")
                probe = apply_availability_discount(
                    apply_scouting_bonus(base_preds.copy()))
                if probe["availability_factor"].nunique() <= 1:
                    # Archived pre-R16 feature snapshots predate the
                    # start_rate_lag column; the counterfactual cannot be
                    # replayed on frozen data. The deployment-time
                    # validation (whitepaper 11g) is the evidence.
                    round_summary["availability_ab"] = {
                        "not_applicable":
                            "archived features predate start_rate_lag"}
                    print("  availability A/B: n/a (archived features "
                          "predate start_rate_lag)")
                    raise _SkipAB
                ab = {}
                for label, use_discount in (("with_discount", True),
                                            ("without_discount", False)):
                    p = apply_scouting_bonus(base_preds.copy())
                    if use_discount:
                        p = apply_availability_discount(p)
                        p["predicted_points"] = p["effective_points"]
                    else:
                        p["predicted_points"] = p["effective_points"]
                    _, _, cap_ab, starters_ab = _solve_squad(p, stage, round_id)
                    _, _, total_ab = _score_squad(starters_ab, cap_ab, realised)
                    ab[label] = int(total_ab)
                round_summary["availability_ab"] = ab
                print(f"  availability A/B (heuristic): with {ab['with_discount']} "
                      f"vs without {ab['without_discount']}")
            except _SkipAB:
                pass
            except Exception as e:
                print(f"  availability A/B FAILED: {e}")

        # Random baseline.
        # Use the heuristic predictions table as the candidate-pool source.
        try:
            preds_h = _predict(feat.copy(), "heuristic")
            base = _random_baseline(preds_h, stage, round_id, realised)
            round_summary["random_baseline"] = base
            print(f"  random_baseline: mean {base['mean']:.1f} "
                  f"median {base['median']:.1f} p10 {base['p10']:.1f} p90 {base['p90']:.1f}")
        except Exception as e:
            print(f"  random_baseline FAILED: {e}")

        summary["rounds"].append(round_summary)

    # Cumulative totals.
    by_source: dict[str, int] = {b: 0 for b in BACKENDS}
    by_source["user_actual_net"] = 0
    by_source["random_baseline_mean"] = 0
    for rs in summary["rounds"]:
        for b in rs["backends"]:
            by_source[b["backend"]] = by_source.get(b["backend"], 0) + int(b["realised_total"])
        if "user_actual" in rs:
            by_source["user_actual_net"] = by_source.get("user_actual_net", 0) + int(rs["user_actual"]["net_total"])
        if "random_baseline" in rs:
            by_source["random_baseline_mean"] = by_source.get("random_baseline_mean", 0) + int(rs["random_baseline"]["mean"])
    summary["totals_by_source"] = by_source

    print("\n=== CUMULATIVE (sum across played rounds) ===")
    for src, total in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {src:<25} {total:>6}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "backtest_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nreport -> {out}")
    return summary


if __name__ == "__main__":
    run()
