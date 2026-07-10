"""Assemble every number the dashboard and the notebook display.

This is the reconciliation layer the project lacked: recommendations,
backtests, validation reports, walk-forward results, news signals and
market snapshots live in different files with different shapes, and each
page used to invent its own ad hoc reading of them. Every loader here is
a pure read, takes an explicit directory, and returns None or an empty
frame when the artifact does not exist yet, so pages degrade to a
"not yet generated" note instead of crashing the render thread.

`assemble(root)` bundles the JSON-serializable summary the pages consume
(written to results/report/report_data.json by `python -m
fifa_fantasy.report`). The heavier frames (calibration rows, market
history, raw signals) are returned as DataFrames for the figure builders
and the notebook, and only summary statistics of them enter the JSON.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fifa_fantasy.model.ensemble import routing_from_report

STAGE_ORDER = {
    "GROUP_MD1": 1, "GROUP_MD2": 2, "GROUP_MD3": 3,
    "R32": 4, "R16": 5, "QF": 6, "SF": 7, "FINAL": 8,
}

STAGE_LABEL = {
    "GROUP_MD1": "Group Matchday 1", "GROUP_MD2": "Group Matchday 2",
    "GROUP_MD3": "Group Matchday 3", "R32": "Round of 32",
    "R16": "Round of 16", "QF": "Quarter-final", "SF": "Semi-final",
    "FINAL": "Final",
}

# Backends that appear in the backtest and may have recommendation files.
KNOWN_BACKENDS = ("heuristic", "heuristic_v2", "poisson", "gbm",
                  "monte_carlo", "ensemble")

DEFAULT_OFFICIAL_BACKEND = "ensemble"

_MARKET_TITLE_RE = re.compile(r"^Will (.+) win the 2026 FIFA World Cup\?$")
_SIGNAL_FILE_RE = re.compile(r"signals_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})Z")

_SIGNAL_CLASS_ORDER = {"risk": 0, "boost": 1, "info": 2}


def official_backend() -> str:
    """The backend whose squad this desk recommends (env-overridable)."""
    return os.environ.get("OPTIMIZER_BACKEND", DEFAULT_OFFICIAL_BACKEND)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def list_recommendations(results_dir: Path) -> list[dict]:
    """Every parseable *recommendation*.json payload, newest first."""
    items = []
    for path in sorted(results_dir.glob("*.json"), reverse=True):
        if "recommendation" not in path.name:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        payload["__filename__"] = path.name
        payload["__md_filename__"] = path.with_suffix(".md").name
        items.append(payload)
    items.sort(key=lambda r: r.get("generated_at_utc", ""), reverse=True)
    return items


def _top_stage(recs: list[dict]) -> str | None:
    stages = [r.get("stage") for r in recs if r.get("stage") in STAGE_ORDER]
    if not stages:
        return None
    return max(stages, key=lambda s: STAGE_ORDER[s])


def official_recommendation(
    results_dir: Path,
    backend: str | None = None,
) -> tuple[dict | None, bool]:
    """The squad this desk actually recommends, plus a fallback flag.

    Newest run at the most advanced stage present whose model_backend is
    the official backend. When no run of that backend exists at the top
    stage, the newest run of any backend is returned with fallback=True so
    the page can say so out loud instead of silently substituting.
    """
    backend = backend or official_backend()
    recs = list_recommendations(results_dir)
    if not recs:
        return None, False
    stage = _top_stage(recs)
    at_stage = [r for r in recs if r.get("stage") == stage]
    matching = [r for r in at_stage if r.get("model_backend") == backend]
    if matching:
        return matching[0], False
    return at_stage[0], True


def latest_recommendation_per_backend(
    results_dir: Path,
    stage: str | None = None,
) -> dict[str, dict]:
    """Newest recommendation per backend, restricted to one stage.

    Defaults to the most advanced stage present so the algorithm tabs
    compare squads for the same round.
    """
    recs = list_recommendations(results_dir)
    stage = stage or _top_stage(recs)
    out: dict[str, dict] = {}
    for rec in recs:
        if stage and rec.get("stage") != stage:
            continue
        key = rec.get("model_backend")
        if key and key not in out:
            out[key] = rec
    return out


# ---------------------------------------------------------------------------
# Evaluation artifacts
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_backtest(eval_dir: Path) -> dict | None:
    """Per-round and cumulative realized points per backend.

    Returns {rounds: [{round_id, stage}], series: {backend: [pts]},
    cumulative: {backend: [pts]}, totals: {...}} including the user's
    actual net totals and the random-baseline mean as reference series.
    """
    payload = _read_json(eval_dir / "backtest_summary.json")
    if not payload:
        return None
    rounds = []
    series: dict[str, list[float]] = {}
    for rnd in payload.get("rounds", []):
        rounds.append({"round_id": rnd.get("round_id"),
                       "stage": rnd.get("stage"),
                       "stage_label": STAGE_LABEL.get(rnd.get("stage"),
                                                      rnd.get("stage"))})
        for b in rnd.get("backends", []):
            series.setdefault(b["backend"], []).append(
                float(b.get("realised_total", 0.0)))
        user = rnd.get("user_actual") or {}
        series.setdefault("user_actual", []).append(
            float(user.get("net_total", 0.0)))
        rand = rnd.get("random_baseline") or {}
        series.setdefault("random_baseline", []).append(
            float(rand.get("mean", 0.0)))
    cumulative = {k: list(np.cumsum(v)) for k, v in series.items()}
    return {
        "rounds": rounds,
        "series": series,
        "cumulative": cumulative,
        "totals": payload.get("totals_by_source", {}),
    }


def load_validation_report(training_dir: Path) -> list[dict]:
    """Held-out EPL RMSE rows per position, or [] when absent."""
    payload = _read_json(training_dir / "validation_report.json")
    if not payload:
        return []
    return payload.get("position_rmse", [])


def derive_routing(training_dir: Path) -> dict[str, str]:
    """Per-position ensemble routing derived from the live report."""
    return routing_from_report(training_dir / "validation_report.json")


def load_walkforward(eval_dir: Path) -> dict | None:
    """Leak-free walk-forward RMSE artifact, or None until generated."""
    return _read_json(eval_dir / "wc_forward_validation.json")


def load_gk_sweep(eval_dir: Path) -> list[dict]:
    """GK save-bonus v1 vs v2 A/B rows, newest artifact wins."""
    candidates = sorted(eval_dir.glob("gk_formula_ab_*.json"))
    if not candidates:
        return []
    payload = _read_json(candidates[-1])
    if not payload:
        return []
    return payload.get("rows", [])


def load_market_negative(eval_dir: Path) -> dict | None:
    """Benter combiner negative-result artifact."""
    return _read_json(eval_dir / "with_vs_without_market.json")


# ---------------------------------------------------------------------------
# Calibration: pre-match predictions vs realized points
# ---------------------------------------------------------------------------

_CALIB_COLS = ["player_id", "position", "round_id", "predicted_points",
               "predicted_q10", "predicted_q50", "predicted_q90"]


def load_calibration(processed_dir: Path) -> pd.DataFrame:
    """One row per (player, round): the last pre-match prediction and the
    realized points.

    Honesty rules. Predictions are taken only from snapshot files where
    the fixture was still scheduled, so nothing is scored with hindsight;
    per (player, round) the newest such snapshot wins, which is the most
    informed pre-match view. Realized labels come only from the newest
    snapshot's `round_points` array, indexed at round_id - 1; a player
    whose array is shorter than the round did not play it and is dropped,
    never zero-filled. Rows are grouped by the stamped model_backend and
    routed_backend columns; coverage per backend is whatever was actually
    stamped, not a reconstruction.
    """
    files = sorted(processed_dir.glob("predictions_????-??-??.parquet"))
    if not files:
        return pd.DataFrame()

    frames = []
    for path in files:
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if "fixture_status" not in df.columns:
            continue
        pre = df[df["fixture_status"] == "scheduled"].copy()
        if pre.empty:
            continue
        keep = [c for c in _CALIB_COLS if c in pre.columns]
        sub = pre[keep].copy()
        sub["model_backend"] = pre.get(
            "model_backend", pd.Series("", index=pre.index)).fillna("")
        sub["routed_backend"] = pre.get(
            "routed_backend", pd.Series("", index=pre.index)).fillna("")
        sub["snapshot"] = path.stem.replace("predictions_", "")
        frames.append(sub)
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values("snapshot", inplace=True)
    merged = merged.groupby(["player_id", "round_id"], as_index=False).last()

    # Realized labels from the newest snapshot only.
    newest = pd.read_parquet(files[-1])
    labels: dict[int, np.ndarray] = {}
    for pid, arr in zip(newest["player_id"], newest["round_points"]):
        if arr is not None:
            labels[int(pid)] = np.asarray(arr, dtype=float)

    def realized(row) -> float:
        arr = labels.get(int(row["player_id"]))
        k = int(row["round_id"])
        if arr is None or len(arr) < k:
            return math.nan
        return float(arr[k - 1])

    merged["realized"] = merged.apply(realized, axis=1)
    merged = merged.dropna(subset=["realized"]).reset_index(drop=True)
    return merged


def calibration_summary(calib: pd.DataFrame) -> list[dict]:
    """Per (model_backend, position) accuracy of the pre-match predictions."""
    if calib.empty:
        return []
    rows = []
    for (backend, pos), grp in calib.groupby(["model_backend", "position"]):
        y = grp["realized"].to_numpy(dtype=float)
        p = grp["predicted_points"].to_numpy(dtype=float)
        err = y - p
        row = {
            "model_backend": backend or "unlabeled",
            "position": pos,
            "n": int(len(grp)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
        }
        if len(grp) >= 3 and np.std(y) > 0 and np.std(p) > 0:
            row["spearman_rho"] = float(
                pd.Series(y).corr(pd.Series(p), method="spearman"))
        else:
            row["spearman_rho"] = None
        rows.append(row)
    rows.sort(key=lambda r: (r["model_backend"], r["position"]))
    return rows


# ---------------------------------------------------------------------------
# News signals
# ---------------------------------------------------------------------------

def load_signals(signals_dir: Path, days: float = 3.0) -> pd.DataFrame:
    """Recent player-signal rows, deduplicated.

    Snapshot files re-scan a sliding article window, so consecutive files
    overlap heavily; (player_id, url, signal) identifies a finding.
    """
    files = []
    for path in sorted(signals_dir.glob("signals_*.parquet")):
        m = _SIGNAL_FILE_RE.search(path.name)
        if not m:
            continue
        stamp = datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S").replace(
            tzinfo=timezone.utc)
        files.append((stamp, path))
    if not files:
        return pd.DataFrame()
    files.sort()
    cutoff = files[-1][0] - timedelta(days=days)
    recent = [p for stamp, p in files if stamp >= cutoff]
    frames = []
    for path in recent:
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["player_id", "url", "signal"])
    return merged.reset_index(drop=True)


def aggregate_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, signal): mention count plus the newest evidence.

    Sorted risk first, then boost, then info, and by mention count within
    a class, which is the triage order an analyst wants.
    """
    if signals.empty:
        return pd.DataFrame()
    signals = signals.sort_values("published_at_utc")
    grouped = signals.groupby(
        ["player_id", "full_name", "country_abbr", "position",
         "signal", "signal_class"], as_index=False,
    ).agg(
        mentions=("url", "nunique"),
        evidence=("evidence", "last"),
        article_title=("article_title", "last"),
        source_id=("source_id", "last"),
        source_confidence=("source_confidence", "max"),
        published_at_utc=("published_at_utc", "max"),
        url=("url", "last"),
    )
    grouped["class_order"] = grouped["signal_class"].map(
        _SIGNAL_CLASS_ORDER).fillna(9)
    grouped.sort_values(
        ["class_order", "mentions", "published_at_utc"],
        ascending=[True, False, False], inplace=True,
    )
    return grouped.drop(columns="class_order").reset_index(drop=True)


def signals_for_squad(
    aggregated: pd.DataFrame, squad_ids: list[int],
) -> pd.DataFrame:
    """Aggregated signals restricted to the official squad."""
    if aggregated.empty or not squad_ids:
        return pd.DataFrame()
    wanted = {int(i) for i in squad_ids}
    mask = aggregated["player_id"].astype(int).isin(wanted)
    return aggregated[mask].reset_index(drop=True)


def signals_coverage(signals: pd.DataFrame, news_dir: Path) -> dict:
    """Corpus-level context for the intelligence page."""
    articles = 0
    try:
        import pyarrow.parquet as pq
        for path in news_dir.glob("*.parquet"):
            articles += pq.ParquetFile(path).metadata.num_rows
    except Exception:
        articles = 0
    out = {
        "articles_cached": int(articles),
        "signal_rows": int(len(signals)),
        "players_flagged": int(signals["player_id"].nunique()) if not signals.empty else 0,
        "by_class": {},
        "by_signal": {},
    }
    if not signals.empty:
        out["by_class"] = signals["signal_class"].value_counts().to_dict()
        out["by_signal"] = signals["signal"].value_counts().to_dict()
    return out


def signal_volume_by_day(signals: pd.DataFrame) -> pd.DataFrame:
    """Signal rows per publication day and class, for the volume figure."""
    if signals.empty:
        return pd.DataFrame()
    df = signals.copy()
    df["day"] = pd.to_datetime(
        df["published_at_utc"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["day"])
    return (df.groupby(["day", "signal_class"]).size()
            .unstack(fill_value=0).sort_index())


# ---------------------------------------------------------------------------
# Prediction markets
# ---------------------------------------------------------------------------

def load_market_history(
    markets_dir: Path, top_n: int = 8,
) -> pd.DataFrame:
    """Tidy (country, ts, implied_prob, volume_24h) from the jsonl snapshots.

    Null and dust prices are skipped at parse time; the top_n countries are
    ranked by the latest snapshot's yes price.
    """
    files = sorted(markets_dir.glob("snapshot_*.jsonl"))
    if not files:
        return pd.DataFrame()
    rows = []
    for path in files:
        try:
            with path.open() as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    price = obj.get("yes_price")
                    if price is None or price < 0.005:
                        continue
                    m = _MARKET_TITLE_RE.match(obj.get("title") or "")
                    if not m:
                        continue
                    rows.append({
                        "country": m.group(1),
                        "ts": obj.get("snapshot_utc", ""),
                        "implied_prob": float(price),
                        "volume_24h": obj.get("volume_24h"),
                    })
        except OSError:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(
        df["ts"], format="%Y-%m-%dT%H-%M-%SZ", errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    latest_ts = df["ts"].max()
    latest = df[df["ts"] == latest_ts]
    top = (latest.sort_values("implied_prob", ascending=False)
           .head(top_n)["country"].tolist())
    return df[df["country"].isin(top)].reset_index(drop=True)


def market_latest_table(history: pd.DataFrame) -> list[dict]:
    """Current implied probabilities and volume for the odds table."""
    if history.empty:
        return []
    latest_ts = history["ts"].max()
    latest = history[history["ts"] == latest_ts].sort_values(
        "implied_prob", ascending=False)
    return [
        {
            "country": r.country,
            "implied_prob": float(r.implied_prob),
            "volume_24h": float(r.volume_24h) if r.volume_24h is not None
            and not (isinstance(r.volume_24h, float) and math.isnan(r.volume_24h))
            else None,
        }
        for r in latest.itertuples()
    ]


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def assemble(root: Path) -> dict:
    """Everything the pages need, JSON-serializable.

    Heavy frames stay out; figures consume them directly in
    report.__main__ and the notebook re-loads them itself.
    """
    results_dir = root / "results"
    eval_dir = root / "data" / "evaluation"
    training_dir = root / "data" / "training"
    processed_dir = root / "data" / "processed"
    signals_dir = root / "data" / "external" / "player_signals"
    news_dir = root / "data" / "external" / "news_articles"
    markets_dir = root / "data" / "external" / "prediction_markets"

    official, fallback = official_recommendation(results_dir)
    per_backend = latest_recommendation_per_backend(results_dir)
    signals = load_signals(signals_dir)
    calib = load_calibration(processed_dir)
    market_history = load_market_history(markets_dir)

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "official_backend": official_backend(),
        "official_is_fallback": fallback,
        "official_file": official.get("__filename__") if official else None,
        "official_stage": official.get("stage") if official else None,
        "backends_present": sorted(per_backend.keys()),
        "backtest": load_backtest(eval_dir),
        "validation": load_validation_report(training_dir),
        "routing": derive_routing(training_dir),
        "walkforward": load_walkforward(eval_dir),
        "gk_sweep": load_gk_sweep(eval_dir),
        "market_negative": load_market_negative(eval_dir),
        "calibration_summary": calibration_summary(calib),
        "signals_coverage": signals_coverage(signals, news_dir),
        "market_latest": market_latest_table(market_history),
    }
