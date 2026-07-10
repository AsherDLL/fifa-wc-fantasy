"""Matplotlib figure builders for the dashboard and the notebook.

Every builder takes already-loaded data (from report.data) and an output
path, writes an SVG, and returns the path, or None when the data is not
there yet. The pages inline the SVG contents, so the files must be
self-contained vector output with no external references.

One visual language everywhere: each backend keeps one fixed color in
every figure (Okabe-Ito, colorblind-safe), the user's actual squad is the
black dashed reference, the random baseline is grey dotted. An analyst
learns the encoding once.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BACKEND_COLORS = {
    "heuristic": "#E69F00",
    "heuristic_v2": "#CC79A7",
    "poisson": "#56B4E9",
    "gbm": "#0072B2",
    "monte_carlo": "#D55E00",
    "ensemble": "#009E73",
    "user_actual": "#1b1712",
    "random_baseline": "#8d8579",
}

BACKEND_LABELS = {
    "heuristic": "Heuristic",
    "heuristic_v2": "Heuristic v2",
    "poisson": "Poisson",
    "gbm": "GBM",
    "monte_carlo": "Monte Carlo",
    "ensemble": "Ensemble",
    "user_actual": "User actual (net)",
    "random_baseline": "Random baseline",
}

POSITIONS = ("GK", "DEF", "MID", "FWD")

_INK = "#1b1712"
_INK_SOFT = "#6a6155"
_PANEL = "#fcfaf5"
_GRID = "#e3ddd0"


def apply_style() -> None:
    """Light academic style shared by the dashboard SVGs and the notebook."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "figure.figsize": (8.0, 4.5),
        "figure.dpi": 100,
        "savefig.facecolor": "white",
        "axes.facecolor": _PANEL,
        "axes.edgecolor": _INK_SOFT,
        "axes.labelcolor": _INK,
        "axes.titlecolor": _INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": _GRID,
        "grid.linewidth": 0.7,
        "xtick.color": _INK_SOFT,
        "ytick.color": _INK_SOFT,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "svg.fonttype": "none",
    })


def _save(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _legend_below(ax, ncol: int = 4) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=ncol)


def _series_style(key: str) -> dict:
    style = {"color": BACKEND_COLORS.get(key, _INK_SOFT), "linewidth": 1.8}
    if key == "user_actual":
        style["linestyle"] = "--"
    elif key == "random_baseline":
        style["linestyle"] = ":"
    return style


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def fig_backtest_cumulative(backtest: dict | None, out_path: Path) -> Path | None:
    """Cumulative realized squad points per backend across completed rounds."""
    if not backtest or not backtest.get("rounds"):
        return None
    apply_style()
    stages = [r["stage"] for r in backtest["rounds"]]
    x = np.arange(1, len(stages) + 1)
    fig, ax = plt.subplots()
    for key, series in backtest["cumulative"].items():
        ax.plot(x, series, label=BACKEND_LABELS.get(key, key),
                marker="o", markersize=3.5, **_series_style(key))
    ax.set_xticks(x, stages)
    ax.set_xlabel("Completed round")
    ax.set_ylabel("Cumulative realized points")
    ax.set_title("Realized squad points, cumulative by backend")
    _legend_below(ax)
    return _save(fig, out_path)


def fig_backtest_rounds(backtest: dict | None, out_path: Path) -> Path | None:
    """Per-round realized squad points, grouped bars per backend."""
    if not backtest or not backtest.get("rounds"):
        return None
    apply_style()
    stages = [r["stage"] for r in backtest["rounds"]]
    keys = [k for k in BACKEND_COLORS if k in backtest["series"]]
    x = np.arange(len(stages))
    width = 0.9 / max(len(keys), 1)
    fig, ax = plt.subplots()
    for i, key in enumerate(keys):
        ax.bar(x + (i - (len(keys) - 1) / 2) * width,
               backtest["series"][key], width * 0.92,
               label=BACKEND_LABELS.get(key, key),
               color=BACKEND_COLORS.get(key, _INK_SOFT))
    ax.set_xticks(x, stages)
    ax.set_ylabel("Realized points in round")
    ax.set_title("Realized squad points per round")
    _legend_below(ax)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def fig_holdout_rmse(
    validation_rows: list[dict],
    routing: dict[str, str] | None,
    out_path: Path,
) -> Path | None:
    """Held-out EPL RMSE per position per backend, routing winner marked."""
    if not validation_rows:
        return None
    apply_style()
    backends = ("heuristic", "poisson", "gbm")
    positions = [r["position"] for r in validation_rows]
    x = np.arange(len(positions))
    width = 0.9 / len(backends)
    fig, ax = plt.subplots()
    for i, backend in enumerate(backends):
        vals = [r.get(f"{backend}_rmse") for r in validation_rows]
        ax.bar(x + (i - 1) * width, vals, width * 0.92,
               label=BACKEND_LABELS[backend],
               color=BACKEND_COLORS[backend])
    routing = routing or {}
    for j, row in enumerate(validation_rows):
        winner = routing.get(row["position"])
        if winner in backends:
            i = backends.index(winner)
            val = row.get(f"{winner}_rmse")
            if val:
                ax.annotate("routed", (x[j] + (i - 1) * width, val),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=7.5, color=_INK)
    ax.set_xticks(x, positions)
    ax.set_ylabel("RMSE, held-out EPL 2024-25 GW30-38")
    ax.set_title("Held-out accuracy by position; the ensemble routes each "
                 "position to its winner")
    _legend_below(ax, ncol=3)
    return _save(fig, out_path)


def fig_walkforward(walkforward: dict | None, out_path: Path) -> Path | None:
    """Pooled leak-free walk-forward RMSE per config, per position."""
    if not walkforward or not walkforward.get("pooled"):
        return None
    apply_style()
    pooled = walkforward["pooled"]
    configs = list(pooled.keys())
    cols = list(POSITIONS) + ["ALL"]
    x = np.arange(len(cols))
    width = 0.9 / max(len(configs), 1)
    palette = ["#b8b0a1", "#E69F00", "#0072B2", "#D55E00"]
    fig, ax = plt.subplots()
    for i, cfg in enumerate(configs):
        vals = [pooled[cfg].get(c) for c in cols]
        ax.bar(x + (i - (len(configs) - 1) / 2) * width, vals, width * 0.92,
               label=cfg, color=palette[i % len(palette)])
    ax.set_xticks(x, cols)
    ax.set_ylabel("Pooled RMSE, realized WC rounds")
    ax.set_title("Walk-forward validation on realized World Cup rounds "
                 "(train strictly before each round)")
    _legend_below(ax, ncol=len(configs))
    return _save(fig, out_path)


def fig_calibration(calib: pd.DataFrame, out_path: Path) -> Path | None:
    """Predicted vs realized points, one panel per position."""
    if calib is None or calib.empty:
        return None
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), sharex=False)
    for ax, pos in zip(axes.flat, POSITIONS):
        sub = calib[calib["position"] == pos]
        if sub.empty:
            ax.set_title(f"{pos} (no data)")
            continue
        p = sub["predicted_points"].to_numpy(dtype=float)
        y = sub["realized"].to_numpy(dtype=float)
        # rasterized: thousands of scatter points as SVG elements would put
        # half a megabyte of markup into the inlined page; a raster layer
        # inside the vector figure keeps axes and text crisp and small.
        ax.scatter(p, y, s=9, alpha=0.35, color="#0072B2", edgecolors="none",
                   rasterized=True)
        lim = max(float(np.max(p, initial=1.0)),
                  float(np.max(y, initial=1.0))) * 1.05
        ax.plot([0, lim], [0, lim], color=_INK_SOFT, linewidth=1.0,
                linestyle="--")
        bins = np.linspace(0, max(float(np.max(p, initial=1.0)), 0.5), 8)
        idx = np.digitize(p, bins)
        centers, means = [], []
        for b in range(1, len(bins) + 1):
            mask = idx == b
            if mask.sum() >= 5:
                centers.append(p[mask].mean())
                means.append(y[mask].mean())
        if centers:
            ax.plot(centers, means, color="#D55E00", linewidth=1.8,
                    marker="o", markersize=3.5, label="binned mean")
            ax.legend()
        ax.set_title(f"{pos}  (n={len(sub)})")
        ax.set_xlabel("Predicted points (pre-match)")
        ax.set_ylabel("Realized points")
    fig.suptitle("Pre-match predictions vs realized points, completed rounds",
                 fontsize=11, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_path)


def fig_gk_sweep(rows: list[dict], out_path: Path) -> Path | None:
    """GK save-bonus formula v1 vs v2, GK rows across datasets."""
    gk_rows = [r for r in rows if r.get("position") == "GK"]
    if not gk_rows:
        return None
    apply_style()
    datasets = [r["dataset"] for r in gk_rows]
    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - width / 2, [r["rmse_v1"] for r in gk_rows], width,
           label="v1 flat bonus", color="#b8b0a1")
    ax.bar(x + width / 2, [r["rmse_v2"] for r in gk_rows], width,
           label="v2 opponent-xG scaled (0.50)", color="#56B4E9")
    for j, r in enumerate(gk_rows):
        ax.annotate(f"{r['delta_rmse']:+.3f}", (x[j] + width / 2, r["rmse_v2"]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8, color=_INK)
    ax.set_xticks(x, [d.replace("_", " ") for d in datasets])
    ax.set_ylabel("GK RMSE (n per dataset varies)")
    ax.set_title("Goalkeeper save-bonus formula, v1 vs v2")
    _legend_below(ax, ncol=2)
    return _save(fig, out_path)


def fig_market_negative(payload: dict | None, out_path: Path) -> Path | None:
    """Delta RMSE from blending market prices in; positive means worse."""
    if not payload or not payload.get("rows"):
        return None
    apply_style()
    rows = payload["rows"]
    backends = sorted({r["backend"] for r in rows})
    rounds = sorted({r["round_id"] for r in rows})
    x = np.arange(len(rounds))
    width = 0.9 / max(len(backends), 1)
    fig, ax = plt.subplots()
    for i, backend in enumerate(backends):
        vals = []
        for rnd in rounds:
            match = [r for r in rows
                     if r["backend"] == backend and r["round_id"] == rnd]
            vals.append(match[0]["delta_rmse"] if match else np.nan)
        ax.bar(x + (i - (len(backends) - 1) / 2) * width, vals, width * 0.92,
               label=BACKEND_LABELS.get(backend, backend),
               color=BACKEND_COLORS.get(backend, _INK_SOFT))
    ax.axhline(0, color=_INK_SOFT, linewidth=1.0)
    ax.set_xticks(x, [f"round {r}" for r in rounds])
    ax.set_ylabel("Delta RMSE (combined minus model)")
    ax.set_title("Blending market prices hurt every backend in every round")
    _legend_below(ax, ncol=min(len(backends), 4))
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Intelligence
# ---------------------------------------------------------------------------

def fig_market_odds(history: pd.DataFrame, out_path: Path) -> Path | None:
    """Implied tournament-winner probability over time, top countries."""
    if history is None or history.empty:
        return None
    apply_style()
    fig, ax = plt.subplots()
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
               "#E69F00", "#56B4E9", "#8d8579", "#1b1712"]
    order = (history.groupby("country")["implied_prob"].last()
             .sort_values(ascending=False).index.tolist())
    for i, country in enumerate(order):
        sub = history[history["country"] == country].sort_values("ts")
        ax.plot(sub["ts"], sub["implied_prob"] * 100, label=country,
                color=palette[i % len(palette)], linewidth=1.6)
    ax.set_ylabel("Implied win probability, %")
    ax.set_title("Prediction-market winner odds (Polymarket, 3-hour snapshots)")
    fig.autofmt_xdate(rotation=25)
    _legend_below(ax, ncol=4)
    return _save(fig, out_path)


def fig_signal_volume(volume: pd.DataFrame, out_path: Path) -> Path | None:
    """News signal rows per publication day, stacked by class."""
    if volume is None or volume.empty:
        return None
    apply_style()
    class_colors = {"risk": "#a8620a", "boost": "#2f6b3f", "info": "#56B4E9"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bottom = np.zeros(len(volume))
    x = np.arange(len(volume))
    for cls in ("risk", "boost", "info"):
        if cls not in volume.columns:
            continue
        vals = volume[cls].to_numpy(dtype=float)
        ax.bar(x, vals, 0.7, bottom=bottom, label=cls,
               color=class_colors[cls])
        bottom += vals
    ax.set_xticks(x, [d.strftime("%m-%d") for d in volume.index], rotation=0)
    ax.set_ylabel("Signal rows")
    ax.set_title("Player-signal volume by publication day")
    _legend_below(ax, ncol=3)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Formulas
# ---------------------------------------------------------------------------

def render_formula(lines: tuple[str, ...], out_path: Path) -> Path | None:
    """Typeset mathtext lines to a tight transparent SVG."""
    if not lines:
        return None
    apply_style()
    fig = plt.figure(figsize=(7.0, 0.62 * len(lines)))
    fig.patch.set_alpha(0.0)
    for i, line in enumerate(lines):
        fig.text(0.01, 1.0 - (i + 0.5) / len(lines), line,
                 fontsize=13, color=_INK, va="center")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight",
                transparent=True)
    plt.close(fig)
    return out_path
