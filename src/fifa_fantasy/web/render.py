"""Build the dashboard HTML from the results/ directory.

Separated from the CLI so the snapshot loop's HTTP server can render the
page fresh on each request: the backend keeps producing results, and a
manual browser refresh always reflects the most recent ones. No polling,
no auto-refresh.

`build_html(results_dir, refresh_seconds=0)` returns the page as a string.
With `refresh_seconds=0` (the default) no meta-refresh tag is emitted.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "templates"

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

# Pitch is drawn top (attack) to bottom (keeper).
LINE_ORDER = ["FWD", "MID", "DEF", "GK"]

# Freshness sources: label, newest-of glob (relative to the project root),
# and the cadence (hours) the loop refreshes them on. Age past 1.5x the
# cadence is a warning, past 3x is stale.
FRESHNESS_SOURCES = [
    ("Squad + fixtures", "data/raw/players_*.parquet", 12.0),
    ("GBM retrain", "data/models/gbm_FWD_mean.txt", 12.0),
    ("Prediction markets", "data/external/prediction_markets/*.jsonl", 3.0),
    ("News feed", "data/external/news_articles/*.parquet", 6.0),
    ("Elo ratings", "data/external/country_elo.csv", 24.0),
]


def _clean(value):
    """NaN and None both render as a blank; everything else passes through."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _list_recommendations(results_dir: Path) -> list[dict]:
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
    return items


def _group_by_stage(items: list[dict]) -> list[dict]:
    """Return {stage, items} groups in canonical round order, newest first."""
    by_stage: dict[str, list[dict]] = {}
    for item in items:
        stage = item.get("stage") or "UNKNOWN"
        by_stage.setdefault(stage, []).append(item)

    groups = []
    for stage in sorted(by_stage.keys(), key=lambda s: STAGE_ORDER.get(s, 99)):
        ordered = sorted(by_stage[stage],
                         key=lambda x: x.get("generated_at_utc", ""),
                         reverse=True)
        groups.append({
            "stage": stage,
            "stage_label": STAGE_LABEL.get(stage, stage),
            "items": ordered,
        })
    return groups


def _augment_featured(rec: dict) -> dict:
    """Split the squad into pitch lines, bench, captain and vice for display."""
    squad = rec.get("squad", [])
    lines = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    bench = []
    captain = vice = None
    for p in squad:
        p = dict(p)
        p["display_name"] = _clean(p.get("known_name")) or p.get("full_name")
        if p.get("role") == "Captain":
            captain = p
        elif p.get("role") == "Vice":
            vice = p
        if p.get("in_starting_xi"):
            lines.setdefault(p.get("position", "?"), []).append(p)
        else:
            bench.append(p)

    def ceiling(p):
        return _clean(p.get("predicted_q90")) or p.get("predicted_points", 0.0)

    for pos in lines:
        lines[pos].sort(key=ceiling, reverse=True)
    bench.sort(key=lambda p: p.get("bench_priority", 99))

    starters_flat = []
    for pos in ["GK", "DEF", "MID", "FWD"]:
        starters_flat.extend(lines[pos])

    return {
        "pitch_lines": [(pos, lines.get(pos, [])) for pos in LINE_ORDER],
        "starters": starters_flat,
        "bench": bench,
        "captain": captain,
        "vice": vice,
    }


def _relative_age(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s ago"
    minutes = seconds / 60
    if minutes < 90:
        return f"{int(minutes)}m ago"
    hours = minutes / 60
    if hours < 36:
        return f"{hours:.1f}h ago"
    return f"{hours / 24:.1f}d ago"


def _freshness(results_dir: Path, now: datetime) -> list[dict]:
    base = results_dir.parent if results_dir.name == "results" else Path.cwd()
    out = []
    for label, pattern, cadence_h in FRESHNESS_SOURCES:
        matches = list(base.glob(pattern))
        if not matches:
            out.append({"label": label, "status": "missing",
                        "age_str": "no data", "when": ""})
            continue
        newest = max(matches, key=lambda p: p.stat().st_mtime)
        mtime = datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc)
        age_s = (now - mtime).total_seconds()
        age_h = age_s / 3600
        if age_h <= cadence_h * 1.5:
            status = "fresh"
        elif age_h <= cadence_h * 3:
            status = "warn"
        else:
            status = "stale"
        out.append({
            "label": label,
            "status": status,
            "age_str": _relative_age(age_s),
            "when": mtime.strftime("%Y-%m-%d %H:%M UTC"),
        })
    return out


def _list_live_reports(results_dir: Path) -> list[dict]:
    items = []
    for path in sorted(results_dir.glob("*.md"), reverse=True):
        if "_live_" not in path.name:
            continue
        items.append({"filename": path.name})
    return items


def build_html(results_dir: Path, refresh_seconds: int = 0) -> tuple[str, int, int]:
    """Render the dashboard. Returns (html, n_recommendations, n_live_reports)."""
    now = datetime.now(timezone.utc)
    recs = _list_recommendations(results_dir)
    grouped = _group_by_stage(recs)

    featured = None
    featured_extra = None
    if grouped:
        top_group = grouped[-1]  # most advanced stage present
        featured = top_group["items"][0]  # newest run of that stage
        featured["stage_label"] = top_group["stage_label"]
        featured_extra = _augment_featured(featured)

    history = []
    for group in reversed(grouped):
        items = [r for r in group["items"] if r is not featured]
        if items:
            history.append({**group, "items": items})

    lives = _list_live_reports(results_dir)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.jinja")
    html = template.render(
        featured=featured,
        f=featured_extra,
        history=history,
        freshness=_freshness(results_dir, now),
        total_recommendations=len(recs),
        n_stages=len(grouped),
        live_reports=lives,
        built_at=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        built_at_iso=now.isoformat(),
        refresh_seconds=refresh_seconds,
    )
    return html, len(recs), len(lives)
