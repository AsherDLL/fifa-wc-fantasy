"""Generate a static HTML report from results/ JSONs.

    python -m fifa_fantasy.web

Writes `results/index.html`. Open it in a browser; no server needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "templates"
DEFAULT_RESULTS = Path("results")


STAGE_ORDER = {
    "GROUP_MD1": 1, "GROUP_MD2": 2, "GROUP_MD3": 3,
    "R32": 4, "R16": 5, "QF": 6, "SF": 7, "FINAL": 8,
}


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
    """Return a list of {stage, items} groups in canonical round order.

    Within a stage, items are sorted by generated_at_utc descending so the
    most recent recommendation for that stage is at the top.
    """
    by_stage: dict[str, list[dict]] = {}
    for item in items:
        stage = item.get("stage") or "UNKNOWN"
        by_stage.setdefault(stage, []).append(item)

    groups = []
    for stage in sorted(by_stage.keys(),
                        key=lambda s: STAGE_ORDER.get(s, 99)):
        ordered = sorted(
            by_stage[stage],
            key=lambda x: x.get("generated_at_utc", ""),
            reverse=True,
        )
        groups.append({"stage": stage, "items": ordered})
    return groups


def _list_live_reports(results_dir: Path) -> list[dict]:
    items = []
    for path in sorted(results_dir.glob("*.md"), reverse=True):
        if "_live_" not in path.name:
            continue
        items.append({"filename": path.name})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(prog="fifa_fantasy.web")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=None,
                        help="output HTML path (default: <results-dir>/index.html)")
    args = parser.parse_args()

    recs = _list_recommendations(args.results_dir)
    grouped = _group_by_stage(recs)
    lives = _list_live_reports(args.results_dir)
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.jinja")
    html = template.render(
        groups=grouped,
        total_recommendations=len(recs),
        live_reports=lives,
    )

    out = args.out or (args.results_dir / "index.html")
    out.write_text(html)
    print(f"wrote {out}  ({len(recs)} recommendations, {len(lives)} live reports)")


if __name__ == "__main__":
    main()
