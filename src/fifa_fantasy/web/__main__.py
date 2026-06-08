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
    lives = _list_live_reports(args.results_dir)
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.jinja")
    html = template.render(recommendations=recs, live_reports=lives)

    out = args.out or (args.results_dir / "index.html")
    out.write_text(html)
    print(f"wrote {out}  ({len(recs)} recommendations, {len(lives)} live reports)")


if __name__ == "__main__":
    main()
