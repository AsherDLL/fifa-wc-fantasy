"""Write the static dashboard to results/index.html.

    python -m fifa_fantasy.web

The snapshot loop serves results/ over HTTP and renders the page fresh on
each request, so the served portal always reflects the latest results and a
manual browser refresh is enough. This CLI writes a static fallback copy
(useful for opening the file directly or committing a snapshot); by default
it emits no auto-refresh tag. Rendering lives in `render.build_html`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .render import build_html

DEFAULT_RESULTS = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser(prog="fifa_fantasy.web")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=None,
                        help="output HTML path (default: <results-dir>/index.html)")
    parser.add_argument("--refresh-seconds", type=int, default=0,
                        help="meta-refresh interval; 0 (default) disables it")
    args = parser.parse_args()

    html, n_recs, n_live = build_html(args.results_dir, args.refresh_seconds)
    out = args.out or (args.results_dir / "index.html")
    out.write_text(html)
    print(f"wrote {out}  ({n_recs} recommendations, {n_live} live reports)")


if __name__ == "__main__":
    main()
