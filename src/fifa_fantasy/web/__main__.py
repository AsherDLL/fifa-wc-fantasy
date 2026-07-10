"""Write the dashboard pages as static files into results/.

    python -m fifa_fantasy.web [--results-dir DIR] [--page NAME]

The served portal renders on request, so these files are a convenience
snapshot: they let every page open directly over file:// and keep a
committable copy current. Called on the FIFA tick after
`python -m fifa_fantasy.report` has refreshed the dataset and figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .render import PAGES, build_all


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write the static dashboard pages.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="directory holding recommendation files "
                             "(default: results)")
    parser.add_argument("--page", choices=sorted(PAGES),
                        help="write a single page instead of all four")
    args = parser.parse_args()

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.page:
        pages = {args.page: PAGES[args.page](results_dir)}
    else:
        pages = build_all(results_dir)

    for name, html in pages.items():
        out = results_dir / name
        out.write_text(html)
        print(f"written: {out} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
