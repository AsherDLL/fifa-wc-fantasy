"""Use the `soccerdata` PyPI library as a parser proxy.

soccerdata wraps the scrapers for FBref, Sofascore, WhoScored, ESPN, etc.
It already handles a lot of the anti-bot work. We use it as a primary
source where it has WC 2026 coverage; fall back to our own StealthClient
parsers when it does not.

CALIBRATION NOTE: soccerdata's WC coverage varies. As of WC 2026, the
FBref backend has tournament fixtures but not always predicted XIs.
Sofascore and FotMob are more current for predicted lineups but
require the user to have set up the library's config (`~/soccerdata/`).
"""
from __future__ import annotations

import pandas as pd

from ..models import RawLineup


SOURCE_NAME = "soccerdata"
SOURCE_CONFIDENCE = 0.75


def fetch_predicted_xis(fixtures: pd.DataFrame) -> list[RawLineup]:
    """Try to fetch predicted XIs for the given WC 2026 fixtures.

    Never wired up: soccerdata's WC-2026 predicted-XI surfaces did not
    materialize during the tournament, so the ESPN parser remained the
    only live source. Kept as the extension point named in the docs;
    returns [] unless soccerdata integration is actually implemented.

    `fixtures` is the per-fixture frame from data/raw/fixtures_*.parquet.
    """
    try:
        import soccerdata as sd  # noqa: F401
    except ImportError:
        return []
    return []
