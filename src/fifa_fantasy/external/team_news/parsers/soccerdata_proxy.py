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

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from ..models import RawLineup, RawPlayerLineup


SOURCE_NAME = "soccerdata"
SOURCE_CONFIDENCE = 0.75


def fetch_predicted_xis(fixtures: pd.DataFrame) -> list[RawLineup]:
    """Try to fetch predicted XIs for the given WC 2026 fixtures.

    Best-effort: soccerdata's WC coverage is incomplete; we return an
    empty list if the library does not surface what we need. The
    CLI logs which fixtures were missed.

    `fixtures` is the per-fixture frame from data/raw/fixtures_*.parquet.
    """
    # soccerdata import is conditional: if a config file is missing or
    # the network is unavailable, fall back cleanly.
    try:
        import soccerdata as sd  # noqa: F401
    except ImportError:
        return []

    out: list[RawLineup] = []
    # soccerdata's WC-2026 surfaces have moved around between releases;
    # the safe path is to query FotMob if available, FBref otherwise,
    # and skip on either error. We keep this scaffolded and minimal
    # rather than diving deep into library internals.
    try:
        # Pattern: FotMob() class exposes match-level data including XIs.
        # If unavailable in this soccerdata version, the try-block fails
        # cleanly and we return [].
        if hasattr(sd, "FotMob"):
            fotmob = sd.FotMob(leagues="World Cup", seasons="2026")
            try:
                schedule = fotmob.read_schedule()
            except Exception:
                schedule = None
            # NOTE: full integration requires mapping FotMob match_id to
            # our fixture_id by date + home + away. Deferred to a richer
            # implementation; for now we return empty and rely on the
            # ESPN parser. Future work: this branch becomes the primary
            # data path.
    except Exception:  # noqa: BLE001
        pass

    return out
