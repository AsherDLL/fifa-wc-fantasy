"""Pydantic models for team-news scraping.

Two-stage validation (same pattern as `collector/schemas.py`):

  - RawLineup: what a parser produces (player names, raw strings, no FIFA ids)
  - PredictedXI: canonical normalized form after name matching
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


PlayerStatus = Literal["starting", "bench", "doubtful", "out", "unknown"]


class RawPlayerLineup(BaseModel):
    """One player as scraped (no FIFA id yet)."""
    name: str               # raw name as scraped (may have accents)
    status: PlayerStatus    # parser's classification
    position: str | None = None
    shirt_number: int | None = None


class RawLineup(BaseModel):
    """One scraped lineup record from one source for one fixture."""
    source: str
    scraped_at_utc: datetime
    home_team_name: str
    away_team_name: str
    home_lineup: list[RawPlayerLineup]
    away_lineup: list[RawPlayerLineup]
    confidence: float = 0.5
    raw_text: str | None = None


class PredictedXIPlayer(BaseModel):
    """One player after FIFA id resolution."""
    player_id: int | None    # None if unmatched
    name_scraped: str
    status: PlayerStatus
    match_confidence: float  # 0.0-1.0, how confident in the FIFA id match


class PredictedXI(BaseModel):
    """Canonical per-fixture predicted lineup, FIFA-ids resolved."""
    source: str
    scraped_at_utc: datetime
    fixture_id: int | None             # None if fixture lookup failed
    home_squad_abbr: str
    away_squad_abbr: str
    home_starting_player_ids: list[int]
    away_starting_player_ids: list[int]
    home_bench_player_ids: list[int]
    away_bench_player_ids: list[int]
    home_doubtful_player_ids: list[int] = Field(default_factory=list)
    away_doubtful_player_ids: list[int] = Field(default_factory=list)
    source_confidence: float           # 0.0-1.0 per-source reliability
    unmatched_names: list[str] = Field(default_factory=list)
    raw_text: str | None = None
