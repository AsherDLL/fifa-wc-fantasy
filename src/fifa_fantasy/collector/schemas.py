"""Pydantic schemas for raw API payloads and normalized records.

Two-stage design:

- `Raw*` models mirror the API verbatim (camelCase field names). They catch
  schema drift at the boundary.
- The normalized models (`Squad`, `Player`, `Fixture`) use snake_case names,
  derive convenience fields, and join across the three endpoints so the rest
  of the codebase has a single, stable contract.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict

from fifa_fantasy.scoring import Position


class Stage(str, Enum):
    GROUP_MD1 = "GROUP_MD1"
    GROUP_MD2 = "GROUP_MD2"
    GROUP_MD3 = "GROUP_MD3"
    R32 = "R32"
    R16 = "R16"
    QF = "QF"
    SF = "SF"
    FINAL = "FINAL"


# The FIFA Fantasy API numbers rounds 1–8 in sequence. The mapping below is
# the published tournament structure (MD1, MD2, MD3, R32, R16, QF, SF, Final).
ROUND_ID_TO_STAGE: dict[int, Stage] = {
    1: Stage.GROUP_MD1,
    2: Stage.GROUP_MD2,
    3: Stage.GROUP_MD3,
    4: Stage.R32,
    5: Stage.R16,
    6: Stage.QF,
    7: Stage.SF,
    8: Stage.FINAL,
}


# ---------------------------------------------------------------------------
# Raw payload models (mirror the API)
# ---------------------------------------------------------------------------


class RawSquad(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    abbr: str
    group: str | None = None
    isEliminated: bool


class RawPlayer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    firstName: str
    lastName: str | None = None
    knownName: str | None = None
    squadId: int
    position: str
    price: float
    status: str
    percentSelected: float
    oneToWatch: bool = False
    oneToWatchText: str | None = None


class RawFixture(BaseModel):
    """One match inside a `tournaments` array on a round."""

    model_config = ConfigDict(extra="ignore")

    id: int
    date: datetime
    status: str
    homeSquadId: int
    homeSquadName: str
    homeSquadAbbr: str
    awaySquadId: int
    awaySquadName: str
    awaySquadAbbr: str
    venueName: str | None = None
    venueCity: str | None = None
    homeScore: int | None = None
    awayScore: int | None = None


class RawRound(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    status: str
    startDate: datetime
    endDate: datetime
    tournaments: list[RawFixture]


# ---------------------------------------------------------------------------
# Normalized models (downstream contract)
# ---------------------------------------------------------------------------


class Squad(BaseModel):
    squad_id: int
    name: str
    abbr: str
    group: str | None
    is_eliminated: bool


class Player(BaseModel):
    player_id: int
    first_name: str
    last_name: str | None
    known_name: str | None
    full_name: str
    position: Position
    squad_id: int
    country: str
    country_abbr: str
    price_millions: float
    ownership_fraction: float  # 0.0–1.0 (API returns percent; we divide by 100)
    status: str  # raw API value: "playing", "transferred", …
    is_eliminated: bool
    one_to_watch: bool = False
    one_to_watch_text: str | None = None


class Fixture(BaseModel):
    fixture_id: int
    round_id: int
    stage: Stage
    home_squad_id: int
    home_squad_name: str
    home_squad_abbr: str
    away_squad_id: int
    away_squad_name: str
    away_squad_abbr: str
    kickoff: datetime
    venue_name: str | None
    venue_city: str | None
    status: str
    home_score: int | None
    away_score: int | None
