"""Parse raw API payloads into normalized records.

These are pure functions: they take a JSON-like Python object and return
typed `Squad`/`Player`/`Fixture` lists. No I/O, no network. The tests
exercise them with recorded fixtures so they're fast and offline.
"""

from __future__ import annotations

from typing import Any

from fifa_fantasy.scoring import Position

from .schemas import (
    Fixture,
    Player,
    RawFixture,
    RawPlayer,
    RawRound,
    RawSquad,
    ROUND_ID_TO_STAGE,
    Squad,
)


def parse_squads(payload: list[dict[str, Any]]) -> list[Squad]:
    squads = []
    for entry in payload:
        raw = RawSquad.model_validate(entry)
        squads.append(
            Squad(
                squad_id=raw.id,
                name=raw.name,
                abbr=raw.abbr,
                group=raw.group,
                is_eliminated=raw.isEliminated,
            )
        )
    return squads


def _round_points(rp: list[int] | dict[str, int]) -> list[int]:
    # Pre-WC the API returned []. Once the tournament started it switched
    # to {"1": 7, "2": 0, ...}. Normalize to a list ordered by round id so
    # downstream code sees a stable shape.
    if isinstance(rp, dict):
        if not rp:
            return []
        max_round = max(int(k) for k in rp.keys())
        return [int(rp.get(str(i), 0)) for i in range(1, max_round + 1)]
    return list(rp)


def _full_name(raw: RawPlayer) -> str:
    if raw.knownName:
        return raw.knownName
    parts = [raw.firstName]
    if raw.lastName:
        parts.append(raw.lastName)
    return " ".join(parts)


def parse_players(
    payload: list[dict[str, Any]],
    squads: list[Squad],
) -> list[Player]:
    squads_by_id = {s.squad_id: s for s in squads}
    players = []
    for entry in payload:
        raw = RawPlayer.model_validate(entry)
        squad = squads_by_id[raw.squadId]
        players.append(
            Player(
                player_id=raw.id,
                first_name=raw.firstName,
                last_name=raw.lastName,
                known_name=raw.knownName,
                full_name=_full_name(raw),
                position=Position(raw.position),
                squad_id=raw.squadId,
                country=squad.name,
                country_abbr=squad.abbr,
                price_millions=raw.price,
                ownership_fraction=raw.percentSelected / 100.0,
                status=raw.status,
                is_eliminated=squad.is_eliminated,
                one_to_watch=raw.oneToWatch,
                one_to_watch_text=raw.oneToWatchText,
                total_points=raw.stats.totalPoints,
                last_round_points=raw.stats.lastRoundPoints,
                form=raw.stats.form,
                round_points=_round_points(raw.stats.roundPoints),
            )
        )
    return players


def parse_fixtures(payload: list[dict[str, Any]]) -> list[Fixture]:
    fixtures = []
    for round_entry in payload:
        rnd = RawRound.model_validate(round_entry)
        stage = ROUND_ID_TO_STAGE[rnd.id]
        for raw_fix in rnd.tournaments:
            fixtures.append(_fixture_from_raw(raw_fix, rnd.id, stage))
    return fixtures


def _fixture_from_raw(raw: RawFixture, round_id: int, stage) -> Fixture:
    return Fixture(
        fixture_id=raw.id,
        round_id=round_id,
        stage=stage,
        home_squad_id=raw.homeSquadId,
        home_squad_name=raw.homeSquadName,
        home_squad_abbr=raw.homeSquadAbbr,
        away_squad_id=raw.awaySquadId,
        away_squad_name=raw.awaySquadName,
        away_squad_abbr=raw.awaySquadAbbr,
        kickoff=raw.date,
        venue_name=raw.venueName,
        venue_city=raw.venueCity,
        status=raw.status,
        home_score=raw.homeScore,
        away_score=raw.awayScore,
    )
