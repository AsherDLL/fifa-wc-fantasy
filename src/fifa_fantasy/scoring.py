"""Canonical FIFA Fantasy World Cup 2026 scoring rules.

Every other module in this project consumes scores from these functions, so they
must match the official rubric exactly. See docs/scoring-rules.md for the source
of truth tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Position(Enum):
    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


@dataclass(frozen=True)
class MatchStats:
    minutes: int
    goals: int = 0
    assists: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0
    penalties_won: int = 0
    penalties_conceded: int = 0
    free_kick_goals: int = 0
    saves: int = 0
    penalty_saves: int = 0
    goals_conceded: int = 0
    clean_sheet: bool = False
    tackles: int = 0
    chances_created: int = 0
    shots_on_target: int = 0
    # Fraction in [0, 1]. Default 1.0 means "owned by everyone" so the scouting
    # bonus never triggers unless ownership is provided explicitly.
    ownership_pct: float = 1.0


SCOUTING_BONUS_POINTS_THRESHOLD = 4
SCOUTING_BONUS_OWNERSHIP_THRESHOLD = 0.05
SCOUTING_BONUS = 2


def _appearance_points(minutes: int) -> int:
    if minutes >= 60:
        return 2
    if minutes >= 1:
        return 1
    return 0


def _common_points(stats: MatchStats) -> int:
    return (
        _appearance_points(stats.minutes)
        + 3 * stats.assists
        - 1 * stats.yellow_cards
        - 2 * stats.red_cards
        - 2 * stats.own_goals
        + 2 * stats.penalties_won
        - 1 * stats.penalties_conceded
        + 1 * stats.free_kick_goals
    )


def _scouting_bonus(base_points: int, ownership_pct: float) -> int:
    if (
        base_points > SCOUTING_BONUS_POINTS_THRESHOLD
        and ownership_pct < SCOUTING_BONUS_OWNERSHIP_THRESHOLD
    ):
        return SCOUTING_BONUS
    return 0


def _gk_base(stats: MatchStats) -> int:
    points = _common_points(stats)
    points += 9 * stats.goals
    if stats.clean_sheet and stats.minutes >= 60:
        points += 5
    points += 3 * stats.penalty_saves
    points += stats.saves // 3
    points -= max(0, stats.goals_conceded - 1)
    return points


def _def_base(stats: MatchStats) -> int:
    points = _common_points(stats)
    points += 7 * stats.goals
    if stats.clean_sheet and stats.minutes >= 60:
        points += 5
    points -= max(0, stats.goals_conceded - 1)
    return points


def _mid_base(stats: MatchStats) -> int:
    points = _common_points(stats)
    points += 6 * stats.goals
    if stats.clean_sheet and stats.minutes >= 60:
        points += 1
    points += stats.tackles // 3
    points += stats.chances_created // 2
    return points


def _fwd_base(stats: MatchStats) -> int:
    points = _common_points(stats)
    points += 5 * stats.goals
    points += stats.shots_on_target // 2
    return points


_DISPATCH = {
    Position.GK: _gk_base,
    Position.DEF: _def_base,
    Position.MID: _mid_base,
    Position.FWD: _fwd_base,
}


def calc_points(position: Position, stats: MatchStats) -> int:
    """Return total fantasy points for a player at a given position.

    Returns 0 immediately if the player did not appear (0 minutes); no other
    scoring component can fire without minutes on the pitch.
    """
    if stats.minutes <= 0:
        return 0
    base = _DISPATCH[position](stats)
    return base + _scouting_bonus(base, stats.ownership_pct)
