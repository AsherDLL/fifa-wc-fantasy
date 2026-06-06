"""Tests for FIFA Fantasy WC 2026 scoring rules.

Every official scoring component is exercised, including boundary cases on the
60-minute clean-sheet gate, floor-divided counters (saves/tackles/chances/SoT),
and the scouting bonus's strict inequalities.
"""

import pytest

from fifa_fantasy.scoring import MatchStats, Position, calc_points


# ---------------------------------------------------------------------------
# Appearance points (and 0-minutes short-circuit)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("position", list(Position))
@pytest.mark.parametrize(
    "minutes, expected",
    [(0, 0), (1, 1), (30, 1), (59, 1), (60, 2), (90, 2), (120, 2)],
)
def test_appearance_points(position, minutes, expected):
    # An otherwise-empty stat line: appearance is the only contributor.
    assert calc_points(position, MatchStats(minutes=minutes)) == expected


@pytest.mark.parametrize("position", list(Position))
def test_zero_minutes_short_circuits_everything(position):
    # Player with great stats but 0 minutes → no points at all.
    stats = MatchStats(
        minutes=0, goals=3, assists=2, clean_sheet=True, saves=9, tackles=9,
        chances_created=4, shots_on_target=4, ownership_pct=0.01,
    )
    assert calc_points(position, stats) == 0


# ---------------------------------------------------------------------------
# Position-specific goals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "position, points_per_goal",
    [(Position.GK, 9), (Position.DEF, 7), (Position.MID, 6), (Position.FWD, 5)],
)
@pytest.mark.parametrize("goals", [0, 1, 2, 3])
def test_goal_points_per_position(position, points_per_goal, goals):
    stats = MatchStats(minutes=90, goals=goals)
    expected = 2 + points_per_goal * goals
    assert calc_points(position, stats) == expected


# ---------------------------------------------------------------------------
# Clean sheets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "position, cs_points",
    [(Position.GK, 5), (Position.DEF, 5), (Position.MID, 1), (Position.FWD, 0)],
)
def test_clean_sheet_full_match(position, cs_points):
    stats = MatchStats(minutes=90, clean_sheet=True)
    assert calc_points(position, stats) == 2 + cs_points


@pytest.mark.parametrize("position", list(Position))
def test_clean_sheet_requires_60_minutes(position):
    # 59 minutes → appearance +1, no clean-sheet bonus.
    stats = MatchStats(minutes=59, clean_sheet=True)
    assert calc_points(position, stats) == 1


# ---------------------------------------------------------------------------
# Goals conceded — GK/DEF only, -1 per goal AFTER the first
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("position", [Position.GK, Position.DEF])
@pytest.mark.parametrize(
    "goals_conceded, expected_penalty",
    [(0, 0), (1, 0), (2, -1), (3, -2), (4, -3), (5, -4)],
)
def test_goals_conceded_penalty(position, goals_conceded, expected_penalty):
    stats = MatchStats(minutes=90, goals_conceded=goals_conceded)
    assert calc_points(position, stats) == 2 + expected_penalty


@pytest.mark.parametrize("position", [Position.MID, Position.FWD])
@pytest.mark.parametrize("goals_conceded", [0, 3, 5])
def test_goals_conceded_ignored_for_mid_fwd(position, goals_conceded):
    stats = MatchStats(minutes=90, goals_conceded=goals_conceded)
    assert calc_points(position, stats) == 2


# ---------------------------------------------------------------------------
# Floor-divided counters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "saves, expected_bonus",
    [(0, 0), (1, 0), (2, 0), (3, 1), (5, 1), (6, 2), (9, 3)],
)
def test_gk_saves_every_three(saves, expected_bonus):
    stats = MatchStats(minutes=90, saves=saves)
    assert calc_points(Position.GK, stats) == 2 + expected_bonus


def test_gk_penalty_save():
    stats = MatchStats(minutes=90, penalty_saves=1)
    assert calc_points(Position.GK, stats) == 2 + 3


@pytest.mark.parametrize(
    "tackles, expected_bonus",
    [(0, 0), (2, 0), (3, 1), (5, 1), (6, 2)],
)
def test_mid_tackles_every_three(tackles, expected_bonus):
    stats = MatchStats(minutes=90, tackles=tackles)
    assert calc_points(Position.MID, stats) == 2 + expected_bonus


@pytest.mark.parametrize(
    "chances, expected_bonus",
    [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)],
)
def test_mid_chances_every_two(chances, expected_bonus):
    stats = MatchStats(minutes=90, chances_created=chances)
    assert calc_points(Position.MID, stats) == 2 + expected_bonus


@pytest.mark.parametrize(
    "sot, expected_bonus",
    [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)],
)
def test_fwd_shots_on_target_every_two(sot, expected_bonus):
    stats = MatchStats(minutes=90, shots_on_target=sot)
    assert calc_points(Position.FWD, stats) == 2 + expected_bonus


# ---------------------------------------------------------------------------
# Common (all-position) rules in isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("position", list(Position))
@pytest.mark.parametrize(
    "field, value, delta",
    [
        ("assists", 1, 3),
        ("assists", 2, 6),
        ("yellow_cards", 1, -1),
        ("red_cards", 1, -2),
        ("own_goals", 1, -2),
        ("penalties_won", 1, 2),
        ("penalties_conceded", 1, -1),
        ("free_kick_goals", 1, 1),
    ],
)
def test_common_actions(position, field, value, delta):
    stats = MatchStats(minutes=90, **{field: value})
    assert calc_points(position, stats) == 2 + delta


# ---------------------------------------------------------------------------
# Scouting bonus (strict > 4 points, strict < 5% ownership)
# ---------------------------------------------------------------------------

def test_scouting_bonus_triggers_when_both_conditions_met():
    # FWD: 60 min appearance (+2) + 1 goal (+5) = 7. Owned by 1% → bonus.
    stats = MatchStats(minutes=60, goals=1, ownership_pct=0.01)
    assert calc_points(Position.FWD, stats) == 7 + 2


def test_scouting_bonus_strict_points_threshold():
    # Base = exactly 4 → strict > 4 fails → no bonus.
    # FWD, 90 min (+2), 1 assist (+3), 1 yellow (-1) = 4.
    stats = MatchStats(
        minutes=90, assists=1, yellow_cards=1, ownership_pct=0.01,
    )
    assert calc_points(Position.FWD, stats) == 4


def test_scouting_bonus_strict_ownership_threshold():
    # Ownership = exactly 0.05 → strict < 0.05 fails → no bonus.
    stats = MatchStats(minutes=60, goals=1, ownership_pct=0.05)
    assert calc_points(Position.FWD, stats) == 7


def test_scouting_bonus_high_base_high_ownership_no_bonus():
    stats = MatchStats(minutes=90, goals=2, ownership_pct=0.5)
    assert calc_points(Position.FWD, stats) == 2 + 10


def test_scouting_bonus_low_base_low_ownership_no_bonus():
    # Base = 2 (just appearance), ownership 0.01 → no bonus.
    stats = MatchStats(minutes=90, ownership_pct=0.01)
    assert calc_points(Position.FWD, stats) == 2


# ---------------------------------------------------------------------------
# Realistic end-to-end scenarios (hand-calculated totals)
# ---------------------------------------------------------------------------

def test_scenario_mbappe_brace_and_assist():
    # FWD, 90 min, 2 goals, 1 assist, 40% owned.
    # appearance 2 + goals 10 + assist 3 = 15. No scouting bonus.
    stats = MatchStats(minutes=90, goals=2, assists=1, ownership_pct=0.4)
    assert calc_points(Position.FWD, stats) == 15


def test_scenario_donnarumma_clean_sheet_5_saves_pen_save():
    # GK, 90 min, clean sheet, 5 saves, 1 pen save, 25% owned.
    # appearance 2 + CS 5 + pen save 3 + saves 1 (5//3) = 11.
    stats = MatchStats(
        minutes=90, clean_sheet=True, saves=5, penalty_saves=1, ownership_pct=0.25,
    )
    assert calc_points(Position.GK, stats) == 11


def test_scenario_stones_goal_clean_sheet_low_ownership():
    # DEF, 90 min, clean sheet, 1 goal, 3% owned.
    # appearance 2 + CS 5 + goal 7 = 14. 14 > 4 and 0.03 < 0.05 → +2 bonus = 16.
    stats = MatchStats(
        minutes=90, goals=1, clean_sheet=True, ownership_pct=0.03,
    )
    assert calc_points(Position.DEF, stats) == 16


def test_scenario_midfielder_complete_box_score():
    # MID, 90 min, 1 goal (FK), 1 assist, 6 tackles, 4 chances, 0 cards, 40% owned.
    # appearance 2 + goal 6 + assist 3 + FK bonus 1 + tackles 2 (6//3) + chances 2 (4//2) = 16.
    stats = MatchStats(
        minutes=90, goals=1, assists=1, free_kick_goals=1,
        tackles=6, chances_created=4, ownership_pct=0.4,
    )
    assert calc_points(Position.MID, stats) == 16
