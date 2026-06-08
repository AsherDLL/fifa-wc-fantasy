"""Tests for the Phase 5 live decision module."""

from __future__ import annotations

import pandas as pd
import pytest

from fifa_fantasy.live.captain import (
    build_playbook,
    live_recommendation,
)
from fifa_fantasy.live.state import LiveState
from fifa_fantasy.live.subs import recommend_subs


def _player(
    pid: int,
    name: str,
    position: str,
    e: float,
    kickoff: str,
    match_status: str = "pre_match",
    live_points: int = 0,
):
    return {
        "player_id": pid, "full_name": name, "position": position,
        "predicted_points": e, "kickoff": pd.Timestamp(kickoff, tz="UTC"),
        "match_status": match_status, "live_points": live_points,
    }


def _fake_state(players_rows: list[dict], round_id: int = 1) -> LiveState:
    return LiveState(
        round_id=round_id,
        fixtures=pd.DataFrame(),
        players=pd.DataFrame(players_rows),
    )


# ---------------------------------------------------------------------------
# Captain playbook
# ---------------------------------------------------------------------------

def test_playbook_initial_captain_is_top_E_in_earliest_window():
    rows = [
        _player(1, "EarlyLow", "DEF", 4.0, "2026-06-11 18:00"),
        _player(2, "EarlyHigh", "FWD", 5.0, "2026-06-11 18:00"),
        _player(3, "LaterTop", "FWD", 9.0, "2026-06-13 18:00"),
    ]
    pb = build_playbook(_fake_state(rows), starter_ids=[1, 2, 3])
    assert pb.initial_captain_id == 2  # earliest window, top E
    assert pb.initial_expected == pytest.approx(5.0)


def test_playbook_switch_chain_only_to_higher_E():
    rows = [
        _player(1, "A", "FWD", 5.0, "2026-06-11 18:00"),
        _player(2, "B", "FWD", 4.0, "2026-06-12 18:00"),  # lower E, skipped
        _player(3, "C", "FWD", 6.0, "2026-06-13 18:00"),
    ]
    pb = build_playbook(_fake_state(rows), starter_ids=[1, 2, 3])
    assert pb.initial_captain_id == 1
    assert len(pb.steps) == 1
    assert pb.steps[0].to_player_id == 3


def test_playbook_threshold_equals_target_E_not_double():
    rows = [
        _player(1, "Low", "FWD", 4.0, "2026-06-11 18:00"),
        _player(2, "High", "FWD", 8.0, "2026-06-12 18:00"),
    ]
    pb = build_playbook(_fake_state(rows), starter_ids=[1, 2])
    assert pb.steps[0].threshold == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Live captain recommendation
# ---------------------------------------------------------------------------

def test_live_recommendation_switch_when_below_threshold():
    rows = [
        _player(1, "Cap", "FWD", 4.0, "2026-06-11 18:00",
                match_status="completed", live_points=2),
        _player(2, "Alt", "FWD", 8.0, "2026-06-12 18:00"),
    ]
    rec = live_recommendation(_fake_state(rows), starter_ids=[1, 2], current_captain_id=1)
    assert rec.action == "switch"
    assert rec.switch_target_id == 2


def test_live_recommendation_stick_when_above_threshold():
    rows = [
        _player(1, "Cap", "FWD", 4.0, "2026-06-11 18:00",
                match_status="completed", live_points=10),  # already big
        _player(2, "Alt", "FWD", 8.0, "2026-06-12 18:00"),
    ]
    rec = live_recommendation(_fake_state(rows), starter_ids=[1, 2], current_captain_id=1)
    assert rec.action == "stick"


def test_live_recommendation_locked_when_in_progress():
    rows = [
        _player(1, "Cap", "FWD", 4.0, "2026-06-11 18:00",
                match_status="live", live_points=2),
        _player(2, "Alt", "FWD", 8.0, "2026-06-12 18:00"),
    ]
    rec = live_recommendation(_fake_state(rows), starter_ids=[1, 2], current_captain_id=1)
    assert rec.action == "locked"


def test_live_recommendation_stick_when_no_unplayed_alternative():
    rows = [
        _player(1, "Cap", "FWD", 8.0, "2026-06-11 18:00",
                match_status="completed", live_points=4),
        _player(2, "Alt", "FWD", 9.0, "2026-06-12 18:00",
                match_status="completed", live_points=5),
    ]
    rec = live_recommendation(_fake_state(rows), starter_ids=[1, 2], current_captain_id=1)
    assert rec.action == "stick"


# ---------------------------------------------------------------------------
# Sub advisor
# ---------------------------------------------------------------------------

def _full_squad_with(states):
    """Build a complete 2/5/5/3 squad with caller-set states for first GK/DEF/MID/FWD.

    `states` is a dict like {"FWD": {"match_status": "completed", "live_points": 1, "E": 0.0}}
    applied to player_id=1 (FWD). All other players are pre_match defaults.
    """
    counts = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    rows = []
    pid = 0
    for pos, n in counts.items():
        for i in range(n):
            pid += 1
            override = states.get(pos, {}) if i == 0 else {}
            rows.append(_player(
                pid, f"{pos}{i}", pos,
                e=override.get("E", 5.0 + (1 if pos == "FWD" else 0)),
                kickoff="2026-06-11 18:00",
                match_status=override.get("match_status", "pre_match"),
                live_points=override.get("live_points", 0),
            ))
    return rows


def test_subs_no_recommendation_when_nothing_completed():
    rows = _full_squad_with({})
    state = _fake_state(rows)
    advice = recommend_subs(state, starter_ids=list(range(1, 12)),
                            bench_priority_order=list(range(12, 16)))
    assert advice.candidates == []
    assert advice.auto_sub_cancellation_warning is False


def test_subs_recommends_when_finished_starter_underperformed():
    # Build a 15-player squad with explicit player ids. Starting XI in a
    # 3-4-3 formation (1 GK, 3 DEF, 4 MID, 3 FWD). One starting FWD finishes
    # with 0 points; one benched FWD has E=8.0 and has not played.
    rows = [
        # 2 GKs (id 1, 16)
        _player(1, "GK0", "GK", 3.0, "2026-06-11 18:00"),
        _player(16, "GK1", "GK", 2.5, "2026-06-12 18:00"),
        # 5 DEFs (ids 2-6)
        _player(2, "DEF0", "DEF", 4.0, "2026-06-11 18:00"),
        _player(3, "DEF1", "DEF", 4.0, "2026-06-11 18:00"),
        _player(4, "DEF2", "DEF", 4.0, "2026-06-11 18:00"),
        _player(5, "DEF3", "DEF", 3.5, "2026-06-11 18:00"),
        _player(6, "DEF4", "DEF", 3.5, "2026-06-11 18:00"),
        # 5 MIDs (ids 7-11)
        _player(7, "MID0", "MID", 5.0, "2026-06-11 18:00"),
        _player(8, "MID1", "MID", 5.0, "2026-06-11 18:00"),
        _player(9, "MID2", "MID", 5.0, "2026-06-11 18:00"),
        _player(10, "MID3", "MID", 5.0, "2026-06-11 18:00"),
        _player(11, "MID4", "MID", 4.5, "2026-06-11 18:00"),
        # 3 FWDs (ids 12-14): 12 = the underperformer (started but scored 0).
        _player(12, "BadFWD", "FWD", 6.0, "2026-06-11 18:00",
                match_status="completed", live_points=0),
        _player(13, "OkFWD", "FWD", 6.0, "2026-06-11 18:00"),
        _player(14, "OkFWD2", "FWD", 5.5, "2026-06-11 18:00"),
        # Bench FWD with high E and not yet played (id 15).
        _player(15, "GoodBench", "FWD", 8.0, "2026-06-13 18:00"),
    ]
    starter_ids = [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14]  # 1+3+4+3 = 11
    bench_ids = [11, 5, 6, 16, 15]  # last is the high-E FWD bench

    state = _fake_state(rows)
    advice = recommend_subs(state, starter_ids=starter_ids,
                            bench_priority_order=bench_ids)
    fwd_swap = [c for c in advice.candidates
                if c.in_player_id == 15 and c.out_player_id == 12]
    assert fwd_swap, f"expected FWD swap 12->15, got: {advice.candidates}"
    assert advice.auto_sub_cancellation_warning is True
