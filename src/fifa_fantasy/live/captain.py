"""Captain switching logic.

Two modes share the same EV math:

- Pre-round: no fixture has finished yet. Produce a playbook of (initial
  captain, threshold to switch, alternate, threshold, alternate, ...) by
  chaining starters in kickoff order.
- Live: at least one fixture has finished. Look at the current captain's
  actual points and decide stick or switch.

Rule (from docs/Fantasy.md): a captain can only be switched to a player
whose match has not started, and only when the current captain's match is
not in progress. Switching forfeits the doubling on the current captain.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .state import LiveState


@dataclass(frozen=True)
class SwitchStep:
    window_kickoff: pd.Timestamp
    from_player_id: int
    from_player_name: str
    from_expected: float
    to_player_id: int
    to_player_name: str
    to_expected: float
    threshold: float  # from_player must score below this for the switch to win in EV


@dataclass(frozen=True)
class CaptainPlaybook:
    initial_captain_id: int
    initial_captain_name: str
    initial_expected: float
    steps: list[SwitchStep]
    is_live: bool = False


@dataclass(frozen=True)
class LiveCaptainRecommendation:
    current_captain_id: int
    current_captain_name: str
    captain_match_status: str
    captain_live_points: int
    action: str  # "stick", "switch", "locked"
    switch_target_id: int | None = None
    switch_target_name: str | None = None
    switch_target_expected: float | None = None
    rationale: str = ""


def _starters_in_round(state: LiveState, starter_ids: list[int]) -> pd.DataFrame:
    df = state.players[state.players["player_id"].isin(starter_ids)].copy()
    return df.sort_values(["kickoff", "predicted_points"], ascending=[True, False])


def build_playbook(state: LiveState, starter_ids: list[int]) -> CaptainPlaybook:
    """Greedy pre-round playbook: pick the highest-E starter in the earliest
    kickoff window, then chain switches to any later starter whose E is higher.
    """
    starters = _starters_in_round(state, starter_ids).reset_index(drop=True)
    if starters.empty:
        raise ValueError("no starters in the current round")

    earliest = starters["kickoff"].min()
    first_window = starters[starters["kickoff"] == earliest]
    initial = first_window.loc[first_window["predicted_points"].idxmax()]

    steps: list[SwitchStep] = []
    current = initial
    for _, row in starters.iterrows():
        if row["kickoff"] <= current["kickoff"]:
            continue
        if row["predicted_points"] <= current["predicted_points"]:
            continue
        # The captain bonus is the captain's points counted once extra. Switching
        # exchanges X_current (observed) for E[X_candidate] as the bonus, so the
        # switch wins in EV iff X_current < E[X_candidate].
        steps.append(SwitchStep(
            window_kickoff=row["kickoff"],
            from_player_id=int(current["player_id"]),
            from_player_name=str(current["full_name"]),
            from_expected=float(current["predicted_points"]),
            to_player_id=int(row["player_id"]),
            to_player_name=str(row["full_name"]),
            to_expected=float(row["predicted_points"]),
            threshold=float(row["predicted_points"]),
        ))
        current = row

    return CaptainPlaybook(
        initial_captain_id=int(initial["player_id"]),
        initial_captain_name=str(initial["full_name"]),
        initial_expected=float(initial["predicted_points"]),
        steps=steps,
    )


def live_recommendation(
    state: LiveState,
    starter_ids: list[int],
    current_captain_id: int,
) -> LiveCaptainRecommendation:
    """Apply the switching rule to live state."""
    captain_row = state.players[state.players["player_id"] == current_captain_id]
    if captain_row.empty:
        raise ValueError(f"captain {current_captain_id} not in squad")
    cap = captain_row.iloc[0]
    cap_status = cap["match_status"]
    cap_live = int(cap["live_points"])

    if cap_status == "live":
        return LiveCaptainRecommendation(
            current_captain_id=current_captain_id,
            current_captain_name=str(cap["full_name"]),
            captain_match_status=cap_status,
            captain_live_points=cap_live,
            action="locked",
            rationale="captain's match in progress; switching not allowed",
        )

    starters = _starters_in_round(state, starter_ids)
    unplayed = starters[starters["match_status"] == "pre_match"]
    if cap_status == "completed":
        if unplayed.empty:
            return LiveCaptainRecommendation(
                current_captain_id=current_captain_id,
                current_captain_name=str(cap["full_name"]),
                captain_match_status=cap_status,
                captain_live_points=cap_live,
                action="stick",
                rationale="no unplayed starters left",
            )
        best = unplayed.loc[unplayed["predicted_points"].idxmax()]
        threshold = float(best["predicted_points"])  # E[X_candidate], not 2x
        if cap_live < threshold:
            return LiveCaptainRecommendation(
                current_captain_id=current_captain_id,
                current_captain_name=str(cap["full_name"]),
                captain_match_status=cap_status,
                captain_live_points=cap_live,
                action="switch",
                switch_target_id=int(best["player_id"]),
                switch_target_name=str(best["full_name"]),
                switch_target_expected=float(best["predicted_points"]),
                rationale=(f"captain scored {cap_live}; switch to "
                           f"{best['full_name']} (E={best['predicted_points']:.2f}); "
                           f"captain bonus would improve by "
                           f"{threshold - cap_live:.2f}"),
            )
        return LiveCaptainRecommendation(
            current_captain_id=current_captain_id,
            current_captain_name=str(cap["full_name"]),
            captain_match_status=cap_status,
            captain_live_points=cap_live,
            action="stick",
            rationale=(f"captain scored {cap_live}; best alternate E="
                       f"{best['predicted_points']:.2f}, switching would lose "
                       f"{cap_live - threshold:.2f} in captain bonus"),
        )

    # captain is pre_match -> no live decision yet
    return LiveCaptainRecommendation(
        current_captain_id=current_captain_id,
        current_captain_name=str(cap["full_name"]),
        captain_match_status=cap_status,
        captain_live_points=0,
        action="stick",
        rationale="captain's match has not started yet",
    )
