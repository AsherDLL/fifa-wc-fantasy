"""Manual substitution advisor.

For each starter whose match has finished, find the best unplayed bench
player whose entry preserves a valid formation and beats the finished
starter on expected points. Recommend the highest-EV swap.

Important Fantasy rule (docs/Fantasy.md): any manual change cancels all
automatic substitutions for the round. The recommendation surfaces this
as a warning whenever a manual swap is proposed; v1 does not estimate the
auto-sub forfeit cost numerically because no DNP probabilities exist yet.

Pre-round mode (no finished matches) emits no concrete recommendation but
echoes the current auto-sub bench priority for reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .state import LiveState

# Each formation: (DEF, MID, FWD). GK is always 1. Same table as the optimizer.
VALID_FORMATIONS: dict[str, tuple[int, int, int]] = {
    "4-4-2": (4, 4, 2), "4-3-3": (4, 3, 3), "4-5-1": (4, 5, 1),
    "3-4-3": (3, 4, 3), "3-5-2": (3, 5, 2), "5-4-1": (5, 4, 1), "5-3-2": (5, 3, 2),
}


@dataclass(frozen=True)
class SubCandidate:
    out_player_id: int
    out_player_name: str
    out_actual_points: int
    in_player_id: int
    in_player_name: str
    in_expected: float
    ev_gain: float


@dataclass(frozen=True)
class SubAdvice:
    candidates: list[SubCandidate]  # all positive-EV swaps, descending
    auto_sub_cancellation_warning: bool


def _position_counts(df: pd.DataFrame) -> dict[str, int]:
    return df["position"].value_counts().to_dict()


def _formation_still_valid(after: pd.DataFrame) -> bool:
    counts = _position_counts(after)
    gks, defs, mids, fwds = (counts.get(p, 0) for p in ("GK", "DEF", "MID", "FWD"))
    if gks != 1:
        return False
    return (defs, mids, fwds) in VALID_FORMATIONS.values()


def recommend_subs(
    state: LiveState,
    starter_ids: list[int],
    bench_priority_order: list[int],
) -> SubAdvice:
    """Return the list of viable positive-EV swaps, sorted best-first."""
    players = state.players.set_index("player_id")
    starters = players.loc[starter_ids].copy()
    bench = players.loc[bench_priority_order].copy()

    completed_starters = starters[starters["match_status"] == "completed"]
    unplayed_bench = bench[bench["match_status"] == "pre_match"]

    candidates: list[SubCandidate] = []
    for sid, out_row in completed_starters.iterrows():
        for bid, in_row in unplayed_bench.iterrows():
            new_starters = starters.drop(sid).copy()
            new_starters.loc[bid] = in_row
            if not _formation_still_valid(new_starters):
                continue
            actual_out = int(out_row["live_points"])
            expected_in = float(in_row["predicted_points"])
            gain = expected_in - actual_out
            if gain > 0:
                candidates.append(SubCandidate(
                    out_player_id=int(sid),
                    out_player_name=str(out_row["full_name"]),
                    out_actual_points=actual_out,
                    in_player_id=int(bid),
                    in_player_name=str(in_row["full_name"]),
                    in_expected=expected_in,
                    ev_gain=gain,
                ))
    candidates.sort(key=lambda c: -c.ev_gain)
    return SubAdvice(
        candidates=candidates,
        auto_sub_cancellation_warning=bool(candidates),
    )
