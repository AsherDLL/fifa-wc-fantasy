"""Build a LiveState from the latest Parquet snapshots.

Per-player fields needed for live decisions:

- predicted_points    from data/processed/predictions_<date>.parquet
- live_points         from players.round_points indexed by the current round
- match_status        from the player's squad's fixture in the current round
                      (pre_match / live / completed)
- kickoff             that fixture's UTC kickoff datetime

The "current round" defaults to the lowest-numbered round that still has
at least one fixture not in a completed state. Callers can override.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MatchStatus = str  # one of: "pre_match", "live", "completed"

COMPLETED_STATUSES = {"finished", "completed", "ft", "post_match"}
LIVE_STATUSES = {"in_progress", "live", "during", "first_half", "second_half",
                 "half_time", "extra_time"}


def _normalize_match_status(raw: str | None) -> MatchStatus:
    if raw is None:
        return "pre_match"
    s = str(raw).lower()
    if s in COMPLETED_STATUSES:
        return "completed"
    if s in LIVE_STATUSES:
        return "live"
    return "pre_match"


@dataclass(frozen=True)
class LiveState:
    round_id: int
    fixtures: pd.DataFrame
    players: pd.DataFrame  # squad players in the current round, joined live

    @property
    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)


def _detect_current_round(fixtures: pd.DataFrame) -> int:
    """Lowest round_id with at least one fixture not completed."""
    by_round = fixtures.assign(
        norm=fixtures["status"].map(_normalize_match_status)
    )
    not_done = by_round[by_round["norm"] != "completed"]
    if not_done.empty:
        return int(fixtures["round_id"].max())
    return int(not_done["round_id"].min())


def load_live_state(
    *,
    squad_player_ids: list[int],
    target_round: int | None = None,
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
) -> LiveState:
    """Assemble a LiveState scoped to the given squad and round."""
    players_path = _latest(raw_dir, "players")
    fixtures_path = _latest(raw_dir, "fixtures")
    predictions_path = _latest(processed_dir, "predictions")

    players_full = pd.read_parquet(players_path)
    fixtures = pd.read_parquet(fixtures_path)
    fixtures["kickoff"] = pd.to_datetime(fixtures["kickoff"], utc=True)
    predictions = pd.read_parquet(predictions_path)

    round_id = target_round or _detect_current_round(fixtures)
    round_fixtures = fixtures[fixtures["round_id"] == round_id].copy()
    round_fixtures["status_norm"] = round_fixtures["status"].map(_normalize_match_status)

    squad = players_full[players_full["player_id"].isin(squad_player_ids)].copy()
    if squad.empty:
        raise ValueError(f"no squad players matched in {players_path}")

    # Live points: index into round_points by (round_id - 1), with bounds.
    def _live_pts(rp):
        idx = round_id - 1
        return int(rp[idx]) if 0 <= idx < len(rp) else 0
    squad["live_points"] = squad["round_points"].apply(_live_pts)

    # Join per-player fixture context for the current round.
    home_view = round_fixtures[["fixture_id", "home_squad_id", "kickoff",
                                "status_norm"]] \
        .rename(columns={"home_squad_id": "squad_id"})
    home_view = home_view.assign(is_home=True)
    away_view = round_fixtures[["fixture_id", "away_squad_id", "kickoff",
                                "status_norm"]] \
        .rename(columns={"away_squad_id": "squad_id"})
    away_view = away_view.assign(is_home=False)
    sched = pd.concat([home_view, away_view], ignore_index=True)
    squad = squad.merge(sched, on="squad_id", how="left")
    squad["match_status"] = squad["status_norm"].fillna("pre_match")

    # Bring in predicted_points for the current round.
    preds_round = predictions[predictions["round_id"] == round_id][
        ["player_id", "predicted_points"]
    ]
    squad = squad.merge(preds_round, on="player_id", how="left")
    squad["predicted_points"] = squad["predicted_points"].fillna(0.0)

    return LiveState(
        round_id=round_id,
        fixtures=round_fixtures.sort_values("kickoff").reset_index(drop=True),
        players=squad.reset_index(drop=True),
    )


def _latest(dir_: Path, prefix: str) -> Path:
    matches = sorted(dir_.glob(f"{prefix}_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"no {prefix}_*.parquet under {dir_}")
    return matches[-1]
