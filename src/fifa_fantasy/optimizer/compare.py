"""Diff two recommendations.

Used by the `--compare-to` CLI flag to surface what changed between
yesterday's and today's pick - typically driven by ownership shifts
ahead of lockout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RecommendationDiff:
    previous_path: Path
    squad_in: list[int]            # in new, not in previous
    squad_out: list[int]           # in previous, not in new
    starter_in: list[int]
    starter_out: list[int]
    captain_changed: bool
    previous_captain_id: int
    new_captain_id: int
    formation_changed: bool
    previous_formation: str
    new_formation: str
    md1_expected_delta: float      # new - previous


def diff(
    previous_path: Path,
    new_squad_ids: list[int],
    new_starter_ids: list[int],
    new_captain_id: int,
    new_formation: str,
    new_xi_expected: float,
) -> RecommendationDiff:
    prev = json.loads(previous_path.read_text())
    prev_squad = set(prev["squad_player_ids"])
    new_squad = set(new_squad_ids)
    prev_starters = set(prev["lineup"]["starter_ids"])
    new_starters = set(new_starter_ids)

    return RecommendationDiff(
        previous_path=previous_path,
        squad_in=sorted(new_squad - prev_squad),
        squad_out=sorted(prev_squad - new_squad),
        starter_in=sorted(new_starters - prev_starters),
        starter_out=sorted(prev_starters - new_starters),
        captain_changed=(prev["lineup"]["captain_id"] != new_captain_id),
        previous_captain_id=prev["lineup"]["captain_id"],
        new_captain_id=new_captain_id,
        formation_changed=(prev["lineup"]["formation"] != new_formation),
        previous_formation=prev["lineup"]["formation"],
        new_formation=new_formation,
        md1_expected_delta=new_xi_expected - prev["lineup"]["expected_points"],
    )


def render_diff_markdown(d: RecommendationDiff, players: pd.DataFrame) -> str:
    """Markdown section showing the squad/lineup/captain changes."""
    if "player_id" in players.columns and players.index.name != "player_id":
        players = players.set_index("player_id")

    def _row(pid: int) -> str:
        p = players.loc[pid]
        return (f"**{p['full_name']}** ({p['country_abbr']}, "
                f"{p['position']}, ${float(p['price_millions']):.1f}M)")

    if not (d.squad_in or d.squad_out or d.starter_in or d.starter_out
            or d.captain_changed or d.formation_changed):
        return f"\n## Changes from {d.previous_path.name}\n\n_(no changes)_\n"

    lines = [f"\n## Changes from {d.previous_path.name}\n"]
    if d.squad_in or d.squad_out:
        lines.append("### Squad")
        for pid in d.squad_out:
            lines.append(f"- **OUT**: {_row(pid)}")
        for pid in d.squad_in:
            lines.append(f"- **IN**:  {_row(pid)}")
        lines.append("")
    if d.starter_in or d.starter_out:
        lines.append("### Starting XI")
        for pid in d.starter_out:
            lines.append(f"- benched: {_row(pid)}")
        for pid in d.starter_in:
            lines.append(f"- started: {_row(pid)}")
        lines.append("")
    if d.captain_changed:
        lines.append(
            f"### Captain: {_row(d.previous_captain_id)} → {_row(d.new_captain_id)}\n"
        )
    if d.formation_changed:
        lines.append(
            f"### Formation: {d.previous_formation} → {d.new_formation}\n"
        )
    sign = "+" if d.md1_expected_delta >= 0 else ""
    lines.append(f"**Target-round expected delta: {sign}{d.md1_expected_delta:.2f} pts**\n")
    return "\n".join(lines)
