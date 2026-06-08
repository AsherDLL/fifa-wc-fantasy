"""Render live decisions as plain ASCII markdown."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .captain import CaptainPlaybook, LiveCaptainRecommendation, SwitchStep
from .state import LiveState
from .subs import SubAdvice


def _fmt_time(t: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M UTC")


def render_round_summary(state: LiveState) -> str:
    rows = ["| Window | Kickoff | Match | Status |", "|---|---|---|---|"]
    for _, fx in state.fixtures.iterrows():
        match = f"{fx.get('home_squad_abbr', '?')} vs {fx.get('away_squad_abbr', '?')}"
        rows.append(
            f"| R{state.round_id} | {_fmt_time(fx['kickoff'])} | {match} | "
            f"{fx['status_norm']} |"
        )
    return f"## Round {state.round_id} fixtures\n\n" + "\n".join(rows) + "\n"


def render_captain(
    pb: CaptainPlaybook,
    live: LiveCaptainRecommendation | None,
) -> str:
    if live is not None:
        if live.action == "switch":
            body = (
                f"Captain switch RECOMMENDED.\n\n"
                f"- Current captain: {live.current_captain_name} "
                f"({live.captain_match_status}, live points {live.captain_live_points}).\n"
                f"- Switch to: {live.switch_target_name} "
                f"(E={live.switch_target_expected:.2f}).\n"
                f"- Rationale: {live.rationale}.\n"
            )
        else:
            body = (
                f"Recommendation: {live.action}.\n\n"
                f"- Captain: {live.current_captain_name} "
                f"({live.captain_match_status}, live points {live.captain_live_points}).\n"
                f"- Rationale: {live.rationale}.\n"
            )
        return "## Captain (live)\n\n" + body

    # Pre-round playbook
    lines = [
        "## Captain playbook",
        "",
        f"Initial captain: **{pb.initial_captain_name}** (E={pb.initial_expected:.2f}).",
        "",
    ]
    if not pb.steps:
        lines.append("No switch alternates with higher expected points; stick throughout.")
        return "\n".join(lines) + "\n"

    lines.append("Switch chain (apply only if the prior captain's actual score falls below the threshold):")
    lines.append("")
    lines.append("| Step | After kickoff | If prior scored below | Switch to | Target E[pts] |")
    lines.append("|---|---|---|---|---|")
    for i, step in enumerate(pb.steps, start=1):
        lines.append(
            f"| {i} | {_fmt_time(step.window_kickoff)} | {step.threshold:.2f} | "
            f"{step.to_player_name} | {step.to_expected:.2f} |"
        )
    return "\n".join(lines) + "\n"


def render_subs(advice: SubAdvice) -> str:
    lines = ["## Manual substitutions"]
    if not advice.candidates:
        lines.append("")
        lines.append("No positive-EV manual sub available. Let auto-subs run.")
        return "\n".join(lines) + "\n"

    lines.append("")
    lines.append("Positive-EV swaps, ranked best first:")
    lines.append("")
    lines.append("| OUT (finished) | Scored | IN (unplayed) | Target E[pts] | Gain |")
    lines.append("|---|---|---|---|---|")
    for c in advice.candidates:
        lines.append(
            f"| {c.out_player_name} | {c.out_actual_points} | {c.in_player_name} "
            f"| {c.in_expected:.2f} | +{c.ev_gain:.2f} |"
        )

    if advice.auto_sub_cancellation_warning:
        lines.append("")
        lines.append("Warning: any manual change cancels ALL auto-subs for this round. "
                     "If any of your unplayed starters DNP, you will not be auto-substituted in. "
                     "Take the swap only if you are confident the remaining starters will play.")
    return "\n".join(lines) + "\n"
