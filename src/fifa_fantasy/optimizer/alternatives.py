"""Sensitivity / robustness summary.

Pure formatting over the already-solved squad: top-3 captain candidates
with gaps, position depth (squad members ranked by single-round
predicted_points), and a one-line "swap risk" for each starter.
"""

from __future__ import annotations

import pandas as pd

POSITIONS = ("GK", "DEF", "MID", "FWD")


def render_alternatives_markdown(
    squad_round: pd.DataFrame,
    starter_ids: list[int],
    captain_id: int,
) -> str:
    """Markdown sensitivity section.

    `squad_round` has predicted_points for each of the 15 squad players in
    the target round.
    """
    starter_set = set(starter_ids)
    df = squad_round.assign(is_starter=squad_round["player_id"].isin(starter_set))
    starters_sorted = df[df["is_starter"]].sort_values(
        "predicted_points", ascending=False
    )

    # ---- Top 3 captain candidates --------------------------------------
    top_caps = starters_sorted.head(3)
    top1 = top_caps.iloc[0]["predicted_points"]
    captain_rows = []
    for rank, (_, row) in enumerate(top_caps.iterrows(), start=1):
        gap = row["predicted_points"] - top1
        gap_str = "—" if rank == 1 else f"{gap:+.2f}"
        captain_rows.append(
            f"| {rank} | {row['full_name']} | {row['country_abbr']} | "
            f"{row['position']} | {row['predicted_points']:.2f} | {gap_str} |"
        )
    captain_section = (
        "### Top 3 captain candidates\n\n"
        "| Rank | Player | Cty | Pos | E[pts] | Gap to #1 |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(captain_rows)
    )

    # ---- Position depth (in the squad) ---------------------------------
    depth_lines = ["### Squad depth (target-round E[pts])"]
    for pos in POSITIONS:
        pos_rows = df[df["position"] == pos].sort_values(
            "predicted_points", ascending=False
        )
        if pos_rows.empty:
            continue
        parts = []
        for _, row in pos_rows.iterrows():
            marker = "★" if row["is_starter"] else " "
            cap_marker = " (C)" if row["player_id"] == captain_id else ""
            parts.append(
                f"{marker} {row['full_name']} ({row['country_abbr']}, "
                f"{row['predicted_points']:.2f}){cap_marker}"
            )
        depth_lines.append(f"- **{pos}**: " + "; ".join(parts))

    # ---- Starter vs first-on-bench swap risk ---------------------------
    risk_lines = ["### Bench → starter swap risk (lower gap = riskier pick)"]
    for pos in POSITIONS:
        starters = df[(df["position"] == pos) & df["is_starter"]]
        benchers = df[(df["position"] == pos) & ~df["is_starter"]]
        if benchers.empty or starters.empty:
            continue
        bench_top = benchers.sort_values("predicted_points", ascending=False).iloc[0]
        starter_lowest = starters.sort_values("predicted_points").iloc[0]
        gap = starter_lowest["predicted_points"] - bench_top["predicted_points"]
        risk_lines.append(
            f"- **{pos}**: weakest starter "
            f"{starter_lowest['full_name']} ({starter_lowest['predicted_points']:.2f}) "
            f"vs best bench {bench_top['full_name']} "
            f"({bench_top['predicted_points']:.2f}) — gap **{gap:+.2f}**"
        )

    return (
        "\n## Alternatives & sensitivity\n\n"
        + captain_section
        + "\n\n"
        + "\n".join(depth_lines)
        + "\n\n"
        + "\n".join(risk_lines)
        + "\n"
    )
