"""MILP solvers for squad selection and starting XI.

Both use PuLP with the bundled CBC backend. Inputs are pandas DataFrames
with player-level effective_points and round-level fixture context. The
squad solver returns 15 selected player_ids; the lineup solver returns 11
starters + 4 bench in priority order + captain/vice picks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

from fifa_fantasy.scoring import Position

from .stage_config import StageConfig

SQUAD_SIZE = 15
SQUAD_POSITION_COUNTS: dict[Position, int] = {
    Position.GK: 2,
    Position.DEF: 5,
    Position.MID: 5,
    Position.FWD: 3,
}

# Each formation: (DEF, MID, FWD) — GK is always 1.
VALID_FORMATIONS: dict[str, tuple[int, int, int]] = {
    "4-4-2": (4, 4, 2),
    "4-3-3": (4, 3, 3),
    "4-5-1": (4, 5, 1),
    "3-4-3": (3, 4, 3),
    "3-5-2": (3, 5, 2),
    "5-4-1": (5, 4, 1),
    "5-3-2": (5, 3, 2),
}


# ---------------------------------------------------------------------------
# Squad solver (15 players over a multi-round horizon)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SquadSolution:
    player_ids: list[int]
    objective: float
    budget_used: float


def solve_squad(
    players: pd.DataFrame,
    config: StageConfig,
    exclude_eliminated: bool = True,
    verbose: bool = False,
) -> SquadSolution:
    """Pick the 15-player squad maximizing total_effective_points.

    Required columns on `players`:
        player_id, position, country, price_millions,
        total_effective_points, is_eliminated
    """
    if exclude_eliminated:
        players = players[~players["is_eliminated"].astype(bool)].reset_index(drop=True)
    if len(players) < SQUAD_SIZE:
        raise ValueError(f"only {len(players)} candidates; need {SQUAD_SIZE}")

    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    x = {
        int(row.player_id): pulp.LpVariable(f"x_{int(row.player_id)}", cat="Binary")
        for row in players.itertuples()
    }

    prob += pulp.lpSum(
        x[int(row.player_id)] * float(row.total_effective_points)
        for row in players.itertuples()
    )

    # Exactly 15 players total
    prob += pulp.lpSum(x.values()) == SQUAD_SIZE

    # Position counts
    for position, count in SQUAD_POSITION_COUNTS.items():
        ids = [int(r.player_id) for r in players.itertuples() if r.position == position.value]
        prob += pulp.lpSum(x[i] for i in ids) == count, f"pos_{position.value}"

    # Budget
    prob += (
        pulp.lpSum(
            x[int(r.player_id)] * float(r.price_millions) for r in players.itertuples()
        )
        <= config.budget_millions,
        "budget",
    )

    # Nationality cap
    for country, group in players.groupby("country", sort=False):
        ids = [int(pid) for pid in group["player_id"]]
        prob += pulp.lpSum(x[i] for i in ids) <= config.max_per_country, f"nat_{country}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"squad solver failed: status={pulp.LpStatus[status]}")

    chosen = sorted(int(pid) for pid, var in x.items() if var.value() > 0.5)
    budget_used = float(
        players[players["player_id"].isin(chosen)]["price_millions"].sum()
    )
    return SquadSolution(
        player_ids=chosen,
        objective=float(pulp.value(prob.objective)),
        budget_used=budget_used,
    )


# ---------------------------------------------------------------------------
# Lineup solver (starting XI + formation + captain pick)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineupSolution:
    formation: str
    starter_ids: list[int]
    bench_ids: list[int]  # in auto-sub priority order
    captain_id: int
    vice_captain_id: int
    objective: float


def solve_lineup(squad_round: pd.DataFrame) -> LineupSolution:
    """Pick best XI from the 15-player squad for a single round.

    Required columns on `squad_round`:
        player_id, position, predicted_points
    (predicted_points should be for the target round only; for captain/vice
    we want the round's own scoring potential, not the multi-round total.)
    """
    if len(squad_round) != SQUAD_SIZE:
        raise ValueError(f"expected {SQUAD_SIZE} squad rows; got {len(squad_round)}")

    prob = pulp.LpProblem("lineup", pulp.LpMaximize)
    y = {
        int(r.player_id): pulp.LpVariable(f"y_{int(r.player_id)}", cat="Binary")
        for r in squad_round.itertuples()
    }
    f = {
        name: pulp.LpVariable(f"f_{name}", cat="Binary")
        for name in VALID_FORMATIONS
    }

    prob += pulp.lpSum(
        y[int(r.player_id)] * float(r.predicted_points) for r in squad_round.itertuples()
    )

    # Exactly one formation
    prob += pulp.lpSum(f.values()) == 1, "one_formation"

    # Exactly 11 starters
    prob += pulp.lpSum(y.values()) == 11, "eleven_starters"

    # Exactly 1 GK
    gk_ids = [int(r.player_id) for r in squad_round.itertuples() if r.position == "GK"]
    prob += pulp.lpSum(y[i] for i in gk_ids) == 1, "one_gk"

    # Outfield counts must match the chosen formation
    for pos_name, idx in (("DEF", 0), ("MID", 1), ("FWD", 2)):
        ids = [int(r.player_id) for r in squad_round.itertuples() if r.position == pos_name]
        prob += (
            pulp.lpSum(y[i] for i in ids)
            == pulp.lpSum(f[name] * counts[idx] for name, counts in VALID_FORMATIONS.items()),
            f"{pos_name}_count",
        )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"lineup solver failed: status={pulp.LpStatus[status]}")

    starter_ids = sorted(int(pid) for pid, var in y.items() if var.value() > 0.5)
    formation = next(name for name, var in f.items() if var.value() > 0.5)

    starters = squad_round[squad_round["player_id"].isin(starter_ids)]
    bench = squad_round[~squad_round["player_id"].isin(starter_ids)]

    # Auto-sub priority: outfield bench ordered by predicted_points desc.
    # GK bench (the non-starting GK) is always at end — game auto-subs GK
    # only with the other GK.
    bench_gk = bench[bench["position"] == "GK"]
    bench_outfield = bench[bench["position"] != "GK"].sort_values(
        "predicted_points", ascending=False
    )
    bench_ordered = list(bench_outfield["player_id"].astype(int)) + list(
        bench_gk["player_id"].astype(int)
    )

    starter_sorted = starters.sort_values("predicted_points", ascending=False)
    captain_id = int(starter_sorted.iloc[0]["player_id"])
    vice_captain_id = int(starter_sorted.iloc[1]["player_id"])

    return LineupSolution(
        formation=formation,
        starter_ids=starter_ids,
        bench_ids=bench_ordered,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        objective=float(pulp.value(prob.objective)),
    )
