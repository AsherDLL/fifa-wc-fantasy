"""MILP solvers for squad selection, transfers, and starting XI.

All use PuLP with the bundled CBC backend. Inputs are pandas DataFrames
with player-level effective_points and round-level fixture context.

- `solve_squad`    fresh 15-player selection under stage constraints
- `solve_transfer` 15-player selection given an existing squad + a free
                   transfer quota; pays a configurable hit per extra
                   transfer above the quota
- `solve_lineup`   starting XI + formation + captain pick within a chosen squad
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

# Fantasy.md: -3 points per additional transfer above the free quota.
TRANSFER_HIT_POINTS = 3

# Each formation: (DEF, MID, FWD) - GK is always 1.
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
# Transfer solver (15 players given an existing squad + free transfer quota)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferSolution:
    player_ids: list[int]
    transfers_in: list[int]
    transfers_out: list[int]
    n_transfers: int
    n_extra_transfers: int
    transfer_cost_points: int
    objective: float  # net of the hit
    gross_objective: float  # before subtracting the hit
    budget_used: float


def solve_transfer(
    players: pd.DataFrame,
    current_squad_ids: list[int],
    config: StageConfig,
    rolled_over_transfers: int = 0,
    exclude_eliminated: bool = True,
    verbose: bool = False,
) -> TransferSolution:
    """Pick the best new 15-player squad given the current squad.

    Objective: maximize Σ total_effective_points · x[p]  −  3 · extra
    where `extra` = max(0, transfers_in − (free_transfers + rolled_over)).

    Stages with unlimited free transfers (`config.free_transfers is None`)
    short-circuit to `solve_squad` and report the diff against the current
    squad - same shape of result so the caller doesn't branch.
    """
    if config.free_transfers is None:
        fresh = solve_squad(players, config, exclude_eliminated, verbose)
        return _diff_to_transfer_solution(
            fresh.player_ids,
            current_squad_ids,
            objective_gross=fresh.objective,
            budget_used=fresh.budget_used,
            free_transfers=10**9,  # treat as unlimited
        )

    if exclude_eliminated:
        players = players[~players["is_eliminated"].astype(bool)].reset_index(drop=True)
    if len(players) < SQUAD_SIZE:
        raise ValueError(f"only {len(players)} candidates; need {SQUAD_SIZE}")

    free = config.free_transfers + rolled_over_transfers
    current_set = set(current_squad_ids)

    prob = pulp.LpProblem("transfer", pulp.LpMaximize)
    x = {
        int(r.player_id): pulp.LpVariable(f"x_{int(r.player_id)}", cat="Binary")
        for r in players.itertuples()
    }
    extra = pulp.LpVariable("extra_transfers", lowBound=0, cat="Continuous")

    # Same hard constraints as solve_squad.
    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    for position, count in SQUAD_POSITION_COUNTS.items():
        ids = [int(r.player_id) for r in players.itertuples() if r.position == position.value]
        prob += pulp.lpSum(x[i] for i in ids) == count, f"pos_{position.value}"
    prob += (
        pulp.lpSum(
            x[int(r.player_id)] * float(r.price_millions) for r in players.itertuples()
        )
        <= config.budget_millions,
        "budget",
    )
    for country, group in players.groupby("country", sort=False):
        ids = [int(pid) for pid in group["player_id"]]
        prob += pulp.lpSum(x[i] for i in ids) <= config.max_per_country, f"nat_{country}"

    # Transfers-in = players in new squad that weren't in old squad.
    new_picks = pulp.lpSum(
        x[int(r.player_id)] for r in players.itertuples()
        if int(r.player_id) not in current_set
    )
    prob += new_picks <= free + extra, "transfer_quota"

    prob += (
        pulp.lpSum(
            x[int(r.player_id)] * float(r.total_effective_points)
            for r in players.itertuples()
        )
        - TRANSFER_HIT_POINTS * extra
    )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"transfer solver failed: status={pulp.LpStatus[status]}")

    chosen = sorted(int(pid) for pid, var in x.items() if var.value() > 0.5)
    budget_used = float(
        players[players["player_id"].isin(chosen)]["price_millions"].sum()
    )
    gross = float(
        sum(
            float(r.total_effective_points)
            for r in players.itertuples()
            if int(r.player_id) in set(chosen)
        )
    )
    return _diff_to_transfer_solution(
        chosen,
        current_squad_ids,
        objective_gross=gross,
        budget_used=budget_used,
        free_transfers=free,
    )


def _diff_to_transfer_solution(
    new_squad: list[int],
    current_squad: list[int],
    *,
    objective_gross: float,
    budget_used: float,
    free_transfers: int,
) -> TransferSolution:
    new_set, old_set = set(new_squad), set(current_squad)
    transfers_in = sorted(new_set - old_set)
    transfers_out = sorted(old_set - new_set)
    n_transfers = len(transfers_in)
    n_extra = max(0, n_transfers - free_transfers)
    cost = TRANSFER_HIT_POINTS * n_extra
    return TransferSolution(
        player_ids=new_squad,
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        n_transfers=n_transfers,
        n_extra_transfers=n_extra,
        transfer_cost_points=cost,
        objective=objective_gross - cost,
        gross_objective=objective_gross,
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

    When an `effective_points` column is present (scouting bonus and
    availability discount applied upstream), the XI objective and the
    bench auto-sub priority use it instead of the raw prediction. Squad
    selection already optimizes effective points; without this the XI
    step silently reverted to raw points, so a rotation-risk player the
    discount had penalized could still be started over a nailed teammate
    with a lower raw mean.
    """
    if len(squad_round) != SQUAD_SIZE:
        raise ValueError(f"expected {SQUAD_SIZE} squad rows; got {len(squad_round)}")

    score_col = ("effective_points" if "effective_points" in squad_round.columns
                 else "predicted_points")

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
        y[int(r.player_id)] * float(getattr(r, score_col))
        for r in squad_round.itertuples()
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

    # Auto-sub priority: outfield bench ordered by the same score the XI
    # was chosen on. GK bench (the non-starting GK) is always at end; the
    # game auto-subs GK only with the other GK.
    bench_gk = bench[bench["position"] == "GK"]
    bench_outfield = bench[bench["position"] != "GK"].sort_values(
        score_col, ascending=False
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
