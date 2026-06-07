"""Per-stage rules: budget, nationality cap, free transfers, boosters.

All values come from `docs/Fantasy.md`. Free-transfer pre-MD1 is "unlimited"
(initial squad selection); we represent unlimited as `None`.
"""

from __future__ import annotations

from dataclasses import dataclass

from fifa_fantasy.collector.schemas import Stage


@dataclass(frozen=True)
class StageConfig:
    stage: Stage
    budget_millions: float
    max_per_country: int
    free_transfers: int | None  # None = unlimited
    available_boosters: tuple[str, ...]


# Boosters per Fantasy.md §Boosters.
_ALL_KNOCKOUT_BOOSTERS = ("Wildcard", "12th Man", "Maximum Captain",
                          "Qualification Booster", "Mystery Booster")

STAGE_CONFIGS: dict[Stage, StageConfig] = {
    Stage.GROUP_MD1: StageConfig(
        stage=Stage.GROUP_MD1,
        budget_millions=100.0,
        max_per_country=3,
        free_transfers=None,  # pre-tournament unlimited selection
        available_boosters=("12th Man", "Maximum Captain"),  # Wildcard not for MD1
    ),
    Stage.GROUP_MD2: StageConfig(
        stage=Stage.GROUP_MD2,
        budget_millions=100.0,
        max_per_country=3,
        free_transfers=2,  # plus up to 1 rolled from MD1
        available_boosters=("Wildcard", "12th Man", "Maximum Captain"),
    ),
    Stage.GROUP_MD3: StageConfig(
        stage=Stage.GROUP_MD3,
        budget_millions=100.0,
        max_per_country=3,
        free_transfers=2,
        available_boosters=("Wildcard", "12th Man", "Maximum Captain"),
    ),
    Stage.R32: StageConfig(
        stage=Stage.R32,
        budget_millions=105.0,
        max_per_country=3,
        free_transfers=None,  # unlimited entering knockouts
        available_boosters=("12th Man", "Maximum Captain",
                            "Qualification Booster", "Mystery Booster"),
    ),
    Stage.R16: StageConfig(
        stage=Stage.R16,
        budget_millions=105.0,
        max_per_country=4,
        free_transfers=4,
        available_boosters=_ALL_KNOCKOUT_BOOSTERS,
    ),
    Stage.QF: StageConfig(
        stage=Stage.QF,
        budget_millions=105.0,
        max_per_country=5,
        free_transfers=4,
        available_boosters=_ALL_KNOCKOUT_BOOSTERS,
    ),
    Stage.SF: StageConfig(
        stage=Stage.SF,
        budget_millions=105.0,
        max_per_country=6,
        free_transfers=5,
        available_boosters=_ALL_KNOCKOUT_BOOSTERS,
    ),
    Stage.FINAL: StageConfig(
        stage=Stage.FINAL,
        budget_millions=105.0,
        max_per_country=8,
        free_transfers=6,
        available_boosters=_ALL_KNOCKOUT_BOOSTERS,
    ),
}


# How many rounds the squad must last under one selection. Pre-tournament the
# squad covers MD1+MD2+MD3, then transfers happen. After R32 transfers are
# unlimited so each knockout round is solved independently for its own round.
DEFAULT_ROUND_HORIZON: dict[Stage, tuple[int, ...]] = {
    Stage.GROUP_MD1: (1, 2, 3),
    Stage.GROUP_MD2: (2, 3),
    Stage.GROUP_MD3: (3,),
    Stage.R32: (4,),
    Stage.R16: (5,),
    Stage.QF: (6,),
    Stage.SF: (7,),
    Stage.FINAL: (8,),
}
