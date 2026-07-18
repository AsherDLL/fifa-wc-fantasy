"""The squad solvers must value the backup GK at autosub-only worth."""
import pandas as pd

from fifa_fantasy.optimizer.solvers import solve_squad, solve_transfer
from fifa_fantasy.optimizer.stage_config import STAGE_CONFIGS
from fifa_fantasy.collector.schemas import Stage


def _pool() -> pd.DataFrame:
    rows = []
    # Two elite GKs and one cheap donkey. Under a squad-sum objective the
    # two elite GKs (5.0 each) win; under the autosub-discounted objective
    # one elite + donkey wins and the freed money buys the better forward.
    rows += [
        dict(player_id=1, position="GK", country="A", price_millions=5.0,
             total_effective_points=5.0),
        dict(player_id=2, position="GK", country="B", price_millions=5.0,
             total_effective_points=4.8),
        dict(player_id=3, position="GK", country="C", price_millions=3.5,
             total_effective_points=1.0),
    ]
    for i in range(5):
        rows.append(dict(player_id=10 + i, position="DEF", country="ABCDE"[i],
                         price_millions=5.0, total_effective_points=3.0))
    for i in range(5):
        rows.append(dict(player_id=20 + i, position="MID", country="ABCDE"[i],
                         price_millions=6.0, total_effective_points=4.0))
    rows += [
        dict(player_id=30, position="FWD", country="A", price_millions=8.0,
             total_effective_points=5.0),
        dict(player_id=31, position="FWD", country="B", price_millions=8.0,
             total_effective_points=5.0),
        # the marginal forward the GK savings should fund: +1.5 over #33
        dict(player_id=32, position="FWD", country="C", price_millions=9.5,
             total_effective_points=6.0),
        dict(player_id=33, position="FWD", country="D", price_millions=8.0,
             total_effective_points=4.5),
    ]
    df = pd.DataFrame(rows)
    df["is_eliminated"] = False
    return df


def test_squad_takes_one_gk_plus_cheapest():
    # At 90.0 both GK setups are affordable, but the elite pair leaves no
    # room for forward 32: only the discounted objective finds the swap.
    config = STAGE_CONFIGS[Stage.FINAL].__class__(
        stage=Stage.FINAL, budget_millions=90.0, max_per_country=8,
        free_transfers=6, available_boosters=())
    sol = solve_squad(_pool(), config)
    assert 1 in sol.player_ids          # elite starter kept
    assert 3 in sol.player_ids          # cheap backup chosen
    assert 2 not in sol.player_ids      # second elite GK rejected
    assert 32 in sol.player_ids         # savings spent on the better FWD


def test_transfer_solver_also_discounts_gk2():
    config = STAGE_CONFIGS[Stage.FINAL].__class__(
        stage=Stage.FINAL, budget_millions=90.0, max_per_country=8,
        free_transfers=6, available_boosters=())
    current = [1, 2, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 33]
    sol = solve_transfer(_pool(), current_squad_ids=current, config=config)
    assert 3 in sol.player_ids and 2 not in sol.player_ids
    assert 32 in sol.player_ids
