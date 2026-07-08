"""Tests for the lagged-form feature and the per-position ensemble backend."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fifa_fantasy.model.ensemble import (
    DEFAULT_ROUTING, ensemble_predict, routing_from_report,
)
from fifa_fantasy.training.features import add_lagged_form


def test_add_lagged_form_is_leak_free_and_trailing():
    # Two players, one season, 4 gameweeks each. Points ascend so the
    # trailing mean is easy to check by hand.
    df = pd.DataFrame({
        "season": ["2024-25"] * 8,
        "player_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "gameweek": [1, 2, 3, 4, 1, 2, 3, 4],
        "total_points": [2, 4, 6, 8, 10, 0, 0, 0],
    })
    out = add_lagged_form(df, window=3).sort_values(["player_id", "gameweek"])
    p1 = out[out["player_id"] == 1]["form_lag"].tolist()
    # GW1: no prior -> NaN. GW2: mean(2)=2. GW3: mean(2,4)=3. GW4: mean(2,4,6)=4.
    assert np.isnan(p1[0])
    assert p1[1:] == [2.0, 3.0, 4.0]
    # The current row's own points never enter its own feature (leak-free):
    # player 1 GW4 label is 8 but form_lag is 4 (mean of 2,4,6).
    assert p1[3] == 4.0


def test_add_lagged_form_does_not_bleed_across_players():
    df = pd.DataFrame({
        "season": ["2024-25"] * 4,
        "player_id": [1, 1, 2, 2],
        "gameweek": [1, 2, 1, 2],
        "total_points": [10, 10, 0, 0],
    })
    out = add_lagged_form(df, window=3).sort_values(["player_id", "gameweek"])
    # Player 2's first row must be NaN, not influenced by player 1's 10s.
    p2 = out[out["player_id"] == 2]["form_lag"].tolist()
    assert np.isnan(p2[0])
    assert p2[1] == 0.0


def test_add_lagged_form_is_idempotent():
    df = pd.DataFrame({
        "season": ["2024-25"] * 3,
        "player_id": [1, 1, 1],
        "gameweek": [1, 2, 3],
        "total_points": [2, 4, 6],
    })
    once = add_lagged_form(df, window=3)
    twice = add_lagged_form(once, window=3)
    # Second call is a no-op: the column is preserved bit-for-bit.
    pd.testing.assert_series_equal(once["form_lag"], twice["form_lag"])


def _toy_features() -> pd.DataFrame:
    positions = ["GK", "DEF", "MID", "FWD"]
    rows = []
    for i, pos in enumerate(positions):
        rows.append({
            "player_id": i,
            "position": pos,
            "price_millions": 5.0 + i,
            "is_home": True,
            "strength_diff": 1.0,
            "squad_top_n_avg_price": 6.0,
            "opp_squad_top_n_avg_price": 5.0,
            "rank_diff": np.nan,
            "ownership_fraction": 0.10,
            "status": "playing",
            "is_eliminated": False,
            "total_points": 20,
            "round_points": [4, 5, 6],
            "form_lag": 5.0,
        })
    return pd.DataFrame(rows)


def test_ensemble_routes_each_position_to_its_backend():
    feats = _toy_features()
    out = ensemble_predict(feats)
    routed = dict(zip(out["position"], out["routed_backend"]))
    assert routed == DEFAULT_ROUTING
    # Every routed row got a finite, non-negative prediction.
    assert (out["predicted_points"] >= 0).all()
    assert out["predicted_points"].notna().all()


def test_ensemble_matches_component_backends_per_position():
    from fifa_fantasy.model.baseline import heuristic_predict
    from fifa_fantasy.model.poisson import poisson_predict

    feats = _toy_features()
    ens = ensemble_predict(feats)
    heur = heuristic_predict(feats)
    pois = poisson_predict(feats)

    # DEF routed to heuristic; GK routed to poisson. The ensemble value for
    # those positions must equal the component backend value exactly.
    def val(df, pos):
        return float(df[df["position"] == pos]["predicted_points"].iloc[0])

    assert val(ens, "DEF") == val(heur, "DEF")
    assert val(ens, "GK") == val(pois, "GK")


def test_availability_discount_scales_by_participation():
    from fifa_fantasy.optimizer.pipeline import (
        apply_availability_discount, apply_scouting_bonus, availability_factor,
    )
    df = pd.DataFrame({
        "player_id": [1, 2, 3],
        "predicted_points": [6.0, 6.0, 6.0],
        "ownership_fraction": [0.10, 0.10, 0.10],
        "start_rate_lag": [1.0, 0.0, np.nan],
    })
    out = apply_availability_discount(apply_scouting_bonus(df), floor=0.5)
    # Ever-present player unchanged; never-plays halved. Player 3 is NaN in
    # a frame where others HAVE history (tournament underway), so NaN means
    # "never took the pitch" and is floored too, not given a free pass.
    eff = dict(zip(out["player_id"], out["effective_points"]))
    assert eff[1] == 6.0                       # factor 1.0
    assert eff[2] == 3.0                       # factor 0.5 (floor)
    assert eff[3] == 3.0                       # NaN with history present -> floor
    assert availability_factor(0.5, floor=0.5) == 0.75
    assert bool(out.loc[out.player_id == 2, "rotation_risk"].iloc[0]) is True
    assert bool(out.loc[out.player_id == 1, "rotation_risk"].iloc[0]) is False


def test_availability_discount_nan_is_neutral_pre_tournament():
    from fifa_fantasy.optimizer.pipeline import (
        apply_availability_discount, apply_scouting_bonus,
    )
    # Before MD1 nobody has history: every start_rate_lag is NaN and the
    # discount must make no availability claim (factor 1.0 for everyone).
    df = pd.DataFrame({
        "player_id": [1, 2],
        "predicted_points": [6.0, 4.0],
        "ownership_fraction": [0.10, 0.10],
        "start_rate_lag": [np.nan, np.nan],
    })
    out = apply_availability_discount(apply_scouting_bonus(df), floor=0.5)
    assert (out["availability_factor"] == 1.0).all()
    assert (out["effective_points"] == out["predicted_points"]).all()
    assert not out["rotation_risk"].any()


def test_captain_ceiling_term_prefers_high_q90_when_chasing():
    from fifa_fantasy.optimizer.captain import select_captain_vice
    # Two XI players: A has higher mean, B has a much higher ceiling (q90).
    xi = pd.DataFrame({
        "player_id": [1, 2],
        "full_name": ["A steady", "B explosive"],
        "country_abbr": ["ARG", "FRA"],
        "position": ["MID", "FWD"],
        "opponent_abbr": ["EGY", "PAR"],
        "is_home": [True, True],
        "predicted_points": [6.0, 5.0],
        "predicted_p10": [4.0, 0.5],
        "predicted_p90": [7.0, 14.0],
        "ownership_pct": [40.0, 20.0],
        "country_elo_diff": [100.0, 300.0],
        "fixture_id": [10, 20],
    })
    # Leader (chase=0) captains the steady mean; last place (chase=1) the ceiling.
    lead = select_captain_vice(xi, standings_pos_pct=0.0)
    chase = select_captain_vice(xi, standings_pos_pct=1.0)
    assert lead.captain_id == 1
    assert chase.captain_id == 2


def test_captain_ignores_nan_quantiles_from_ensemble_rows():
    from fifa_fantasy.optimizer.captain import select_captain_vice
    # Ensemble frames carry the quantile COLUMNS for every row but only
    # gbm-routed positions have values; GK/DEF hold NaN. A NaN ceiling must
    # degrade to the mean, never poison the sort (this once made a 4.7-mean
    # goalkeeper captain over an 11-mean forward).
    xi = pd.DataFrame({
        "player_id": [1, 2, 3],
        "full_name": ["GK nan-q", "DEF nan-q", "FWD gbm"],
        "country_abbr": ["ARG", "ESP", "FRA"],
        "position": ["GK", "DEF", "FWD"],
        "opponent_abbr": ["SUI", "BEL", "MAR"],
        "is_home": [True, True, True],
        "predicted_points": [4.7, 4.0, 11.0],
        "predicted_p10": [np.nan, np.nan, 1.5],
        "predicted_p90": [np.nan, np.nan, 14.7],
        "ownership_pct": [35.0, 15.0, 69.0],
        "country_elo_diff": [np.nan, 100.0, 300.0],
        "fixture_id": [10, 20, 30],
    })
    decision = select_captain_vice(xi, standings_pos_pct=0.9)
    assert decision.captain_id == 3
    # Every composite score must be finite: NaN keys make sort order
    # undefined and the winner arbitrary.
    assert all(np.isfinite(c["composite_score"])
               for c in decision.candidates_ranked)


def test_routing_from_report_picks_min_rmse(tmp_path):
    report = tmp_path / "validation_report.json"
    report.write_text(
        '{"position_rmse": ['
        '{"position": "GK", "heuristic_rmse": 2.7, "poisson_rmse": 2.5, "gbm_rmse": 2.6},'
        '{"position": "FWD", "heuristic_rmse": 3.5, "poisson_rmse": 4.5, "gbm_rmse": 3.1}'
        ']}'
    )
    routing = routing_from_report(report)
    assert routing["GK"] == "poisson"    # 2.5 is the min
    assert routing["FWD"] == "gbm"       # 3.1 is the min
    # A position absent from the report keeps its default.
    assert routing["DEF"] == DEFAULT_ROUTING["DEF"]
