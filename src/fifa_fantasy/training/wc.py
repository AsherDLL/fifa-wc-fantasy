"""Extract WC player-round labelled rows from collector data.

Once group-stage matches start producing points, the FIFA Fantasy API
populates each player's `round_points` list with one integer per
completed round. This module mines those values plus the matching
fixture context (opponent, home/away, kickoff) and the squad-level
strength signals to produce a training-shape DataFrame that lines up
with the EPL FPL table from `fpl.py`. Combining the two and retraining
the GBM is then a single concat.

Output columns mirror what `training.features.build_training_table`
emits for EPL, so the model trainer does not branch on data source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fifa_fantasy.collector.rankings import load_rankings
from fifa_fantasy.features.build import (
    completed_rounds_by_squad, pad_round_points, team_gc_history,
)
from fifa_fantasy.features.squad import squad_strength
from fifa_fantasy.training.features import FORM_WINDOW

DEFAULT_RAW = Path("data/raw")
DEFAULT_PROCESSED = Path("data/processed")


def _latest(dir_: Path, prefix: str) -> Path:
    matches = sorted(dir_.glob(f"{prefix}_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"no {prefix}_*.parquet under {dir_}")
    return matches[-1]


def extract_wc_training_rows(
    raw_dir: Path = DEFAULT_RAW,
    rankings_path: Path | None = None,
) -> pd.DataFrame:
    """One row per (player, completed round). Empty before MD1 plays out.

    `target` column is the realised fantasy points for that round (from
    the API's `round_points` list).
    """
    players = pd.read_parquet(_latest(raw_dir, "players"))
    squads = pd.read_parquet(_latest(raw_dir, "squads"))
    fixtures = pd.read_parquet(_latest(raw_dir, "fixtures"))
    fixtures["kickoff"] = pd.to_datetime(fixtures["kickoff"], utc=True)
    rankings = load_rankings(rankings_path) if rankings_path else load_rankings()

    strength = squad_strength(players, squads, rankings=rankings)

    # Leak-free trailing team goals-conceded per (squad, round). Used for
    # the team_gc_form label feature; keyed on (squad_id, round_id).
    gc_hist = team_gc_history(fixtures, window=FORM_WINDOW)
    gc_lookup = {
        (int(r["squad_id"]), int(r["round_id"])): r["team_gc_form"]
        for _, r in gc_hist.iterrows()
    }

    # The API truncates round_points at the player's last recorded round,
    # silently dropping trailing DNPs. Pad to the squad's completed-round
    # count so those rounds exist as target=0 rows and so later rounds'
    # lagged features see them; this keeps the training-side definition
    # identical to inference (features.build pads the same way).
    completed_counts = completed_rounds_by_squad(fixtures)

    rows = []
    for _, p in players.iterrows():
        rp = pad_round_points(p.get("round_points"),
                              completed_counts.get(int(p["squad_id"])))
        if len(rp) == 0:
            continue
        for idx, pts in enumerate(rp):
            round_id = idx + 1  # round_points is 0-indexed; round_id is 1-based
            # Lagged form: trailing mean of the player's points over the
            # previous FORM_WINDOW completed rounds. Strictly past (rp[:idx]
            # excludes the current round), so it never leaks this round's
            # label. NaN before the player's first completed round; matches
            # the EPL-side and inference-side definition exactly.
            prior = rp[:idx]
            if len(prior) > 0:
                window_vals = list(prior)[-FORM_WINDOW:]
                form_lag = float(sum(window_vals)) / len(window_vals)
                # Participation proxy: fraction of recent rounds with points>0.
                part_vals = [1.0 if v > 0 else 0.0 for v in window_vals]
                start_rate_lag = float(sum(part_vals)) / len(part_vals)
            else:
                form_lag = float("nan")
                start_rate_lag = float("nan")
            # Find the player's squad's fixture in this round.
            squad_id = int(p["squad_id"])
            fx = fixtures[
                (fixtures["round_id"] == round_id)
                & ((fixtures["home_squad_id"] == squad_id)
                   | (fixtures["away_squad_id"] == squad_id))
            ]
            if fx.empty:
                continue
            fx_row = fx.iloc[0]
            is_home = bool(fx_row["home_squad_id"] == squad_id)
            opp_id = int(fx_row["away_squad_id"] if is_home else fx_row["home_squad_id"])
            strength_row = strength[strength["squad_id"] == squad_id].iloc[0]
            opp_strength_row = strength[strength["squad_id"] == opp_id].iloc[0]
            rows.append({
                "player_id": int(p["player_id"]),
                "position": str(p["position"]),
                "country": str(p["country"]),
                "gameweek": round_id,
                "is_home": is_home,
                "price_millions": float(p["price_millions"]),
                "squad_top_n_avg_price": float(strength_row["squad_top_n_avg_price"]),
                "opp_squad_top_n_avg_price": float(opp_strength_row["squad_top_n_avg_price"]),
                "strength_diff": float(strength_row["squad_top_n_avg_price"]
                                       - opp_strength_row["squad_top_n_avg_price"]),
                "form_lag": form_lag,
                "start_rate_lag": start_rate_lag,
                "team_gc_form": gc_lookup.get((squad_id, round_id), float("nan")),
                "minutes": None,  # we do not have per-round minutes from the API
                "target": int(pts),
                "source": "wc",
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Real-xG trailing form per (country, round) from the community WC
    # dataset; leak-free by construction, NaN wherever not fetched.
    from fifa_fantasy.external.wc2026_dataset import team_xg_form
    xg = team_xg_form(raw_dir=raw_dir)
    if xg.empty:
        df["team_xg_form_real"] = float("nan")
        df["team_xga_form_real"] = float("nan")
        return df
    return df.merge(
        xg.drop(columns=["squad_id"]).rename(columns={"round_id": "gameweek"}),
        on=["country", "gameweek"], how="left")
