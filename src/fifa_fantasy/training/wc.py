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
from fifa_fantasy.features.squad import squad_strength

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

    rows = []
    for _, p in players.iterrows():
        rp = p.get("round_points")
        if rp is None or len(rp) == 0:
            continue
        for idx, pts in enumerate(rp):
            round_id = idx + 1  # round_points is 0-indexed; round_id is 1-based
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
                "minutes": None,  # we do not have per-round minutes from the API
                "target": int(pts),
                "source": "wc",
            })
    return pd.DataFrame(rows)
