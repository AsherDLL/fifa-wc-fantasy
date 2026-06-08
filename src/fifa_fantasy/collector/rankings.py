"""Static FIFA Men's World Ranking snapshot loader.

Source: https://www.fifa.com/en/rankings/men
The CSV under data/static/fifa_rankings.csv is hand-maintained; FIFA does
not expose a public JSON endpoint for the ranking, so the file is
refreshed manually when the user wants newer values. The loader is
defensive: an absent file returns an empty DataFrame so the rest of the
pipeline can fall back to the price-only strength proxy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/static/fifa_rankings.csv")


def load_rankings(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Return a DataFrame with columns: country, rank_points, rank_position.

    `rank_position` is computed on load (1 is strongest) so the file only
    needs to maintain `rank_points`.
    """
    if not path.exists():
        return pd.DataFrame(columns=["country", "rank_points", "rank_position"])
    df = pd.read_csv(path, comment="#")
    df = df.dropna(subset=["rank_points"]).copy()
    df["rank_position"] = (
        df["rank_points"].rank(ascending=False, method="min").astype("int64")
    )
    return df.reset_index(drop=True)
