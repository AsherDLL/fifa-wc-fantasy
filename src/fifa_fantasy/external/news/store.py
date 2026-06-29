"""Persist news articles as parquet rows with a disk budget cap.

One parquet file per day under data/external/news_articles/. Each row
is one article. The collector enforces an overall byte budget by
pruning the oldest day's file before writing a new one when needed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path("data/external/news_articles")


def _today_path(out_dir: Path) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return out_dir / f"news_{today}.parquet"


def append_articles(rows: list[dict], out_dir: Path = DEFAULT_DIR) -> Path:
    """Append article rows to today's parquet, creating it if absent."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _today_path(out_dir)
    new_df = pd.DataFrame(rows)
    if new_df.empty:
        return path
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        # Dedup by URL (keep most recent).
        combined = combined.sort_values("collected_at_utc").drop_duplicates(
            "url", keep="last")
    else:
        combined = new_df
    combined.to_parquet(path, index=False)
    return path


def disk_usage_bytes(out_dir: Path = DEFAULT_DIR) -> int:
    if not out_dir.exists():
        return 0
    return sum(p.stat().st_size for p in out_dir.glob("news_*.parquet"))


def prune_oldest(out_dir: Path = DEFAULT_DIR) -> Path | None:
    """Delete the oldest news_<date>.parquet file. Returns the deleted path."""
    files = sorted(out_dir.glob("news_*.parquet"))
    if not files:
        return None
    oldest = files[0]
    oldest.unlink()
    return oldest


def load_articles(out_dir: Path = DEFAULT_DIR,
                  since_days: float | None = None) -> pd.DataFrame:
    """Concatenate all stored articles. Optional `since_days` filter."""
    if not out_dir.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(out_dir.glob("news_*.parquet")):
        try:
            frames.append(pd.read_parquet(p))
        except (OSError, ValueError):
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if since_days is not None and "collected_at_utc" in df.columns:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=since_days)
        df["collected_at_utc"] = pd.to_datetime(df["collected_at_utc"], utc=True)
        df = df[df["collected_at_utc"] >= cutoff]
    return df
