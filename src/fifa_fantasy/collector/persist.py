"""Persistence: save raw JSON snapshots and normalized Parquet files.

Layout under `data_dir` (default `data/raw/`):

    data/raw/
    ├── players_2026-06-07.parquet      # normalized, one per UTC day
    ├── squads_2026-06-07.parquet
    ├── fixtures_2026-06-07.parquet
    └── raw/
        ├── players_2026-06-07T13-04-22Z.json   # verbatim API payload
        ├── squads_2026-06-07T13-04-22Z.json
        └── rounds_2026-06-07T13-04-22Z.json

The raw JSON is cheap insurance — if the schema changes and parsing breaks,
we still have the original payload to re-parse with an updated `RawPlayer`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

DEFAULT_DATA_DIR = Path("data/raw")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _date_stamp(now: datetime | None = None) -> str:
    return (now or _utc_now()).strftime("%Y-%m-%d")


def _timestamp(now: datetime | None = None) -> str:
    return (now or _utc_now()).strftime("%Y-%m-%dT%H-%M-%SZ")


def save_raw_json(
    payload: Any,
    name: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    now: datetime | None = None,
) -> Path:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name}_{_timestamp(now)}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False))
    return path


def save_parquet(
    records: list[BaseModel],
    name: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    now: datetime | None = None,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{name}_{_date_stamp(now)}.parquet"
    rows = [r.model_dump(mode="json") for r in records]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
