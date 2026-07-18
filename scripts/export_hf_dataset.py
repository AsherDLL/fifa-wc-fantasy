"""Assemble the Hugging Face dataset release under dist/hf-dataset/.

Copies ONLY data this project collected or produced itself:
  raw/              daily FIFA Fantasy API snapshots (players, squads,
                    fixtures parquets)
  markets/          Polymarket + Kalshi prediction-market snapshots
  processed/        derived feature and prediction parquets
  evaluation/       analysis artifacts (validation, backtests, round
                    predictions)
  recommendations/  per-run optimizer outputs (json + md)

Third-party upstream data (vaastav FPL mirror, football-data.co.uk,
Elo sources, the mominullptr WC dataset) is deliberately NOT included;
consumers can fetch those from their own homes, catalogued in
docs/data-provenance.md.

This script only stages files and writes the dataset card. Publishing
is a separate, post-final, manual step: see docs/hf-release.md.

Usage:
    python scripts/export_hf_dataset.py [--out dist/hf-dataset]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GROUPS = {
    "raw": ["data/raw/players_*.parquet", "data/raw/squads_*.parquet",
            "data/raw/fixtures_*.parquet"],
    "markets": ["data/external/prediction_markets/snapshot_*.jsonl"],
    "processed": ["data/processed/features_*.parquet",
                  "data/processed/predictions_*.parquet"],
    "evaluation": ["data/evaluation/*.json"],
    "recommendations": ["results/*_recommendation_*.json",
                        "results/*_recommendation_*.md"],
}

CARD = """\
---
license: cc-by-4.0
pretty_name: FIFA World Cup 2026 Fantasy Decision-Support Dataset
tags:
- football
- soccer
- fantasy-sports
- world-cup-2026
- prediction-markets
size_categories:
- n<1K
---

# FIFA World Cup 2026 Fantasy Decision-Support Dataset

Data collected and produced by the open-source decision-support system
[AsherDLL/fifa-wc-fantasy](https://github.com/AsherDLL/fifa-wc-fantasy)
while playing the official FIFA World Cup 2026 Fantasy game live through
the tournament (11 June - 19 July 2026).

## Contents

| Directory | Contents | Cadence |
|---|---|---|
| `raw/` | Snapshots of the FIFA Fantasy public API: full player pool with prices, ownership and per-round points (`players_*.parquet`), the 48 squads (`squads_*.parquet`), and all fixtures with scores (`fixtures_*.parquet`) | daily |
| `markets/` | Polymarket and Kalshi contract snapshots for World Cup markets (`snapshot_*.jsonl`) | every 3 hours (late tournament) |
| `processed/` | Per-(player, round) feature tables and model predictions produced by the system's pipeline | daily |
| `evaluation/` | Model-evaluation artifacts: held-out and walk-forward validation, per-round cross-model backtests, one-off analyses, round-8 match predictions | per analysis |
| `recommendations/` | Every optimizer run: recommended 15-player squad, starting XI, captain, transfers, in JSON and human-readable markdown | per run |

The pipeline that produced every file - collectors, feature builders,
four predictor backends, MILP optimizer - is GPL-3.0 code in the GitHub
repository, and each artifact names the backend, stage and UTC timestamp
that produced it.

## Provenance and licensing

- The dataset (this repository) is CC-BY-4.0.
- The system's source code is GPL-3.0 (repository above); code is not
  included here.
- Raw snapshots record factual game-state data (prices, points,
  ownership percentages, fixtures) from the public, unauthenticated
  FIFA Fantasy JSON endpoints. No account data is involved.
- Third-party upstream datasets consumed by the system (the vaastav FPL
  mirror, football-data.co.uk, Elo sources, the community WC-2026 stats
  dataset) are NOT redistributed here; the repository's
  `docs/data-provenance.md` catalogues where to fetch each one.

FIFA World Cup and FIFA Fantasy are trademarks of FIFA; this is an
independent research project with no affiliation.

## Citation

```bibtex
@misc{{fifa_wc_fantasy_2026,
  author = {{Davila, Asher and Guajardo, Diego}},
  title = {{FIFA Fantasy WC 2026: prediction, optimization, and live
           decision support}},
  year = {{2026}},
  note = {{\\url{{https://github.com/AsherDLL/fifa-wc-fantasy}}}}
}}
```
"""


def main(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    n_files, n_bytes = 0, 0
    for group, patterns in GROUPS.items():
        dest = out_dir / group
        dest.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            for src in sorted(REPO_ROOT.glob(pattern)):
                shutil.copy2(src, dest / src.name)
                n_files += 1
                n_bytes += src.stat().st_size
    (out_dir / "README.md").write_text(CARD)
    print(f"staged {n_files} files, {n_bytes / 1e6:.1f} MB -> {out_dir}")
    print("publish AFTER the final: see docs/hf-release.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage the HF dataset release (no publishing).")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "dist/hf-dataset")
    args = parser.parse_args()
    main(args.out)
