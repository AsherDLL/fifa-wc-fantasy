"""Phase 2 features: per-(player, round) table joining the Phase 1 raw data.

Builders are pure functions on pandas DataFrames. The CLI
(`python -m fifa_fantasy.features`) reads the latest Parquet snapshots from
`data/raw/`, runs the builders, and writes a single feature table to
`data/processed/features_<UTC-date>.parquet` that Phase 3 (models) and
Phase 4 (optimizer) consume.
"""
