"""Phase 1 collector: fetch player pool, national squads, and fixtures.

The data backs the FIFA Fantasy WC 2026 game and lives at three public JSON
endpoints under https://play.fifa.com/json/fantasy/. We snapshot all three
into Parquet (one file per logical entity per day) plus the raw JSON for
forensic replay if the schema changes.

See docs/api-endpoints.md for endpoint specs.
"""
