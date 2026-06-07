# Phase 2 — Features

The feature builder produces a single Parquet file at
`data/processed/features_<UTC-date>.parquet` containing **one row per
(player, round) pair where that player's national squad has a fixture**.
Phase 3 (models) and Phase 4 (optimizer) consume this table.

## Squad-strength proxy

Before any match has been played, the player-pool prices are the cleanest
strength signal we have: the game's pricing model already encodes expected
points. We aggregate prices into a per-squad table:

| Column | Meaning |
|---|---|
| `squad_total_price` | Sum of `price_millions` over all players in the pool from this squad. |
| `squad_avg_price` | Mean price across the whole squad pool entry. |
| `squad_top_n_avg_price` | Mean price of the squad's top-`n` most expensive players (default `n=11`, the starting XI size). |
| `squad_top_n_rank` | Rank by `squad_top_n_avg_price`; 1 = strongest squad. |
| `squad_size` | Number of pool players from this squad. |

`squad_top_n_avg_price` is the headline proxy. Total price unfairly
rewards squads with bigger pool entries (more bench depth registered);
the mean across the whole pool penalises squads whose deep bench gets
into the pool at cheap prices. The top-N mean approximates "how good is
the side they'll actually start" without needing any tactical info.

Empirically on the day-1 pool: England, France, Spain, Portugal,
Argentina are top; Curaçao, Jordan, Haiti are bottom — matches the broad
consensus.

## Column dictionary

The output table has 35 columns. Logical groupings:

### Player identity & metadata
`player_id`, `first_name`, `last_name`, `known_name`, `full_name`,
`position`, `squad_id`, `country`, `country_abbr`, `price_millions`,
`ownership_fraction`, `status` (per-player availability), `is_eliminated`.

### Player's own squad strength
`squad_total_price`, `squad_avg_price`, `squad_top_n_avg_price`,
`squad_top_n_rank`.

### Fixture
`fixture_id`, `round_id` (1–8), `stage` (`GROUP_MD1` … `FINAL`),
`kickoff` (tz-aware), `venue_name`, `venue_city`,
`fixture_status` (the match's scheduled/finished status, renamed to
avoid collision with the player's `status`).

### Opponent (mirrored from squad strength)
`opponent_squad_id`, `opponent_name`, `opponent_abbr`,
`opp_squad_total_price`, `opp_squad_avg_price`,
`opp_squad_top_n_avg_price`, `opp_squad_top_n_rank`.

### Derived
| Column | Definition |
|---|---|
| `is_home` | Boolean — true if the player's squad is the home side. |
| `strength_diff` | `squad_top_n_avg_price - opp_squad_top_n_avg_price`. Positive = strength advantage. |
| `days_since_prev_match` | Days between the player's squad's previous match's kickoff and this one. NaN for the first match. |
| `days_to_next_match` | Same in the forward direction. NaN for the last match. |

Rest-day features are computed at the (squad, round) level on a deduped
schedule table, then merged back — every player on the same squad in the
same round shares the same rest figure.

## Snapshot semantics

- Inner join on `(player.squad_id, fixture.{home,away}_squad_id)`. A
  squad with no fixture in a given round (e.g. eliminated knockout side)
  generates no rows for that round — they're naturally absent.
- The API's `rounds.json` populates knockout brackets only as group stage
  finishes. Re-running the feature build picks up new fixtures
  automatically.
- The table is rebuilt from scratch each invocation, so there's no
  incremental write path — simpler to reason about, fast enough at this
  size (~4,500 rows × 35 cols).

## What's intentionally not here yet

- Per-match performance stats (minutes, goals, saves, …). Need games to
  have been played; deferred to Phase 3 once results land via the same
  collector.
- Rolling form features (last-N performance). Same reason.
- Component-level features (xG, xA from FBref / Understat). Deferred to a
  later iteration of Phase 3.
- Booster-aware features (qualification probability, double-gameweek
  flags). Phase 4 will compute these from the same table.
