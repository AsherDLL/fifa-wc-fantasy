# 04 - Data sources and schemas

Status: **DRAFT**

## 4.1 FIFA Fantasy API (primary inference source)

Three JSON endpoints under `play.fifa.com/json/fantasy/`:

- `players.json` - per-player record with `id`, position, price (in
  millions), `percentSelected` (ownership, 0-100), `status`
  (playing/transferred/etc.), `oneToWatch`, `stats` block with
  `totalPoints`, `lastRoundPoints`, `form`, `roundPoints`,
  `nextFixtureFromActiveRound`.
- `squads.json` - per-team record with `id`, `name`, `abbr`, `group`,
  `isEliminated`.
- `rounds.json` - per-round record with status, start and end dates, and
  a `tournaments` array of fixtures (home/away squad ids, kickoff,
  scores once played).

### Schema drift incident (MD1 kickoff)

Pre-tournament, `stats.roundPoints` was a `list[int]`. On the day MD1
started, the field switched to `dict[str, int]` keyed by round id
(`{"1": 7}`). Our Pydantic schema initially modelled the field as a
list, which broke the parser. The fix (`src/fifa_fantasy/collector/parse.py`
and `schemas.py`) accepts both shapes via a `list[int] | dict[str, int]`
union and a `_round_points` normaliser that emits a list indexed by
round position.

### Two-stage validation

`Raw*` Pydantic models mirror the API verbatim (camelCase, all-optional
where the source is unreliable). A second normalisation pass produces
the downstream contract (`Squad`, `Player`, `Fixture`) used by every
later module. This is the standard "anti-corruption layer" pattern: the
API can drift without breaking the rest of the codebase, as long as the
normalisation pass is updated.

## 4.2 martj42/international_results (live Elo source)

Public CSV (`raw.githubusercontent.com/martj42/international_results/master/results.csv`)
of every international match since 1872. Columns: date, home_team,
away_team, home_score, away_score, tournament, city, country, neutral.
The companion `goalscorers.csv` and `shootouts.csv` files exist but we
do not consume them.

We compute an Elo rating per country by rolling forward through the
sorted history. The K-factor varies by tournament weight: 60 for World
Cup matches, 25 for qualifiers, 10 for friendlies. A goal-margin
adjustment scales the K-factor by sqrt((margin+1)/2). Home advantage
adds 60 to the home side's expected score except in neutral fixtures.

Output: `data/external/country_elo.csv` with columns `country_name`,
`elo`, `matches`, `last10_form`, `last24m_goals_for`,
`last24m_goals_ag`, `last_match_date`.

## 4.3 football-data.co.uk (club-level enrichment)

Per-(league, season) CSVs at
`football-data.co.uk/mmz4281/<season>/<code>.csv` where `code` is the
league (E0 = Premier League, SP1 = La Liga, D1 = Bundesliga, I1 = Serie
A, F1 = Ligue 1). We pull the most recent three full seasons plus
earlier seasons used as Elo seed data.

We use the data three ways:

- Per-club Elo computed analogously to the international Elo above,
  exported as `data/external/club_elo.csv`.
- A normalised match table (`data/external/fd_matches.parquet`) with
  bookmaker closing odds (Pinnacle, Bet365 fallback) where available.
- A per-(club, date) Elo timeline (`compute_club_elo_history`) used to
  attach team-Elo-at-match-time to the EPL training rows without
  lookahead bias.

## 4.4 Vaastav community FPL dumps (training source)

Public mirror of every Premier League fantasy round at
`github.com/vaastav/Fantasy-Premier-League`. We scrape three full
seasons (2022-23, 2023-24, 2024-25) into per-(player, gameweek) Parquet
files under `data/training/fpl_player_gameweek_<season>.parquet`. After
dropping non-appearances (zero-minute rows), we retain around 34,000
labelled rows across the three seasons.

## 4.5 Static FIFA Men's World Ranking (fallback)

`data/static/fifa_rankings.csv`, hand-maintained from FIFA's own
publication. Used as a fallback for `country_elo_diff` when the martj42
Elo is unavailable (mostly for countries that have not played any
international match in the cached window). The heuristic and Poisson
backends prefer Elo over the static rank when both are present;
falling back to rank when Elo is missing; falling back to a price-only
signal when neither is present.

## 4.6 Country-name harmonisation

A persistent annoyance: each source spells country names differently.
`United States` vs `USA` vs `United States of America`. `Korea Republic`
vs `South Korea`. The mapping table `src/fifa_fantasy/external/mapping.py`
holds the canonical lookups in two directions: `MARTJ42_TO_FIFA` (used
when joining Elo into the per-player feature table) and
`FPL_TO_FD_CLUB` (used when joining club Elo into the EPL training table).

Three near-misses we had to fix during integration:

- `Bosnia and Herzegovina` (both sources agree); our initial mapping
  incorrectly rewrote it.
- `Czechia` (FIFA) vs `Czech Republic` (martj42); add a mapping.
- `Türkiye` (FIFA, the country's official ISO name as of 2022) vs
  `Turkey` (martj42).

These ones were caught by an after-join NaN scan over the `country_elo`
column on the per-(player, round) feature table.
