# Tournament structure and round naming

This document explains the WC 2026 format and what the labels MD1, MD2,
MD3, R32, R16, QF, SF, FINAL mean inside the codebase. None of these
labels is invented by this project; they come straight from Fantasy.md
and from the official FIFA Fantasy game.

## WC 2026 at a glance

- 48 national teams across 12 groups of 4.
- Each team plays 3 group-stage matches against the other 3 teams in
  its group. That's 3 game days per team in the group stage.
- Top 2 from each group plus the 8 best third-placed teams advance to
  the Round of 32 knockout bracket.
- After R32 the bracket halves each round: R16, QF, SF, FINAL.
- A team can play at most 8 matches total: 3 group + 5 knockout.

## Rounds inside the Fantasy game

The fantasy game groups all matches in a "round" into a single window
with one lockout time. Eight rounds total:

| Round id | Code in this repo | What it covers |
|---|---|---|
| 1 | `GROUP_MD1` | Matchday 1 of the group stage: each of the 48 teams plays its first group game. |
| 2 | `GROUP_MD2` | Matchday 2 of the group stage. |
| 3 | `GROUP_MD3` | Matchday 3 of the group stage. After this the group tables are final. |
| 4 | `R32` | Round of 32: first knockout round, 32 teams in 16 matches. |
| 5 | `R16` | Round of 16. |
| 6 | `QF` | Quarter-finals (8 teams, 4 matches). |
| 7 | `SF` | Semi-finals (4 teams, 2 matches). |
| 8 | `FINAL` | The final AND the third-place game: both score fantasy points in round 8 (confirmed 2026-07-15 against the live rounds.json - the round window opens at the bronze kickoff and both SF losers stay non-eliminated - and by Fantasy Football Scout's 2026-07-13 explainer; the play-off also counted in 2010-2022). The round deadline is the bronze kickoff. |

`MD` stands for "matchday". MD1 is the first round of group games. MD2
the second, and so on. Each MD spans roughly one calendar week in the
group stage because the 12 groups play their MD1, MD2, MD3 staggered
across days.

This naming has nothing to do with markdown files (`.md`). The fantasy
round `GROUP_MD1` and the file extension `.md` are unrelated; the
collision is an unfortunate abbreviation that this document is here to
disambiguate.

## What changes round to round

Most rules change between the group stage and the knockout stages. See
Fantasy.md for the exact wording; the summary table:

| Rule | Group (MD1 -> MD3) | Knockouts (R32 -> FINAL) |
|---|---|---|
| Budget | $100M | $105M from R32 onward |
| Squad shape | 2 GK, 5 DEF, 5 MID, 3 FWD (unchanged) | same |
| Starting XI | 11 in a valid formation (4-4-2, 4-3-3, 4-5-1, 3-4-3, 3-5-2, 5-4-1, 5-3-2) | same |
| Nationality cap | 3 per country | 3 (R32), 4 (R16), 5 (QF), 6 (SF), 8 (FINAL) |
| Free transfers per round | 2 before MD2 + can roll 1, 2 before MD3 + can roll 1 (cannot roll into R32) | unlimited before R32; 4 before R16, 4 before QF, 5 before SF, 6 before FINAL |
| Extra transfer cost | -3 points per transfer above the free quota | same |

The optimizer encodes these per-stage rules in
`src/fifa_fantasy/optimizer/stage_config.py`. Tests pin every value
against Fantasy.md.

## What to run when, end to end

A fuller runbook lives in the README, but the short version:

- **Pre-tournament (until June 11, 2026)**: run `./scripts/daily-snapshot.sh`
  daily. It refreshes the live FIFA data, rebuilds features and
  predictions, and writes a new recommendation. Lock in the MD1 squad
  within an hour of the first MD1 kickoff.
- **During each round**: between kickoff windows, refetch with
  `python -m fifa_fantasy.collector` then run
  `python -m fifa_fantasy.live --recommendation <latest>.json` for the
  captain switch and substitution recommendations.
- **Between rounds**: plan transfers with
  `python -m fifa_fantasy.optimizer --stage GROUP_MD2 --from <previous>.json`.
- **Knockout transition (after MD3, before R32)**: unlimited transfers
  apply, so re-solve fresh:
  `python -m fifa_fantasy.optimizer --stage R32`.
