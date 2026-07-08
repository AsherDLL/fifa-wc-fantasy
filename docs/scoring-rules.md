# Scoring Rules (Canonical Reference)

Source: [`docs/Fantasy.md`](./Fantasy.md), the verbatim official FIFA World Cup
2026 Fantasy guidelines from <https://play.fifa.com/fantasy/help/guidelines>.
This document is a code-aligned table view of what `src/fifa_fantasy/scoring.py`
encodes - if they ever diverge, the code is wrong or this doc is stale. The
tests in `tests/test_scoring.py` pin every value below.

## All positions

| Action | Points |
|---|---|
| Appearance (1–59 min) | +1 |
| Appearance (60+ min) | +2 total |
| Assist | +3 |
| Yellow card | −1 |
| Red card | −2 |
| Own goal | −2 |
| Penalty won | +2 |
| Penalty conceded | −1 |
| Direct free-kick goal (bonus on top of the goal) | +1 |

A player on 0 minutes scores 0 - every component below also requires being on the pitch.

## Goalkeeper

| Action | Points |
|---|---|
| Goal scored | +9 |
| Clean sheet (60+ min) | +5 |
| Penalty save (not shootout) | +3 |
| Every 3 saves | +1 |
| Each goal conceded after the first | −1 |

## Defender

| Action | Points |
|---|---|
| Goal scored | +7 |
| Clean sheet (60+ min) | +5 |
| Each goal conceded after the first | −1 |

## Midfielder

| Action | Points |
|---|---|
| Goal scored | +6 |
| Clean sheet (60+ min) | +1 |
| Every 3 tackles | +1 |
| Every 2 chances created | +1 |

## Forward

| Action | Points |
|---|---|
| Goal scored | +5 |
| Every 2 shots on target | +1 |

## Scouting bonus

If a player **scores strictly more than 4 points** in a match **AND** is owned
by **strictly less than 5%** of managers, they receive **+2 bonus points**.

Both inequalities are strict: a base of exactly 4 or ownership of exactly 5%
does not qualify.

## Goals conceded - the "after the first" wording

The official rule docks 1 point for each goal conceded after the first. So:

| Goals conceded | Points lost |
|---:|---:|
| 0 | 0 |
| 1 | 0 |
| 2 | −1 |
| 3 | −2 |
| 4 | −3 |
| 5 | −4 |

Encoded as `-max(0, goals_conceded - 1)`. Only goalkeepers and defenders are
affected; midfielders and forwards are unaffected by goals conceded.
