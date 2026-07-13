# FIFA World Cup Fantasy 2026: official game rules (excerpts)

Source: <https://play.fifa.com/fantasy/help/guidelines>, retrieved
2026-06-07. This file quotes only the game-mechanic rules the codebase
depends on; registration, UI walkthroughs, leaderboards, and mini-league
material from the official page are omitted. Before publication this
repository carried a full copy of the page; it was reduced to these
excerpts on 2026-07-12 because the full page is FIFA's copyrighted
content. The rule values below are restated in code-aligned form in
[`scoring-rules.md`](./scoring-rules.md) and pinned by
`tests/test_scoring.py`; per-stage limits are encoded in
`src/fifa_fantasy/optimizer/stage_config.py`.

## Squad rules

- Budget $100M for a 15-player squad: 2 GK, 5 DEF, 5 MID, 3 FWD.
- "For the Knockout Phase, the Team Budget will increase by $5m"
  (applied automatically when Round of 32 transfers open).
- Player prices are fixed and do not change during the tournament.
- Maximum players per country by stage:

| Tournament stage | Restriction |
|---|---|
| Group stage | max 3 per country |
| Round of 32 | max 3 per country |
| Round of 16 | max 4 per country |
| Quarter-final | max 5 per country |
| Semi-final | max 6 per country |
| Final | max 8 per country |

- Valid formations: 4-4-2, 4-3-3, 4-5-1, 3-4-3, 3-5-2, 5-4-1, 5-3-2.
- Unlimited team changes until the opening match on 11 June 2026.

## Captain and vice-captain

- The captain scores double points. "If your captain doesn't play any
  minute during the matchday, your Vice-Captain will score double
  points instead." The vice-captain doubling applies only if no manual
  changes were made during the live round.
- During a live round the captain may be changed to a player whose
  match has not started, once the previous captain's match is
  complete; the old captain's double is then forfeited.

## Substitutions

- Bench players score points but do not count toward the team total.
- Automatic substitutions replace did-not-play starters at the end of
  a round, in bench priority order 1-3, formation permitting, and only
  if no manual change was made during the live round (any manual
  substitution or captain change cancels auto-subs for that round).
- Manual substitutions during a live round follow locked/unlocked
  state: "Locked Player: A player whose team is currently playing a
  match. Unlocked Player: A player whose team is yet to play." A
  starter who completed their match may be replaced by a bench player
  whose match has not started; locked players cannot be swapped.

## Transfers

Free-transfer allocation by stage; each transfer beyond the allocation
deducts 3 points. Confirmed transfers cannot be reversed. Transfers
made during a live round apply from the next round.

| Stage | Allocation |
|---|---|
| Pre-tournament | unlimited |
| Before Matchday 2 | 2 |
| Before Matchday 3 | 2 |
| Before Round of 32 | unlimited |
| Before Round of 16 | 4 |
| Before Quarter-finals | 4 |
| Before Semi-finals | 5 |
| Before the Final | 6 |

During the group stage one free transfer can carry over to the next
round (none can carry into the Round of 32).

## Boosters

One booster per round, each usable once, not combinable; all except the
Wildcard can be deactivated before lockout.

- Wildcard: "unlimited transfers within a specific round" (not MD1,
  not R32); irreversible once confirmed.
- 12th Man: one additional scoring player for a round; cannot be
  substituted, captained, or transferred; budget and country
  restrictions do not apply; must not already be in the squad.
- Maximum Captain: double points from whichever starter scores most
  that round (captaincy auto-assigned).
- Qualification Booster (R32 onward): "+2 points to any player in your
  starting XI who progresses to the next round of the knockout stage
  or wins the final", minimum 1 minute played; a captain's +2 is not
  doubled.
- Mystery Booster, revealed at R32 as the Clean Sheet Shield:
  goalkeepers, defenders, and midfielders lose their potential
  clean-sheet points only if their team concedes two or more goals in
  the round (60+ minutes played required).

## Scoring

All players: appearance up to 60 min +1; 60+ min a further +1; assist
+3; yellow card -1; red card -2; own goal -2; winning a penalty +2;
conceding a penalty -1.

| Position | Rule | Points |
|---|---|---|
| GK | clean sheet (60+ min) | +5 |
| GK | first goal conceded / each additional | 0 / -1 |
| GK | goal scored | +9 |
| GK | penalty save (not shootouts) | +3 |
| GK | every 3 saves | +1 |
| DEF | clean sheet (60+ min) | +5 |
| DEF | first goal conceded / each additional | 0 / -1 |
| DEF | goal scored | +7 |
| MID | clean sheet (60+ min) | +1 |
| MID | goal scored | +6 |
| MID | every 3 tackles | +1 |
| MID | every 2 chances created | +1 |
| FWD | goal scored | +5 |
| FWD | every 2 shots on target | +1 |

Bonus points: goal from a direct free kick +1 (on top of the goal);
scouting bonus +2 "if any of your players scores more than 4pts in a
match and is in fewer than 5% of all teams' selection".

## Lockout

Fixed lockout for transfers plus a rolling lockout for team changes
during a live round (per-player locking as above).
