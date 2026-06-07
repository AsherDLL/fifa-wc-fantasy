# Operations

How the system handles substitutions, injuries, knockout eliminations,
boosters, and when each component is supposed to run. Phase 0 doesn't
implement any of this yet — this document is the contract later phases
will satisfy. Every rule quoted here comes from
[`docs/Fantasy.md`](./Fantasy.md), the official FIFA WC 2026 Fantasy
guidelines.

## Locked vs. unlocked players (terminology)

From Fantasy.md:

- **Locked player**: a player whose team is currently playing a match.
- **Unlocked player**: a player whose team is yet to play.

A finished player is also unlocked in the sense that their match is over,
but the rules treat them as "completed" — they can be removed from the XI
but cannot be brought back, and they cannot be brought *into* the XI.

## Substitutions

There are three different mechanisms, often confused for each other:

### 1. Pre-round transfers (squad-level)

Swapping a player out of your 15-player squad for the next round. Limited
per stage:

| Stage | Free transfers |
|---|---|
| Pre-tournament | Unlimited |
| Before MD2 | 2 (one can roll over from MD1) |
| Before MD3 | 2 (one can roll over from MD2; cannot roll into R32) |
| Before R32 | Unlimited |
| Before R16 | 4 |
| Before QF | 4 |
| Before SF | 5 |
| Before Final | 6 |

Each transfer above the free quota costs **−3 points** (deducted once the
round locks). Once a transfer is confirmed it cannot be reversed. Transfers
only apply to the next scheduled round; making them during a live round
does not affect the current XI. The optimizer (Phase 4,
`optimizer/transfer_planner.py`) trades expected-point gain against −3 per
extra transfer.

### 2. Auto-substitutions (the game does it for you)

If a starter plays 0 minutes (DNP), the game replaces them with the
highest-priority bench player as long as the new XI is in a valid
formation. Bench players are **prioritised 1–3** (outfield slots) plus the
GK substitute slot. We must therefore output a **bench priority order**
with every Phase 4 squad recommendation, ranked by expected points ×
probability of playing.

Auto-subs only fire **at the end of the round**, so manual subs remain
possible up until the start of the round's final match — and crucially,
auto-subs only fire if **no** manual changes were made during the round.

### 3. Manual mid-round substitutions

Once a starter's match has finished (their player is no longer locked) you
can swap them out for a bench player whose match has not yet started.
Constraints from Fantasy.md:

- The player being removed must not currently be locked (i.e. their match
  is not in progress).
- The bench player coming in must be unlocked (their match has not
  started).
- The new XI must follow a valid formation.
- Once removed, a player cannot be brought back into the XI.
- A locked player cannot be swapped with another locked player under any
  circumstance.
- A finished bench player cannot replace an unlocked starter.

**Important:** any manual change — including changing captain or
vice-captain — **cancels all automatic substitutions for that round.** The
trade-off is therefore an EV calculation handled by Phase 5
(`live/sub_advisor.py`): is the expected gain from this manual sub worth
forfeiting auto-subs on any remaining unplayed starters?

## Captain and vice-captain

- Captain scores double points.
- If the captain plays 0 minutes, the vice-captain scores double points
  instead — but **only** if no manual changes were made to the team during
  the live round.
- Captain and vice-captain can be changed unlimited times before lockout.
- During a live round, captain can be changed unlimited times **as long as
  the new captain has not yet played and the current captain's match has
  completed.** Switching forfeits the old captain's double points and
  applies double to the new one going forward.
- Captain can only be changed when their match is not in progress (before
  start or after completion).
- Changing captain mid-round counts as a manual change and cancels
  auto-subs.

## Injuries and unavailability

- **Source of truth:** the FIFA Fantasy API exposes a per-player availability
  status (available, doubtful, injured, suspended). Phase 1's collector pulls
  this daily.
- **Pre-round:** the optimizer treats injured/suspended players as expected
  points = 0 (or excludes them outright). "Doubtful" gets a probability haircut.
- **Mid-round:** late ruled-out starters are handled by auto-subs. The
  sub-advisor accounts for them when deciding whether to intervene manually.

## Eliminated teams (knockouts)

- After each knockout round, half the teams go home. Their players score 0 in
  every subsequent round.
- The optimizer's `StageConfig.eliminated_teams` set forces those players out
  by setting expected points to 0; the squad solver re-builds within the new
  transfer quota.
- Transfer windows widen precisely to enable this: **pre-R32 is unlimited**
  transfers; R16/QF allow 4; SF allows 5; Final allows 6.
- Nationality caps relax stage by stage: 3 → 3 → 4 → 5 → 6 → 8.

## Playoffs vs. group stage

The scoring rules are **the same** across all stages. What changes is the
`StageConfig`: budget ($100M → $105M from R32), nationality cap, free-transfer
quota, available boosters. The optimizer is stage-aware; the scoring functions
are not.

## Boosters (chips)

Per Fantasy.md, five boosters exist. Each can be used **once**, and
**multiple boosters cannot be used in the same round**. All booster
activations except the Wildcard can be reversed before the round locks via
the booster modal's "deactivate" button. Wildcard cannot be reversed once
confirmed.

| Booster | Effect | Availability |
|---|---|---|
| **Wildcard** | Unlimited transfers within one specific round. | Any round **except** MD1 and R32 (both already unlimited). |
| **12th Man** | Adds 1 extra player to score points for your team that round. The 12th man cannot be substituted, captained, or transferred. The player can be anyone not already in your squad — **budget and nationality caps do not apply** to this pick. | Any round. |
| **Maximum Captain** | Doubles points from whichever starting-XI player scores the most in that round; captaincy is auto-assigned to that player. | Any round. |
| **Qualification Booster** | +2 points to **any starting-XI player whose team advances** to the next knockout round (or wins the Final). Player must play **at least 1 minute** to qualify. If the captain is eligible for this bonus, the +2 is **not doubled**. | Round of 32 onwards. |
| **Mystery Booster** | Effect revealed once Round 3 locks and the Round of 32 opens. Usable in one knockout-stage round including the Final. | Knockout stage (R32 onwards), once revealed. |

The Phase 4 `optimizer/booster_advisor.py` decides which booster to play
in which round under these availability constraints.

## Lockout

The game runs a **fixed lockout for transfers** plus a **rolling lockout**
for live team changes. A player is "locked" once their team's match starts
and "unlocked" until then; the live tools must read this state to know
which captain/sub changes are still allowed.

## Run cadence

| Job | Cadence | What it does |
|---|---|---|
| Data refresh | Daily during the tournament (cron) | Collector pulls prices, ownership, injury flags, match results into Parquet. |
| Pre-round pipeline | Twice per round: ~24h before lockout + final run ~1h before lockout | Rebuild features → run 4 models → run optimizer → output recommended squad, captain policy, bench order. |
| Live tools (CLI) | On-demand between kickoff windows | Captain switcher + sub advisor read latest live scores + pre-computed predictions and print a recommendation. |

This is intentionally **not** a streaming or real-time system. Even the live
tools are batch reads triggered by the user. The original sketch is explicit:
"no real-time ML."

None of the cadence machinery (cron, scheduler, etc.) is built in Phase 0. The
design is captured here so that Phase 1's collector and Phase 4's optimizer can
be wired into it without retrofitting.
