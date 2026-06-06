# Operations

How the system handles substitutions, injuries, knockout eliminations, and
when each component is supposed to run. Phase 0 doesn't implement any of this
yet — this document is the contract later phases will satisfy.

## Substitutions

There are three different mechanisms, often confused for each other:

### 1. Pre-round transfers (squad-level)

Swapping a player out of your 15-player squad for the next round. Limited per
stage:

| Stage | Free transfers |
|---|---|
| Pre-tournament | Unlimited |
| Before MD2 | 2 (can roll 1 from MD1) |
| Before MD3 | 2 (cannot roll into R32) |
| Before R32 | Unlimited |
| Before R16 | 4 |
| Before QF | 4 |
| Before SF | 5 |
| Before Final | 6 |

Each transfer above the free quota costs −3 points. The optimizer
(Phase 4, `optimizer/transfer_planner.py`) trades expected-point gain against
−3 per extra transfer.

### 2. Auto-substitutions (the game does it for you)

If a starter plays 0 minutes, the game replaces them with the highest-priority
bench player who keeps a valid formation. We must therefore output a
**bench priority order** with every Phase 4 squad recommendation, ranked by
expected points × probability of playing.

### 3. Manual mid-round substitutions

Once a starter's match has finished, you can manually swap them out for an
unplayed bench player. Constraints:

- Once removed, a player cannot return to the XI.
- **Any manual change cancels all automatic substitutions for that round.**

The trade-off is therefore an EV calculation, handled by Phase 5
(`live/sub_advisor.py`): is the expected gain from this manual sub worth
forfeiting auto-subs on any remaining unplayed starters?

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
