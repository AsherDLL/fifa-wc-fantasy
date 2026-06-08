"""Phase 5: live decision support.

Two tools, one CLI:

- Captain switcher: produces a pre-round playbook listing the initial
  captain plus the chain of switch targets in subsequent kickoff windows;
  in live mode (after at least one fixture has finished) it produces a
  single concrete recommendation.
- Sub advisor: identifies finished starters whose match underdelivered
  relative to an unplayed bench player; recommends the highest-EV swap
  while flagging the auto-sub cancellation cost.

Inputs are the latest collector + predictions Parquet plus a previously
saved recommendation JSON (the squad and lineup we are evaluating).
"""
