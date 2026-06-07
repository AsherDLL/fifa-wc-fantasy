"""Phase 4: squad selection + starting XI + captain pick under stage-aware rules.

The MILP solvers consume the per-(player, round) predictions produced by
Phase 3 and the rules captured in `docs/Fantasy.md` (codified in
`stage_config.py`). The CLI ties it together end-to-end.
"""
