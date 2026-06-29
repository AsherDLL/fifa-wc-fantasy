# 00b — Authors and contributors

Status: **DRAFT**

This work was performed by a small team combining computational and
domain expertise. We list contributions explicitly to credit the parts
that came from football-watching judgement separately from the parts
that came from code.

## Authors

**Asher Davila** (predator dev, AsherDLL on GitHub).
Computational lead. Designed the system architecture, implemented the
three predictor backends, the MILP optimiser, the live decision tools,
and the data ingestion pipeline. Conducted all held-out validation and
hyperparameter sweeps. Authored the code and the bulk of the whitepaper.
Operated the personal-league fantasy entry through every matchday.

**Diego Guajardo**, Actuarial Scientist.
Domain expert. Watched the qualifier rounds, the pre-tournament
friendlies, and many of the WC 2026 matches. Provided per-player
judgements that the predictive models could not derive from features
alone: minutes risk for specific players based on observed manager
behaviour, in-form vs out-of-form distinctions between players at the
same model-predicted score, and tournament-narrative signals (top
scorer race incentives, must-win match dynamics). Specifically credited
with the pre-tournament call to include Willian Pacho (Ecuador
defender) in the squad; Pacho returned 11 fantasy points across the
group stage at $4.4M, an above-market return per million.

**Claude (Anthropic, model Opus 4.7)**, AI pair-programming assistant.
Used throughout for code authoring, debugging, validation harness
design, whitepaper drafting, and multi-turn decision conversations on
each transfer round. Per the standard disclosure rules adopted by
several CS venues in 2024-2025, we note that the model contributed
substantially to both the code and the written analysis. The final
decisions on squad construction, transfers, and captain choices were
made by the human authors after weighing model and human inputs.

## Contribution statement (in the spirit of CRediT taxonomy)

| Activity | A. Davila | D. Guajardo | AI assistant |
|---|---|---|---|
| Conceptualization | lead | supporting | supporting |
| Methodology (modelling) | lead | reviewing | supporting |
| Methodology (domain priors) | supporting | lead | none |
| Software | lead | none | supporting (pair-programming) |
| Validation (held-out RMSE) | lead | none | supporting |
| Validation (live matchday calls) | shared | shared | advisory |
| Data curation | lead | none | supporting |
| Writing - original draft | lead | none | substantial supporting |
| Writing - review and editing | lead | substantial | supporting |
| Visualization | lead | none | supporting |
| Project administration | lead | none | none |

## On AI assistance

The whitepaper documents a system built collaboratively with an AI
pair-programmer. We treat this honestly:

- Code authored or substantially modified by the AI is in the
  repository under regular version control. There is no separate
  attribution per commit; the contribution mixes throughout.
- Decisions on every transfer round (Section 8) were made by the human
  authors. The AI surfaced numerical analyses and recommendations; the
  human authors weighed them against domain priors and chose.
- Whitepaper sections drafted or substantially revised by the AI are
  reviewed by the human authors before commit. Style consistency (no
  em dashes, no emojis, no LLM filler) is enforced in the editorial
  guide in `docs/whitepaper/README.md`.

This is a working methodology, not a recommendation. The right approach
to AI attribution in academic work is still evolving. We disclose the
collaboration so the reader can weigh the results accordingly.
