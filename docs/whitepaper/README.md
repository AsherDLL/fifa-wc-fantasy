# Whitepaper — A Multi-Backend Predictive Framework for FIFA WC 2026 Fantasy

Master's-level technical write-up of the FIFA Fantasy WC 2026 prediction
system. Drafted in Markdown during the tournament; converted to LaTeX
after the final.

## Structure

Each numbered file under `sections/` is one section of the final paper.
Status legend: **TODO** (not yet started), **DRAFT** (first pass), **REVIEW**
(needs another pass), **FINAL** (ready for LaTeX).

| # | File | Section | Status |
|---|---|---|---|
| 00 | [outline.md](sections/00_outline.md) | Outline & section list | DRAFT |
| 00b | [authors.md](sections/00b_authors_and_contributors.md) | Authors and contributors (Diego Guajardo attribution) | DRAFT |
| 01 | [abstract.md](sections/01_abstract.md) | Abstract | TODO |
| 02 | [introduction.md](sections/02_introduction.md) | Introduction | DRAFT |
| 03 | [background.md](sections/03_background.md) | Background & related work | DRAFT |
| 03b | [literature_review.md](sections/03b_literature_review.md) | Literature review: prior art (Benter, OpenFPL, etc.) | DRAFT |
| 04 | [data.md](sections/04_data.md) | Data sources & schemas | DRAFT |
| 05 | [methodology.md](sections/05_methodology.md) | Methodology | DRAFT |
| 05b | [algorithms_formal.md](sections/05b_algorithms_formal.md) | Algorithms (formal specifications) | DRAFT |
| 06 | [implementation.md](sections/06_implementation.md) | Implementation & architecture | DRAFT |
| 07 | [validation.md](sections/07_validation.md) | Held-out validation | DRAFT |
| 08 | [live_results.md](sections/08_live_results.md) | Live tournament results | DRAFT |
| 09 | [analysis.md](sections/09_analysis.md) | Critical analysis | DRAFT |
| 10 | [lessons.md](sections/10_lessons.md) | Lessons learned | DRAFT |
| 11 | [future_work.md](sections/11_future_work.md) | Better approaches & future work | DRAFT |
| 11b | [innovation_roadmap.md](sections/11b_innovation_roadmap.md) | Innovation roadmap: prediction markets (Polymarket/Kalshi) | DRAFT |
| 12 | [conclusion.md](sections/12_conclusion.md) | Conclusion | TODO |
| 13 | [references.bib.md](sections/13_references.bib.md) | References (BibTeX-ready) | DRAFT |
| AA | [appendix.md](sections/AA_appendix.md) | Appendix: model artefacts | DRAFT |

## Editorial guidelines for the in-tournament drafting

- **Write the rationale as we make decisions, not after.** Memory rots fast.
  When a transfer call goes against the model, note it immediately under
  the relevant section with date, rationale, and observed outcome.
- **Keep numbers reproducible.** Every table and figure must reference a
  script under `scripts/` or `src/` that regenerates it.
- **No em dashes, no emojis, no LLM filler, ASCII flags only.** Same style
  rules as the rest of the project apply to the whitepaper.
- **Cite as you go.** Add references to `13_references.bib.md` in the
  BibTeX format as you draft; do not back-fill at the end.

## LaTeX conversion notes (for the post-tournament pass)

Plan: use **pandoc** with a custom LaTeX template to convert the assembled
Markdown into a single `.tex` source. Then hand-polish:

1. Replace pandoc-generated tables with `booktabs` (`\toprule`, `\midrule`,
   `\bottomrule`).
2. Convert figures to `subcaption` where multi-panel.
3. Add a proper `algorithm` environment for the optimizer pseudocode
   (replace the current code listings).
4. Validate `cite{}` references against `references.bib`.
5. Compile with `latexmk -pdf` and review.

Suggested LaTeX class: **`article`** with `\usepackage[a4paper,margin=2.5cm]{geometry}`,
or an academic template like `IEEEtran` if submitting to a venue. We pick
at conversion time, not now.

## Open questions to resolve before publishing

- How much of the squad-recommendation conversation log is appropriate to
  include as evidence? Anonymize player names from the league? (Probably
  not relevant beyond aggregate stats.)
- Position on AI assistance: this whitepaper documents a system built with
  AI pair-programming. Disclose model use; treat decisions as collaborative.
- Format for the validation tables: per-position RMSE plus full distribution
  (mean, p10, p90) per backend, comparable across all four positions and
  three backends.
