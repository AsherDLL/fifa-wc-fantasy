# Design Decisions Log

A lightweight ADR. Append a new entry whenever we make a non-obvious choice.
Each entry: date, decision, rationale, alternatives considered.

## 2026-06-06 — Project skeleton

### `src/` layout with single `fifa_fantasy` package

**Decision.** Use `src/fifa_fantasy/<subpackage>/...` rather than the original
sketch's flat `src/collector/`, `src/features/`, ... layout.

**Why.** Imports become unambiguous (`from fifa_fantasy.scoring import …`)
without `PYTHONPATH` hacks. The `src/` layout also prevents accidental
imports from the working directory during development. This is the modern
Python convention.

**Alternative considered.** Flat layout from the sketch. Rejected because it
requires `pythonpath = ["src/collector", "src/features", ...]` or sys.path
fiddling; the single-package approach is simpler.

### `MatchStats` dataclass instead of long positional signature

**Decision.** Pass match stats as a `@dataclass(frozen=True) MatchStats` with
defaults, rather than the sketch's 14-parameter function signature.

**Why.** Self-documenting call sites, additive (new fields don't break old
callers), cleaner tests. Frozen makes it implicitly hashable and prevents
accidental mutation.

**Alternative considered.** Plain positional args. Rejected — 14-arg
signatures are a code smell.

### Goals-conceded encoded as `-max(0, gc - 1)` per official rules

**Decision.** Each goal conceded after the first costs −1 (GK/DEF only).

**Why.** Verified against the official FIFA WC 2026 Fantasy rules via a
third-party guide that quotes the exact wording. Differs from FPL-style
"−1 per 2 conceded" — easy to mis-encode.

### Strict inequalities on the scouting bonus

**Decision.** Bonus triggers when base points are **strictly > 4** AND
ownership is **strictly < 5%**. A base of exactly 4 or ownership of exactly
5% does **not** trigger.

**Why.** The official rule wording is "more than 4 points" and "fewer than
5%" — both strict. Encoded explicitly with `>` and `<`; tests pin both
boundaries.

### Scoring functions return `int`

**Decision.** All scoring components are integer, including the +2 scouting
bonus, so the public API returns `int`.

**Why.** Avoids float surprises in downstream optimizer constraints. The
ownership comparison uses a float, but the result of the bonus is an int.

### 0 minutes short-circuits to 0

**Decision.** `calc_points` returns 0 immediately if `minutes <= 0`, ignoring
all other stat fields.

**Why.** Defensive: a player who didn't appear can't have scored a goal,
kept a clean sheet, made saves, etc. Encoding it once at the dispatcher keeps
each per-position function simpler. Tests verify this with a "great stats but
0 minutes" case.


## 2026-06-07 — Phase 1 collector

### Two-stage validation (raw → normalized)

**Decision.** Each endpoint has a `Raw*` model that mirrors the API verbatim
(camelCase fields) plus a normalized model (`Squad`, `Player`, `Fixture`)
that the rest of the codebase imports.

**Why.** If FIFA renames a field or changes a type, the raw layer fails loud
at the boundary instead of silently producing wrong rows downstream. The
normalized layer is a stable contract — we control its field names.

### Persist raw JSON next to Parquet

**Decision.** Every fetch writes `data/raw/raw/<name>_<UTC-timestamp>.json`
alongside the normalized `data/raw/<name>_<UTC-date>.parquet`.

**Why.** Cheap insurance. If a normalization bug or schema drift breaks
parsing, we can re-parse from the original payload without re-fetching
(and without depending on the API still serving the same data).

### Ownership stored as a [0, 1] fraction

**Decision.** The API delivers `percentSelected` as a percent (1.2 means
1.2%). The normalized `Player.ownership_fraction` divides by 100.

**Why.** `scoring.py`'s scouting-bonus check already uses `ownership_pct`
as a fraction (`< 0.05` means "less than 5%"). Aligning the unit at the
normalization boundary keeps every downstream consumer consistent.

### Endpoints discovered by static analysis, not browser automation

**Decision.** Reverse-engineered the SPA's JS bundle and probed sibling
paths under the same static-JSON prefix, rather than driving a headless
browser.

**Why.** Lighter dependency surface (no Playwright + Chromium download)
and reproducible from the bundle alone. The bundle's `Cr.get(...)` and
`na.get(...)` call sites named the convention (`<thing>.json`), and the
three target files turned out to be siblings of the already-known
`checksums.json` / `countries.json` / `faq.json`.

### `argparse` over `click` / `typer`

**Decision.** CLI uses standard-library `argparse`.

**Why.** No extra dependency for a three-flag command. Easy to extend if
the collector ever grows subcommands beyond `--only`.
