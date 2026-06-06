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
