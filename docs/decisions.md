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


## 2026-06-07 — Phase 2 features

### Squad strength = top-N price average

**Decision.** The headline squad-strength proxy is the mean price of the
top `n=11` most expensive players in each squad's pool entry, with rank 1
= strongest.

**Why.** Total price unfairly rewards squads with bigger pool entries
(some have 55 listed, others 30). Whole-roster mean penalises squads
whose deep bench enters at cheap prices. Top-11 mean approximates
starting-XI quality without any tactical input. Empirically the top of
the ranking matches consensus (England, France, Spain, Portugal,
Argentina).

### Rest days at the squad level, merged back

**Decision.** `days_since_prev_match` and `days_to_next_match` are
computed on a deduplicated `(squad_id, round_id, kickoff)` schedule
table, then merged onto the per-player rows.

**Why.** They're a squad-level property. Computing them on the wide
table (12+ rows per squad-round) made `groupby().diff()` see the same
kickoff repeated and return 0 instead of the real gap. Caught by a unit
test before we ever ran live.

### `fixture_status` rename

**Decision.** The fixture's `status` column is renamed to
`fixture_status` inside `flatten_fixtures` to avoid a `_x`/`_y` suffix
collision with the player-availability `status` column when merging.

**Why.** Both columns are real and meaningful; we want both to survive
the merge without ambiguity.

### Inner-join on `squad_id`

**Decision.** The feature table is an inner join of players and fixtures
on `squad_id`. Eliminated teams (no fixture in a given round) simply
produce no rows for that round.

**Why.** The downstream optimizer only needs rows for players in
playable matchups; missing rows are unambiguous. No need to invent
sentinel rows.


## 2026-06-07 — Phase 3a baseline predictor

### Heuristic over a trained model for the first pass

**Decision.** Phase 3a ships a non-trained, deterministic predictor:
position-coef × price × matchup-factor × home-factor, zeroed for
unavailable players. The LightGBM models from the sketch are deferred to
Phase 3b.

**Why.** No labelled data exists yet — the World Cup hasn't started. A
heuristic gets the optimizer end-to-end before kickoff and lets us
sanity-check the full pipeline. The market's price IS a reasonable
expected-points signal (the game's pricing model encodes it explicitly).

### Multiplicative matchup + home, not additive

**Decision.** Fixture difficulty and home advantage modulate the base
prediction multiplicatively, not by adding flat bonuses.

**Why.** A 0.5 pt home bonus would dominate a cheap defender's prediction
and barely move a premium forward's. The multiplicative form scales the
adjustment with the base, which is the right shape for "fixture helps
better players more."

### tanh, not linear, for strength-diff

**Decision.** The strength-diff multiplier uses `tanh(diff / scale)`
rather than a raw linear factor.

**Why.** Caps the matchup effect at ±alpha so an unrealistically large
strength gap doesn't multiply a prediction by 3×. Smooth and
differentiable — won't surprise anyone debugging it later.

### Constants live in module-level globals, not a config file

**Decision.** The eight tuning constants are at the top of `baseline.py`
with documented reasoning.

**Why.** It's an intentionally non-overengineered baseline that will be
replaced by Phase 3b. Adding a YAML config and a loader would be more
machinery than the predictor itself. When the constants need tuning, edit
the file and re-run.


## 2026-06-07 — Phase 4 optimizer

### MILP via PuLP/CBC

**Decision.** Both the squad and the lineup are modeled as MILPs and
solved with PuLP's bundled CBC backend.

**Why.** With 1,481 binary squad variables plus 7 formation indicators
plus 15 lineup variables, the model is tiny by MILP standards and CBC
solves it in <1 s. Greedy heuristics can't simultaneously honour the
budget, position counts, and nationality cap without a repair step that
risks suboptimality. MILP gives provable optimality for the fixed inputs
we feed it.

### Scouting bonus injected at the optimizer, not the predictor

**Decision.** `apply_scouting_bonus` lives in `optimizer/pipeline.py`,
not `model/baseline.py`.

**Why.** The bonus depends on ownership, which is a market signal
external to the scoring function. Keeping it out of the predictor lets
the predictor speak purely to expected match performance; the optimizer
combines that with ownership to compute the effective objective. Also
keeps the LightGBM models (Phase 3b) free of differential reasoning —
they predict points, we add the +2 if applicable.

### Horizon-summed squad, round-specific lineup

**Decision.** The 15-player squad is selected to maximise the sum of
effective points across the horizon (e.g. MD1+MD2+MD3 for the
pre-tournament selection); the lineup is solved only for the first round
in the horizon.

**Why.** Mirrors the game: you pick a squad once per "phase" (subject to
the transfer quota), and you set a lineup per round. Optimising squad
selection per-round and then summing would over-penalise rotation
candidates; this approach picks a roster that performs across the whole
horizon.

### Bench priority: outfield by descending predicted_points, GK last

**Decision.** `bench_ids` returns the outfield bench sorted by
predicted_points desc, then the spare GK.

**Why.** Matches the game's auto-sub rule (Fantasy.md): outfield bench
slots 1–3 sub in order; the GK substitute only ever replaces the other
GK. Encoding the order at the optimizer level means the consumer just
takes the list as-is.

### Captain = highest predicted_points, no ownership-tilt

**Decision.** Captain is the starter with the highest single-round
predicted_points. Vice-captain is the second-highest.

**Why.** Phase 3a is a point estimate, not a distribution; there's
nothing to ownership-tilt against without a sense of variance. Phase 3b's
quantile regression will let captain selection prefer higher-variance
players (P90 ceiling) over higher-mean ones; revisit then.
