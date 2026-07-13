# Design Decisions Log

A lightweight ADR. Append a new entry whenever we make a non-obvious choice.
Each entry: date, decision, rationale, alternatives considered.

## 2026-06-06 - Project skeleton

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

**Alternative considered.** Plain positional args. Rejected - 14-arg
signatures are a code smell.

### Goals-conceded encoded as `-max(0, gc - 1)` per official rules

**Decision.** Each goal conceded after the first costs −1 (GK/DEF only).

**Why.** Verified against the official FIFA WC 2026 Fantasy rules via a
third-party guide that quotes the exact wording. Differs from FPL-style
"−1 per 2 conceded" - easy to mis-encode.

### Strict inequalities on the scouting bonus

**Decision.** Bonus triggers when base points are **strictly > 4** AND
ownership is **strictly < 5%**. A base of exactly 4 or ownership of exactly
5% does **not** trigger.

**Why.** The official rule wording is "more than 4 points" and "fewer than
5%" - both strict. Encoded explicitly with `>` and `<`; tests pin both
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


## 2026-06-07 - Phase 1 collector

### Two-stage validation (raw → normalized)

**Decision.** Each endpoint has a `Raw*` model that mirrors the API verbatim
(camelCase fields) plus a normalized model (`Squad`, `Player`, `Fixture`)
that the rest of the codebase imports.

**Why.** If FIFA renames a field or changes a type, the raw layer fails loud
at the boundary instead of silently producing wrong rows downstream. The
normalized layer is a stable contract - we control its field names.

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


## 2026-06-07 - Phase 2 features

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


## 2026-06-07 - Phase 3a baseline predictor

### Heuristic over a trained model for the first pass

**Decision.** Phase 3a ships a non-trained, deterministic predictor:
position-coef × price × matchup-factor × home-factor, zeroed for
unavailable players. The LightGBM models from the sketch are deferred to
Phase 3b.

**Why.** No labelled data exists yet - the World Cup hasn't started. A
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
differentiable - won't surprise anyone debugging it later.

### Constants live in module-level globals, not a config file

**Decision.** The eight tuning constants are at the top of `baseline.py`
with documented reasoning.

**Why.** It's an intentionally non-overengineered baseline that will be
replaced by Phase 3b. Adding a YAML config and a loader would be more
machinery than the predictor itself. When the constants need tuning, edit
the file and re-run.


## 2026-06-07 - Phase 4 optimizer

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
keeps the LightGBM models (Phase 3b) free of differential reasoning -
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


## 2026-06-07 - Transfer planner (Option C)

### Slack-variable encoding of the −3 hit

**Decision.** The piecewise penalty `3 · max(0, transfers − free)` is
modeled by a continuous non-negative slack variable `extra` with the
constraint `new_picks ≤ free + extra` and a `−3·extra` term in the
objective.

**Why.** Standard LP trick: maximization pushes `extra` to its lower
bound (0), unless squeezed by the constraint - in which case it equals
the overage exactly. No big-M needed, no extra binary variables,
fractional solutions can't appear at the optimum because `new_picks` is
already an integer.

### Unlimited-transfer stages short-circuit to solve_squad

**Decision.** When `config.free_transfers is None` (MD1, R32),
`solve_transfer` calls `solve_squad` and reports the diff against the
caller's "current squad" with cost = 0.

**Why.** The MILP would correctly produce the same answer with `free =
∞`, but skipping the slack variable keeps the model trivial and the
output identical to the existing well-tested code path. The
TransferSolution shape is preserved so the CLI doesn't branch on stage
type.

### `--from <json>` flag, not auto-pick latest

**Decision.** The CLI requires explicit `--from <path>` for transfer
mode. No auto-detection of the latest results file.

**Why.** Multiple machines push to `results/`; auto-picking "latest"
could grab someone else's recommendation. Explicit input keeps planning
deterministic and per-user.

### Rolled-over free transfers as a CLI flag

**Decision.** `--rolled-over N` adds N to the free quota; default 0.

**Why.** Fantasy.md lets you carry one unused free transfer between
group-stage rounds. The system has no memory of what you actually used,
so the user supplies it. Simpler than tracking transfer history in
state.


## 2026-06-08 - Phase 4.5 pre-lockout polish

### `oneToWatch` surfaced but not weighted

**Decision.** Added `one_to_watch` and `one_to_watch_text` to the collector
schema, propagated through features/predictions, rendered as a ⭐ next to
the player name in the report. NOT used as a weighted feature.

**Why.** It's FIFA's curation, not ours, and they hadn't populated any
flags as of 2026-06-08. Surfacing it costs nothing; weighting it without
knowing how FIFA chooses risks double-counting against our own signals.
Revisit once we see how the flag actually moves.

### `--premium-boost` as a single linear knob

**Decision.** The premium-player tilt is `boost × max(0, price − 9.0)`,
added after the existing matchup/home multipliers. One scalar, one
threshold constant. Default 0.0 preserves the original heuristic exactly.

**Why.** Resists the temptation to ship a "real model" without data. The
linear-above-threshold form gives a clean answer to "what would happen
if I cared more about £10M+ players" without inventing curvature. A
single user-facing scalar is easier to reason about than a position-by-
position table of coefficients.

### `--compare-to` as a separate flag from `--from`

**Decision.** `--compare-to` only diffs; `--from` triggers the transfer
MILP. Both can be passed together (transfer-plan AND diff).

**Why.** Different decisions. Pre-lockout you re-solve fresh every day
and want to see what shifted (compare-to). Mid-tournament you have a
real squad and want the cheapest move to next round (from + compare-to).

### Daily script lives in `scripts/`, not invokable as a Python module

**Decision.** `scripts/daily-snapshot.sh` chains the four module CLIs.
Environment variables for `STAGE` and `PREMIUM_BOOST`. Computes
yesterday's path itself and falls back gracefully if absent.

**Why.** Bash is the right shape for a chain-of-commands wrapper. Cron
calls it directly without a Python startup. Anyone reading it sees
exactly what runs.


## 2026-06-08 Phase 4.7 strength signal upgrade

### Static FIFA ranking CSV, not a live endpoint

Decision. The FIFA Men's World Ranking lives in
`data/static/fifa_rankings.csv`, a hand-maintained snapshot the user
refreshes from `https://www.fifa.com/en/rankings/men` when wanted.

Why. The official rankings page is a JS-rendered SPA whose backing
endpoint is gated behind a CMS entry id we could not derive cleanly from
the bundle. Probed candidate paths all returned the SPA shell. A static
snapshot with a documented source URL is the lowest-overhead honest
option; the loader is defensive (empty frame on missing file) so the
pipeline degrades to price-only behaviour automatically.

### Blend, do not replace, the price signal

Decision. The matchup multiplier consumes a weighted blend of the price
and ranking z-scores rather than substituting one for the other.

Why. The two signals capture different information: price tracks
club-league quality and the game's pricing model, ranking tracks
national-team form. Either alone has blind spots (price under-weights
Morocco's WC 2022 run; ranking under-weights a federation that had a
weak qualifying run). The blend defaults to 65/35 in favour of ranking
since national-team output is what we predict.

### Rank weight defaults to 0.65, alpha to 0.40

Decision. The prior alpha of 0.25 was too gentle; the matchup signal
barely moved picks. New alpha is 0.40 with the ranking signal carrying
the bigger share of the blend.

Why. The user flagged opponent strength as first-class. With the prior
values, the Phase 3a heuristic's near-linear-in-price shape dominated;
European players (richer pricing) dominated picks regardless of fixture.
After the change, defenders facing weak attacks (Spain v Cabo Verde,
Germany v Curaçao) jump into the squad as expected.

### NaN-safe fallback to price-only on missing rankings

Decision. Rows where `rank_diff` is NaN (no entry for the country)
silently fall back to the full-weight price signal.

Why. The strength upgrade should never blow up the pipeline if the CSV
is partially populated or out of date. Tests pin this behaviour.


## 2026-06-08 Phase 5 live decision support

### Captain switch threshold is E[candidate], not 2 * E[candidate]

Decision. The pre-round playbook and the live recommendation both use
`threshold = predicted_points` of the next-best unplayed starter.

Why. The captain bonus is the captain's points counted ONCE extra,
not twice. Switching captains exchanges the captain bonus from the
current captain (whose match has ended with observed score X) to the
next captain (unobserved, expected E). EV gain from switching =
E - X; the switch wins iff X < E. An earlier draft used 2 * E by
mistake, which would make the system stick too often. Tests pin the
correct threshold.

### Initial captain is the highest-E starter in the earliest kickoff window

Decision. The playbook chooses the initial captain in the earliest
kickoff window, even if a later window holds a higher-E starter.

Why. Captain switches always exchange a known score for an expected
score, so option value accrues to captains whose match is observed
before the others. By Jensen's inequality, the expected captain bonus
under the switching chain is greater than or equal to E[any single
starter's score]. Initial captain in a later window forfeits this
option and is strictly weaker in expectation.

### Live mode is detected, not flagged

Decision. The live CLI sniffs `match_status` across the squad and
flips to live mode automatically if any starter's match is completed.

Why. The same JSON snapshot serves pre-round and live use; there is no
ambiguity to resolve via a flag. The CLI title still indicates the
detected mode so the report is self-documenting.

### Sub advisor enumerates valid swaps; no DNP probability model

Decision. For each finished starter and each unplayed bench player, the
advisor builds the swapped XI and checks the formation table. If valid
and the EV gain is positive, the swap is a candidate.

Why. DNP probabilities require historical data we do not have. v1 lists
all positive-EV candidates with a single warning that any manual change
cancels auto-subs. The user decides.

### LiveState is a frozen dataclass over already-loaded Parquet

Decision. The live module does not refetch from the network. It reads
the latest Parquet snapshots under `data/raw/` and
`data/processed/`. The user runs `python -m fifa_fantasy.collector`
between kickoff windows for fresh data.

Why. Keeps the live module deterministic and offline-testable. The
collector is the single network surface for the project.


## 2026-06-08 Phase 3b LightGBM predictor

### EPL FPL as donor data, not Euro 2024 / WC 2022

Decision. Training data comes from one completed season of the
Premier League fantasy game (`fantasy.premierleague.com/api`). The
original sketch mentioned Euro 2024, but UEFA's fantasy game is a
different platform with different scoring; FIFA does not publish
archived per-player histories for prior WCs through the public Fantasy
endpoint.

Why. The Premier League FPL endpoint is well documented, freely
accessible, returns one row per (player, gameweek) with the same point
components as the WC game, and is large enough on its own (around
29,000 player-gameweek rows per season) to train per-position models.
Transfer to WC is imperfect (see docs/gbm.md), which is the trade-off
for shipping a real ML predictor before kickoff.

### Per-position models with mean + three quantile heads

Decision. Each of the four positions trains four LightGBM Boosters:
mean regression, q10, q50, q90. Inference returns all four columns,
the optimizer consumes only `predicted_points` (mean) for now.

Why. The sketch called for quantile output to feed downstream captain
and substitution decisions; q10/q90 capture the ceiling/floor without
which captain selection is a coin flip on point estimates alone. Even
though the optimizer's first pass uses the mean, the quantile columns
are written to the same Parquet so a future captain heuristic can read
them without retraining.

### Heuristic stays the default; GBM is opt-in

Decision. `python -m fifa_fantasy.model` defaults to `--backend
heuristic`. The GBM is only used when explicitly asked.

Why. The EPL-to-WC transfer is imperfect and the GBM produces a
defender-heavy squad with cheap selections that differ sharply from
the consensus picks. Until WC group-stage scoring confirms or denies
the EPL pattern, the safer pre-tournament default is the heuristic
that already shows on the recommendation people are using.

### Limit training features to columns present in both data sets

Decision. The training table is restricted to `price_millions`,
`is_home`, `strength_diff`, `squad_top_n_avg_price`,
`opp_squad_top_n_avg_price`. FIFA-specific signals (`rank_diff`,
`ownership_fraction`) are not used by the GBM yet.

Why. EPL FPL has no FIFA World Ranking and ownership data shifts
during the season in ways that do not match international fantasy.
Including features that are present only at inference would force the
model to learn signal it cannot validate during training, increasing
overfit risk. Adding them later once WC data exists is simpler than
removing them.

## 2026-07-12 - Semifinal decision layer

### Optimize the XI plus captain, not the 15-player sum

Decision. Semifinal transfer, captain, and booster advice comes from
`scripts/sf_joint_analysis.py`, a single MILP over squad, XI,
formation, captain, and transfers whose objective is the expected
round score (XI plus doubled captain, minus 3 per extra transfer,
plus a 0.1 bench tiebreak), with a seeded Monte Carlo layer pricing
plans against an ownership-built template XI. The scouting bonus
enters as 2 times P(points above 4) from the quantile heads instead
of a flat +2, and eliminated squad members are retainable zero-point
assets. The production `solve_transfer` objective is unchanged.

Why. A solve constrained to keep two midfielders produced a higher
XI expectation than the unconstrained solve, which is impossible
under a correctly specified objective and proved `solve_transfer`
maximizes bench weight that never scores. Fixing the analysis layer
recovers the points; swapping production objectives days before a
semifinal trades a known small bias for unknown regression risk.

Alternative considered. Patching `solvers.solve_transfer` in place.
Rejected mid-tournament for the same reason heuristic v2 stayed out
of production; queued for after the final (whitepaper 05d, 11).

### Register the semifinal pick comparison before lockout

Decision. `data/evaluation/sf_pick_comparison.json` stores the
manager's squad and the model's squad for round 7 with captain,
bench, booster, scoring rules, and the exact generator command
(pinned parquet, market snapshot, seed), committed before the round
locks.

Why. A comparison specified after results are known invites hindsight
in what gets counted; registering picks, rules, and inputs first
makes the round 8 report auditable (whitepaper 05d.7, 08).
