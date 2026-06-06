# FIFA Fantasy World Cup 2026 — Project Sketch (v2)

## 1. What We're Building

A system that solves three interconnected problems:

1. **Predicts** how many fantasy points each player will score per matchday, using position-specific models aligned to the actual FIFA scoring rubric.
2. **Selects** the optimal 15-player squad and starting XI under budget, formation, and nationality constraints that change per tournament stage.
3. **Advises** on live in-round decisions: captain switching across kickoff windows, manual substitutions, and booster timing.

Problems 1 and 2 are pre-round (batch). Problem 3 is live (reactive, but not real-time ML — it's decision logic over pre-computed predictions).

---

## 2. The Actual Rules (Codified)

Everything below comes from the official FIFA Fantasy rules and must be encoded as constraints, features, or scoring functions in our system.

### 2.1 Squad Composition

- 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD
- Starting XI from those 15, in a valid formation
- Valid formations: 4-4-2, 4-3-3, 4-5-1, 3-4-3, 3-5-2, 5-4-1, 5-3-2

### 2.2 Budget

| Stage | Budget |
|---|---|
| Group stage (MD1–MD3) | $100.0M |
| Round of 32 onward | $105.0M |

Player prices are **fixed** throughout the tournament. No price rises or falls.

### 2.3 Nationality Caps (Dynamic)

| Tournament Stage | Max Players Per Country |
|---|---|
| Group Stage | 3 |
| Round of 32 | 3 |
| Round of 16 | 4 |
| Quarter-finals | 5 |
| Semi-finals | 6 |
| Final | 8 |

This is critical for the optimizer — constraints must be stage-aware.

### 2.4 Transfers

| Stage | Free Transfers |
|---|---|
| Pre-tournament | Unlimited |
| Before MD2 | 2 (can roll 1 from MD1) |
| Before MD3 | 2 (cannot roll into R32) |
| Before Round of 32 | Unlimited |
| Before Round of 16 | 4 |
| Before Quarter-finals | 4 |
| Before Semi-finals | 5 |
| Before Final | 6 |

Additional transfers cost **-3 points each**. The optimizer must model the transfer cost vs. predicted point gain tradeoff.

### 2.5 Scoring System

#### All Positions

| Action | Points |
|---|---|
| Appearance (up to 60 min) | +1 |
| Appearance (60+ min) | +1 (total +2) |
| Assist | +3 |
| Yellow card | -1 |
| Red card | -2 |
| Own goal | -2 |
| Winning a penalty | +2 |
| Conceding a penalty | -1 |
| Direct free-kick goal bonus | +1 |

#### Goalkeeper-Specific

| Action | Points |
|---|---|
| Goal scored | +9 |
| Clean sheet (60+ min) | +5 |
| Penalty save | +3 |
| Every 3 saves | +1 |
| Each goal conceded after the 1st | -1 |

#### Defender-Specific

| Action | Points |
|---|---|
| Goal scored | +7 |
| Clean sheet (60+ min) | +5 |
| Each goal conceded after the 1st | -1 |

#### Midfielder-Specific

| Action | Points |
|---|---|
| Goal scored | +6 |
| Clean sheet (60+ min) | +1 |
| Every 3 tackles | +1 |
| Every 2 chances created | +1 |

#### Forward-Specific

| Action | Points |
|---|---|
| Goal scored | +5 |
| Every 2 shots on target | +1 |

#### Scouting Bonus (Differential Mechanic)

If a player:
- Scores **>4 points** in a match, AND
- Is owned by **<5% of managers**

→ They receive **+2 bonus points**.

This is not a soft heuristic. It's a hard scoring rule that must be modeled.

### 2.6 Captaincy (Live Sequential Decision)

- Captain scores **double points**
- Captain can be **switched unlimited times** during a live round
- Can only switch to players who **haven't played yet**
- Can only switch **after current captain's match is complete**
- Vice-captain only activates if captain gets 0 minutes AND you made zero manual changes

### 2.7 Manual Substitutions (Live)

- Bench players score points but don't count unless subbed in
- You can swap a **finished** starter for an **unplayed** bench player
- Once removed, a player cannot return to the XI
- **Any manual change cancels all automatic substitutions for that round**

### 2.8 Boosters (Chips)

One per round. Five total:

| Booster | Effect | Availability |
|---|---|---|
| Wildcard | Unlimited transfers for one round | Not before MD1 or R32 (already unlimited) |
| 12th Man | Add one extra player (ignores budget + nationality caps, can't be captained) | Any round |
| Maximum Captain | Auto-doubles highest-scoring starter (removes captaincy risk) | Any round |
| Qualification Booster | +2 pts per starting XI player whose team advances | R32 onward |
| Mystery Booster | Unknown — revealed after group stage | Knockout stage |

---

## 3. How the Rules Reshape the Model

### 3.1 Four Separate Prediction Models, Not One

Each position has a fundamentally different point-generating function. A single model would need to learn that "saves" matter only for GKs and "shots on target" only for FWDs — possible but wasteful. Separate models let us:

- Use **position-specific feature sets** (saves, shot-stopping stats for GK; xG, shot volume for FWD)
- Tune hyperparameters independently (GK prediction is noisier, needs more regularization)
- Interpret feature importance per position (debugging "why did it pick this defender?")

```
models/
├── gk_model.lgbm      # target: GK fantasy points
├── def_model.lgbm     # target: DEF fantasy points
├── mid_model.lgbm     # target: MID fantasy points
└── fwd_model.lgbm     # target: FWD fantasy points
```

### 3.2 Ownership Must Be a First-Class Feature

The scouting bonus creates a **non-linear reward for low-ownership picks**. The model pipeline needs:

1. **Pre-round ownership snapshot** from the Fantasy API
2. A feature: `ownership_pct`
3. In the optimizer: if `predicted_points > 4` and `ownership_pct < 0.05`, add +2 to the player's effective predicted score

This is deterministic given our prediction — it goes in the optimizer, not the ML model.

### 3.3 The Optimizer Needs Stage-Aware Constraints

The LP solver can't have hardcoded constraints. It needs a config object:

```python
@dataclass
class StageConfig:
    budget: float               # 100.0 or 105.0
    max_per_country: int        # 3, 4, 5, 6, or 8
    free_transfers: int         # 2, 4, 5, 6, or unlimited
    transfer_cost: int          # -3 per extra transfer
    available_boosters: list    # remaining unused boosters
    eliminated_teams: set       # teams knocked out (empty their players)
```

### 3.4 Captain Optimization Is a Sequential Decision Tree

Pre-round, we can't solve captaincy optimally because it depends on outcomes. But we CAN pre-compute a **captain switching policy**:

```
Before the round:
  - Rank all players by E[points] across kickoff windows
  - Assign initial captain = highest E[points] in earliest window

After each window closes:
  - If captain scored >= threshold T → STICK
  - If captain scored < T and remaining players have higher E[points] → SWITCH to next-best

T is calibrated from historical data (what score makes it worth the risk of switching?)
```

This is closer to a decision tree / threshold policy than an ML model. Simple, interpretable, and backtestable.

### 3.5 Bench Ordering and Sub Strategy

Since manual subs cancel auto-subs, the system should recommend:

1. **Bench priority order** (for the "no-touch" auto-sub path)
2. **Conditional sub recommendations** per kickoff window: "If Player X scores < Y, sub in Player Z"

This requires the model to output not just E[points] but a **distribution** — which LightGBM quantile regression provides.

---

## 4. Revised Project Structure

```
fifa-fantasy/
├── docker-compose.yml
├── Dockerfile
├── README.md
├── pyproject.toml
│
├── src/
│   ├── collector/                  # PHASE 1: Data ingestion
│   │   ├── __init__.py
│   │   ├── fantasy_api.py          # FIFA Fantasy API (prices, ownership, fixtures, points)
│   │   ├── fbref.py                # xG, xA, defensive stats, shot data
│   │   ├── understat.py            # (optional) granular xG
│   │   └── schemas.py              # Pydantic models for raw data validation
│   │
│   ├── features/                   # PHASE 2: Feature engineering
│   │   ├── __init__.py
│   │   ├── gk_features.py          # saves, clean sheet prob, goals faced
│   │   ├── def_features.py         # clean sheet prob, xG from set pieces, tackles
│   │   ├── mid_features.py         # xG, xA, chances created, tackles
│   │   ├── fwd_features.py         # xG, shots on target, penalty duties
│   │   ├── fixture_features.py     # opponent strength, home/away, rest days
│   │   ├── meta_features.py        # ownership %, captain popularity, transfer trends
│   │   └── pipeline.py             # orchestrates all builders, outputs unified table
│   │
│   ├── model/                      # PHASE 3: Prediction (one model per position)
│   │   ├── __init__.py
│   │   ├── position_models.py      # GK/DEF/MID/FWD model definitions
│   │   ├── baseline.py             # ridge regression per position
│   │   ├── train.py                # trains all 4 models + quantile variants
│   │   └── evaluate.py             # backtesting, MAE per position, calibration
│   │
│   ├── optimizer/                  # PHASE 4: Squad selection + live decisions
│   │   ├── __init__.py
│   │   ├── stage_config.py         # tournament stage rules (budget, caps, transfers)
│   │   ├── squad_solver.py         # PuLP: optimal 15-player squad
│   │   ├── lineup_solver.py        # PuLP: optimal starting XI + formation
│   │   ├── captain_policy.py       # sequential captain switching strategy
│   │   ├── scouting_bonus.py       # ownership-based bonus injection
│   │   ├── transfer_planner.py     # multi-round transfer optimization
│   │   └── booster_advisor.py      # when to play each chip
│   │
│   ├── live/                       # PHASE 5: In-round decision support
│   │   ├── __init__.py
│   │   ├── sub_advisor.py          # "should I manual sub?" given current scores
│   │   └── captain_switcher.py     # "stick or twist?" after each kickoff window
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── scoring.py              # canonical scoring rules as pure functions
│       └── logger.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_experiments.ipynb
│   ├── 04_optimizer_validation.ipynb
│   └── 05_captain_policy_backtest.ipynb
│
└── tests/
    ├── test_scoring.py             # verify scoring functions match FIFA rules exactly
    ├── test_features.py
    ├── test_optimizer.py
    └── test_captain_policy.py
```

---

## 5. Phases (Revised Build Order)

### Phase 0 — Scoring Rules as Code (`utils/scoring.py`)

Before anything else, encode the scoring rules as pure functions. This is the contract everything else builds on.

```python
def calc_gk_points(minutes, goals, saves, clean_sheet, goals_conceded,
                   assists, yellow, red, own_goal, pen_save, pen_won,
                   pen_conceded, free_kick_goal, ownership_pct) -> int:
    """Calculate GK fantasy points. No side effects. Fully testable."""
    ...
```

Write exhaustive unit tests for these. If the scoring functions are wrong, every model and optimizer built on top will be wrong. This is your foundation.

### Phase 1 — Data Collection

Same as v1 but with emphasis on collecting **position-specific stats**:
- GK: saves, shots on target faced, clean sheets
- DEF: tackles, interceptions, clean sheets, set-piece xG
- MID: chances created, tackles, xG, xA
- FWD: shots, shots on target, xG, penalty-taking

Plus ownership % snapshots from the Fantasy API (needed for scouting bonus modeling).

### Phase 2 — Feature Engineering (Position-Specific)

Four parallel feature pipelines, one per position. Shared features (fixture difficulty, rest days, tournament stage) are computed once and joined.

Key difference from v1: **predict the components, not just total points.** For a midfielder, predicting "xG per 90" and "chances created per 90" separately, then passing them through the scoring function, may be more accurate than predicting total fantasy points directly. This is called **component-level modeling** and it lets you audit predictions ("the model thinks Bellingham will score 0.4 xG and create 3 chances → 6×0.4 + 1×floor(3/2) = 3.4 pts from those actions alone").

### Phase 3 — Prediction Models

Four LightGBM models (one per position), each outputting:
- **Point estimate**: E[fantasy_points]
- **Quantile estimates**: P10, P50, P90 (via quantile regression)

The quantiles feed captain switching and substitution decisions. A player with E=6, P90=14 is a different captain prospect than one with E=6, P90=8.

Training data: historical FIFA Fantasy data (Euro 2024 used the same scoring system) + club-season stats mapped to international performance.

### Phase 4 — Optimizer

The LP solver now handles:
- Stage-aware budget ($100M vs $105M)
- Stage-aware nationality caps (3 to 8)
- Transfer cost modeling (-3 per extra transfer)
- Scouting bonus injection (if predicted_pts > 4 and ownership < 5%, add +2)
- Squad selection (15 players) AND lineup selection (best XI + formation) as a two-stage solve

### Phase 5 — Live Decision Support

Not a model — a rules engine over pre-computed predictions:
- **Captain switcher**: given current captain's actual score and remaining players' predicted distributions, recommend stick vs. twist
- **Sub advisor**: given finished starters' actual scores and unplayed bench players' predictions, recommend manual subs (accounting for the auto-sub cancellation tradeoff)

This can be as simple as a CLI script you run between kickoff windows:

```bash
$ python -m src.live.captain_switcher --current-captain mbappé --score 3
>>> RECOMMENDATION: Switch to Kane (E[pts]=7.2, P90=13)
>>> Stick expected value: 6.0 (3 × 2)
>>> Switch expected value: 14.4 (7.2 × 2)
>>> SWITCH.
```

---

## 6. Development Workflow (Revised)

```
                         ┌──────────────────────────────────────────────┐
                         │           PRE-ROUND (batch)                  │
                         │                                              │
┌──────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│ Collector │─▶│ Features       │─▶│ Models (×4)    │─▶│ Optimizer    │ │
│ (cron)    │  │ (per position) │  │ GK/DEF/MID/FWD │  │ (squad+XI)  │ │
└──────────┘  └────────────────┘  └────────────────┘  └──────┬───────┘ │
                                                             │         │
                         └─────────────────────────────────────────────┘
                                                             │
                                                             ▼
                                                    Recommended Squad
                                                    + Captain Policy
                                                    + Bench Order
                                                             │
                         ┌──────────────────────────────────────────────┐
                         │           IN-ROUND (reactive)                │
                         │                                              │
                         │  ┌─────────────────┐  ┌──────────────────┐  │
                         │  │ Captain Switcher │  │ Sub Advisor      │  │
                         │  │ (after each      │  │ (after each      │  │
                         │  │  kickoff window) │  │  kickoff window) │  │
                         │  └─────────────────┘  └──────────────────┘  │
                         └──────────────────────────────────────────────┘
```

---

## 7. Booster (Chip) Strategy

Boosters are a tournament-level optimization problem — you have 5 chips across ~10 rounds. A simple heuristic approach:

- **Wildcard**: save for a knockout round transition where many teams get eliminated and your squad needs heavy restructuring
- **12th Man**: use on a round with a clear fixture mismatch (elite player vs. weak team, not in your squad)
- **Maximum Captain**: use on the round with the highest-variance slate (multiple premium players, unsure who to captain)
- **Qualification Booster**: use in R32 or R16 where 8+ of your XI are likely to advance (more starters = more +2 bonuses)
- **Mystery Booster**: TBD — wait for the reveal

A more sophisticated approach would simulate all possible chip-timing combinations across remaining rounds using Monte Carlo, but the heuristic gets you 90% of the way.

---

## 8. Tech Stack (Unchanged)

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.12 | ML ecosystem, you know it |
| Package mgmt | uv | Fast, modern, lockfile support |
| Data storage | Parquet + DuckDB | Zero-infra, columnar, fast |
| ML model | LightGBM (×4) | Best for tabular, small data |
| Baseline | scikit-learn (Ridge ×4) | Sanity check per position |
| Optimization | PuLP | Exact LP solver, simple API |
| Tuning | Optuna | Efficient hyperparameter search |
| Data validation | Pydantic | Catch bad data at ingestion |
| Containers | Docker + Compose | Reproducibility + deployment |
| Notebooks | Jupyter (dev only) | Exploration, not production |
| Testing | pytest | Standard Python testing |

---

## 9. What NOT to Build

- **No web scraping framework** — just functions with httpx
- **No database** — Parquet files until proven otherwise
- **No REST API** — you're the only consumer
- **No neural networks** — will underperform LightGBM here
- **No real-time ML** — live decisions use pre-computed predictions, not live inference
- **No frontend in Phase 1** — Streamlit later if wanted
- **No over-engineered chip optimizer** — heuristics first, Monte Carlo later if you want

---

## 10. Learning Roadmap (Revised)

| Phase | ML Concepts | SWE Concepts |
|---|---|---|
| 0 - Scoring | Domain modeling, test-driven development | Pure functions, exhaustive unit testing |
| 1 - Collector | Data quality, sampling bias | HTTP clients, rate limiting, Pydantic, idempotency |
| 2 - Features | Feature engineering, leakage, rolling stats, component modeling | Data pipelines, Parquet, DuckDB |
| 3 - Models | Regression, time-series CV, regularization, quantile regression, uncertainty | Model serialization, experiment tracking |
| 4 - Optimizer | Linear programming, knapsack, multi-objective optimization | Constraint modeling, stage-aware configuration |
| 5 - Live | Decision trees, expected value calculations, sequential decisions | CLI tools, state management |
| Docker | — | Dockerfile, compose, volumes, cron scheduling |

---

## 11. First Steps (When You're Ready to Code)

1. Write `scoring.py` with all four position scoring functions + tests
2. Set up Docker environment (Dockerfile + compose)
3. Hit the FIFA Fantasy API, save a raw response to Parquet
4. Build 3-4 features for one position (start with FWD — simplest scoring)
5. Train a ridge regression baseline for FWD only
6. Write the PuLP squad optimizer with dummy predictions for all positions
7. Connect real predictions to the optimizer → first real squad recommendation

That loop gets you an end-to-end MVP for one position, then you replicate for the other three.
