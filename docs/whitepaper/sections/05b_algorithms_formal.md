# 05b - Algorithms (formal specifications)

Status: **DRAFT**

This section gives precise mathematical and pseudocode specifications
for the three predictor backends, the MILP optimiser, and the
country-Elo rolling update. Each is implemented in code under
`src/fifa_fantasy/`; the references below point to the canonical source.

## 5b.1 Heuristic predictor

**Source:** `src/fifa_fantasy/model/baseline.py`.

For each (player, round) row in the feature table, the heuristic
returns a predicted points value:

```
y_hat = c_pos * P * (1 + α tanh(z_strength)) * (1 + β · 1[home]) + γ · max(0, P − P_threshold)
```

where:

| Symbol | Meaning | Default |
|---|---|---|
| `c_pos` | per-position coefficient | GK 0.50, DEF 0.55, MID 0.60, FWD 0.65 |
| `P` | player's price in millions | from FIFA API |
| `z_strength` | blended strength signal (see below) | derived |
| `α` | matchup saturation | 0.40 (caps at ±40% adjustment) |
| `β` | home advantage | 0.05 (caps at +5%) |
| `γ` | premium-tier boost | 0.0 default |
| `P_threshold` | premium boundary | 9.0M |

Strength signal `z_strength` follows a priority hierarchy:

```
z_elo   = country_elo_diff / 400         (where present; from martj42)
z_rank  = rank_diff / 250                (where present; static FIFA rank)
z_price = strength_diff / 2              (always available; top-11 average price gap)

if elo present:
    z_strength = 0.35 z_price + 0.65 z_elo
elif rank present:
    z_strength = 0.35 z_price + 0.65 z_rank
else:
    z_strength = z_price
```

Players with `status != 'playing'` or `is_eliminated == True` receive 0.

**Algorithm (pseudocode):**

```
def heuristic_predict(features):
    for row in features:
        if row.status != 'playing' or row.is_eliminated:
            row.predicted_points = 0
            continue
        c = position_coefficient[row.position]
        z = combined_matchup_z(row)
        matchup = 1 + alpha * tanh(z)
        home = 1 + beta * row.is_home
        boost = premium_boost * max(0, row.price - premium_threshold)
        row.predicted_points = c * row.price * matchup * home + boost
    return features
```

## 5b.2 Poisson goals model

**Source:** `src/fifa_fantasy/model/poisson.py`.

Step 1: estimate per-team expected goals for the fixture.

```
matchup = strength_component + price_diff / 6
where strength_component = 
    elo_diff / 400      if country_elo_diff is present
    rank_diff / 800     elif rank_diff is present
    0                   otherwise

own_factor = exp(clip(matchup, -1.2, 1.2)) · (1 + 0.10 · 1[own_home])
opp_factor = exp(-clip(matchup, -1.2, 1.2)) · (1 + 0.10 · 1[opp_home])

own_xg = 1.30 · own_factor
opp_xg = 1.30 · opp_factor
```

Step 2: distribute team xG across positions by share.

```
GOAL_SHARE   = {GK: 0.00, DEF: 0.10, MID: 0.30, FWD: 0.50}
ASSIST_SHARE = {GK: 0.01, DEF: 0.18, MID: 0.50, FWD: 0.31}

player_xg = own_xg · GOAL_SHARE[position]
player_xa = own_xg · 0.7 · ASSIST_SHARE[position]
```

Step 3: clean-sheet probability under Poisson.

```
P(clean sheet) = exp(-opp_xg)         # P(opp scores 0)
E(extra conceded) = max(0, opp_xg - 1 + exp(-opp_xg))   # E(max(0, k-1)) under Poisson(opp_xg)
```

Step 4: sum components into expected fantasy points.

```
GOAL_POINTS     = {GK: 9, DEF: 7, MID: 6, FWD: 5}
CS_POINTS       = {GK: 5, DEF: 5, MID: 1, FWD: 0}
APPEARANCE_POINTS = 2     (60-minute baseline)
ASSIST_POINTS   = 3
GK_SAVE_BONUS_PER_OPP_XG = 0.50   (expected save points per unit of opponent xG; see 09c)
DEF_GC_FACTOR   = 0.5     (DEF concedes-after-first penalty rate)

y = APPEARANCE_POINTS
  + player_xg · GOAL_POINTS[position]
  + player_xa · ASSIST_POINTS
  + P(clean sheet) · CS_POINTS[position]
  + (1[position == GK]) · opp_xg · GK_SAVE_BONUS_PER_OPP_XG
  + (1[position == GK]) · (-E(extra conceded))
  + (1[position == DEF]) · DEF_GC_FACTOR · (-E(extra conceded))

predicted_points = max(0, y) if available else 0
```

The scouting bonus (+2 if predicted > 4 and ownership < 0.05) is not
baked into the Poisson sum; it is applied exactly once for every
backend, downstream in `optimizer/pipeline.apply_scouting_bonus`.

## 5b.3 LightGBM v2

**Source:** `src/fifa_fantasy/model/gbm.py`.

Per-position regressor with four heads.

**Features** (in order):

```
X = [price_millions, is_home (0/1), strength_diff,
     squad_top_n_avg_price, opp_squad_top_n_avg_price, form_lag]
```

**Targets:** `total_points` per (player, gameweek) row from FPL data.

**Heads:**

```
out['mean'] = train_lgbm(objective='regression', metric='rmse')
out['q10']  = train_lgbm(objective='quantile', alpha=0.10, metric='quantile')
out['q50']  = train_lgbm(objective='quantile', alpha=0.50, metric='quantile')
out['q90']  = train_lgbm(objective='quantile', alpha=0.90, metric='quantile')
```

**Hyperparameters** (from `tune.py` sweep on EPL 2024-25 GW 30-38):

```
num_leaves          = 15
learning_rate       = 0.05
n_estimators        = 200
min_child_samples   = 30
feature_fraction    = 0.9
bagging_fraction    = 0.9
bagging_freq        = 5
seed                = 42       (for determinism)
bagging_seed        = 42
feature_fraction_seed = 42
deterministic       = True
```

**Inference contract:** the inference path drops `rank_diff` from the
column list because EPL training has no rank_diff column; the same six
features are used at inference.

**Prediction post-processing:**

```
predicted_points = max(0, mean head prediction)         if available else 0
predicted_q10    = max(0, q10 head prediction)
predicted_q50    = max(0, q50 head prediction)
predicted_q90    = max(0, q90 head prediction)
```

## 5b.4 MILP optimiser

**Source:** `src/fifa_fantasy/optimizer/solvers.py`.

**Constants:**

```
SQUAD_SIZE = 15
SQUAD_POSITION_COUNTS = {GK: 2, DEF: 5, MID: 5, FWD: 3}
VALID_FORMATIONS = {
    '4-4-2': (4, 4, 2), '4-3-3': (4, 3, 3), '4-5-1': (4, 5, 1),
    '3-4-3': (3, 4, 3), '3-5-2': (3, 5, 2),
    '5-4-1': (5, 4, 1), '5-3-2': (5, 3, 2),
}
TRANSFER_HIT_POINTS = 3
```

**Squad selection (fresh):**

```
Variables: x_p ∈ {0, 1} for each candidate player p

Maximize:  Σ_p effective_points[p] · x_p

Subject to:
    Σ_p x_p = SQUAD_SIZE
    For each position pos: Σ_{p: position[p]=pos} x_p = SQUAD_POSITION_COUNTS[pos]
    Σ_p price[p] · x_p ≤ budget_millions
    For each country c: Σ_{p: country[p]=c} x_p ≤ max_per_country
    x_p = 0 if is_eliminated[p]
```

**Transfer with hit accounting:**

```
Variables: x_p ∈ {0, 1}, extra ∈ R⁺

Maximize:  Σ_p effective_points[p] · x_p − TRANSFER_HIT_POINTS · extra

Subject to (all squad-selection constraints, plus):
    new_picks = Σ_{p ∉ current_squad} x_p
    new_picks ≤ free_transfers + extra
```

**Starting XI:**

```
Variables: y_p ∈ {0, 1}, f_F ∈ {0, 1} for each formation F

Maximize:  Σ_p predicted_points[p] · y_p

Subject to:
    Σ_F f_F = 1                                  (exactly one formation)
    Σ_p y_p = 11                                 (exactly 11 starters)
    Σ_{p: GK} y_p = 1
    For each position {DEF, MID, FWD}:
        Σ_{p: pos} y_p = Σ_F f_F · formation_counts_F[pos]

Captain = argmax_{p: y_p = 1} predicted_points[p]
Vice    = second_argmax
```

The solver's captain is a mean-argmax placeholder. In production the
optimiser overrides it with the ceiling-aware composite selector in
`optimizer/captain.py`, which interpolates between the mean and the q90
ceiling by the user's standings percentile (Section 11g.3).

## 5b.5 Country Elo rolling update

**Source:** `src/fifa_fantasy/external/international_elo.py`.

Starting from BASE_ELO = 1500 for every country, roll forward through
the chronologically sorted match history:

```
For each match m in sorted order:
    e_h = Elo[home_team]  (or BASE_ELO if first match)
    e_a = Elo[away_team]
    adv = 0 if m.neutral else HOME_ADVANTAGE (= 60)

    expected_home = 1 / (1 + 10^((e_a - (e_h + adv)) / 400))
    expected_away = 1 - expected_home

    score_home = 1.0 if home_goals > away_goals
               = 0.5 if home_goals == away_goals
               = 0.0 if home_goals < away_goals
    score_away = 1 - score_home
    if scores missing: score_home = score_away = 0.5

    K = K_factor(m.tournament)
        = 60 if 'fifa world cup' in tournament name (and not qualifier)
        = 25 if 'qualif' in tournament name
        = 10 if 'friendly' in tournament name
        = 40 otherwise

    margin = max(1, |home_goals - away_goals|)
    gd_mult = sqrt((margin + 1) / 2)

    Elo[home_team] = e_h + K · gd_mult · (score_home - expected_home)
    Elo[away_team] = e_a + K · gd_mult · (score_away - expected_away)
```

After processing the full history (every international match since
1872), the per-country snapshot is the most recent post-match Elo for
that country.

## 5b.6 Club Elo for training (without lookahead bias)

**Source:** `src/fifa_fantasy/external/football_data.py`,
`compute_club_elo_history`.

Same Elo rolling update as 5b.5 but on football-data.co.uk per-league
match CSVs. Stores the BEFORE-MATCH Elo for both home and away club at
every match date. Training-side joins use `merge_asof` with
`direction='backward'` so an EPL training row at date `D` joins to the
latest Elo snapshot strictly before `D`. This prevents lookahead bias:
the model sees only Elo information that was available before the
match it is predicting.

K-factor: fixed at 20 for club football (less volatile than
international tournament weighting).

## 5b.7 The blended "final score" for live decisions

**Source:** `scripts/md3_dembele_locked.py`, `scripts/r32_rebuild.py`.

For live tournament decisions during the WC, the per-(player, round)
scoring blend is:

```
final = 0.55 · model_ensemble + 0.30 · realised_avg_per_match + 0.15 · form_score
```

where:

- `model_ensemble = (heuristic + GBM) / 2` (Poisson excluded because it
  systematically over-predicts in live runs post-MD2 due to scale
  recalibration after the WC kickoff)
- `realised_avg_per_match = total_points / matches_played_so_far`
- `form_score = FIFA-API-derived form value` (rolling average per match)

The blend weights are hand-tuned in the live scripts, not learned. The
0.55/0.30/0.15 split was chosen by inspection at MD3: realised data
should dominate over model predictions once a player has 2+ games of
realised output. A formalisation (Bayesian posterior with shrinkage)
is in Section 11.

## 5b.8 Rotation risk multiplier

**Source:** `scripts/md3_plan.py`, `scripts/r32_rebuild.py`.

A per-country scalar in [0, 1] applied to model predictions for
group-stage matchdays where a country has already clinched
qualification:

```
adj = (heuristic + GBM) / 2 · ROT[country]
```

where ROT is hand-set per country per matchday:

| Situation | ROT |
|---|---|
| Clinched top spot + easy opponent | 0.55-0.65 |
| Clinched + top-spot battle | 0.80-0.93 |
| Needs result | 1.00 |
| Eliminated | 0.70-0.85 |

A protect-list override applies ROT = 1.0 for individual players in the
top-scorer race (Messi, Mbappé) regardless of country rotation:

```
PROTECT_NAMES = {Lionel Messi, Kylian Mbappé, Lautaro Martínez,
                 Cristiano Ronaldo, Harry Kane, Bukayo Saka,
                 Mohamed Salah, Erling Haaland}
```

This is acknowledged as a code smell in Section 10.2; a proper
rotation-risk model is future work in Section 11.
