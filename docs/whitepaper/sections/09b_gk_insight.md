# 09b - The goalkeeper paradox: defensive strength as a fantasy penalty

Status: **DRAFT**

This section documents a counter-intuitive finding from our live
tournament results: the **best real-world goalkeepers were not the best
fantasy goalkeepers, and the model structurally failed to predict this**.

## 9b.1 The empirical observation

Through the group stage, two goalkeepers from advancing teams illustrate
the paradox:

| Goalkeeper | Country | Real-world quality | Fantasy total (3 games) |
|---|---|---|---|
| Emiliano "Dibu" Martínez | Argentina | World's top GK by Golden Glove and Yashin Trophy 2022-2024 | 14 pts |
| Raúl Rangel | Mexico | Domestic-league starter, not world-class | 21 pts |

Despite Dibu being widely regarded as the world's best goalkeeper,
Rangel **outscored him by 50%** in fantasy points across the group
stage. Why?

## 9b.2 The mechanics of fantasy GK scoring

FIFA Fantasy goalkeeper scoring rewards three distinct components:

1. **Appearance** (60+ minutes): +2 pts
2. **Clean sheet** (no goals conceded with 60+ minutes): +5 pts
3. **Saves**: +1 pt per 3 saves
4. **Penalty saves**: +5 pts each
5. **Goals conceded**: −1 pt per goal after the first

The save bonus is multiplicative in shots faced. A goalkeeper whose
team faces 12 shots and makes 7 saves earns +2 from saves alone. A
goalkeeper whose team faces 3 shots and makes 1 save earns 0.

## 9b.3 The paradox: defensive strength reduces save opportunities

Argentina's defense (Pacho, Otamendi, Romero, Tagliafico, plus
defensive midfielders De Paul, MacAllister) is the strongest in
recent international football. Through three group games:

- Total goals conceded: 0
- Total opponent shots on target: approximately 3-5 per match
- Total Dibu saves: approximately 1-2 per match

Mexico's defense is competent but not elite. Through three group games:

- Total goals conceded: 0 (matched ARG on shutout count)
- Total opponent shots on target: approximately 5-8 per match
- Total Rangel saves: approximately 4-6 per match

**Both goalkeepers achieved clean sheets at the same rate. Rangel
earned the additional save bonus that Dibu did not because Rangel's
team gave up more shots.**

The "best real-world goalkeeper" is the one who minimises goals
conceded per shot faced. The "best fantasy goalkeeper" is the one
who **faces enough shots to accumulate save bonuses while still
shutting them all out**. These are different optimisation problems.

## 9b.4 Our model's structural failure

The Poisson backend (`src/fifa_fantasy/model/poisson.py`) encodes
the GK save bonus as a **flat constant**:

```
GK_SAVE_BONUS = 1.0          # expected points from save bonuses
```

This is wrong. The correct formulation scales the save bonus by
expected shots faced, which is proxied by opponent xG:

```
shots_faced ≈ opp_xg × (1 / shot_to_goal_conversion)   ≈ opp_xg × 4
expected_saves ≈ shots_faced × save_percentage         ≈ 0.85 × shots_faced
save_bonus = expected_saves / 3                        # one pt per 3 saves
           ≈ opp_xg × 1.13                             # empirically
```

Under the corrected formula:

- Dibu vs CPV (opp_xg ≈ 0.4): expected save bonus ≈ 0.45 pts
- Rangel vs ECU (opp_xg ≈ 0.7): expected save bonus ≈ 0.80 pts

That difference, accumulated over the group stage, predicts the
observed 7-point gap between the two goalkeepers.

The heuristic backend has a similar structural problem: its GK
prediction is `position_coef × price × matchup_factor`, where the
matchup factor over-rewards strong own-team strength (making
Argentina players score higher), but does not properly account for
the fact that a strong own-team REDUCES the GK's save opportunities.

## 9b.5 What this implies for picking goalkeepers in fantasy

The optimal fantasy GK profile, derived from the above:

1. **Plays for a team that keeps clean sheets** (clean sheet bonus
   is the largest single component, +5 pts)
2. **Faces moderate opponent xG** (0.5 to 1.0 range): enough to
   accumulate save bonuses, not enough to break the clean sheet
3. **Is the established #1** (avoids rotation risk)
4. **Has a cheap price tag** (frees budget for attacking premiums)

Concrete examples from R32:

- **Rangel (MEX, $3.9M, vs ECU)**: high save opportunity + likely
  clean sheet + cheap. Strong pick.
- **Beach (AUS, $3.5M, vs EGY)**: similar profile, even cheaper.
- **Dibu Martínez (ARG, $5.0M, vs CPV)**: high price, very low save
  opportunity (CPV will not get many shots on target), only +5
  from clean sheet. Overpriced.

This explicitly contradicts a "buy the best real-world goalkeeper"
heuristic.

## 9b.6 Why our optimiser still got it right at R32

The MILP optimiser picked Rangel as starting GK and Beach as bench
GK for R32, choosing Rangel's lower price + similar clean sheet
expectation over Dibu's higher price. The decision was correct, but
**for the wrong reason**: the optimiser preferred Rangel because the
heuristic backend gave him a higher `predicted_points` value through
the matchup-factor times price formula, not because the model
understood the save-opportunity dynamics. The right answer was
reached through a model artifact rather than through the right
mechanism. A different parameterisation could just as easily have
preferred Dibu.

## 9b.7 Fix proposed for R16 onward

The Poisson backend will be amended:

```
GK_SAVE_BONUS_PER_OPP_XG = 1.13     # empirical scaling
gk_save_bonus = GK_SAVE_BONUS_PER_OPP_XG * opp_xg
```

The heuristic backend will gain an asymmetric position adjustment for
goalkeepers: when own-team strength is high, the matchup factor
should be **deflated** for GKs, not inflated, because high own-team
strength reduces save opportunities while only modestly increasing
clean sheet probability (which is already close to the ceiling for
elite defences).

These changes will be validated on the EPL 2024-25 held-out set
before being shipped. If the held-out RMSE for GK improves, they
land. If not, the lesson is documented as a known model bias.

## 9b.8 Generalisation

The GK paradox is a specific instance of a broader phenomenon: **the
fantasy game's scoring rules create non-monotonic relationships
between real-world football quality and fantasy points**. Other
examples we observed during the tournament:

- **Centre-backs in a deep defensive line**: more interceptions and
  blocks per game, but lower scoring chance contributions. Sometimes
  fantasy-points-rich.
- **Wingers in possession-dominant teams**: high model prediction
  from price and matchup, but actual goal/assist contributions
  often modest because the build-up play distributes credit among
  many players.
- **Sub-only forwards on great teams**: zero in fantasy unless they
  enter the game; the model's price-based estimate is materially
  wrong because it implicitly assumes minutes.

A general fix would require explicit minutes modelling and a richer
notion of "expected fantasy points per minute on the pitch". This is
listed in Section 11 future work.
