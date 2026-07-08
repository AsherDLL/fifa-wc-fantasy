# 00b - Authors and contributors

Status: **DRAFT**

This work combines computational and football-domain expertise. We list
contributions explicitly to credit which decisions came from data and
code versus which came from match-watching judgement.

## Authors

**Asher Davila** (AsherDLL on GitHub).
Computational lead. Designed the system architecture, implemented the
three predictor backends, the MILP optimiser, the live decision tools,
and the data ingestion pipeline. Conducted all held-out validation and
hyperparameter sweeps. Authored the code and the whitepaper. Operated
the personal-league fantasy entry through every matchday. Made the
final call on every transfer, every captain choice, and every formation.

**Diego Guajardo**, Actuarial Scientist.
Domain expert. Watched the WC qualifier rounds, the pre-tournament
friendlies, and the WC 2026 matches throughout. Provided per-player
judgements that the predictive models could not derive from features
alone: minutes risk for specific players based on observed manager
behaviour, in-form versus out-of-form distinctions between players at
the same model-predicted score, and tournament-narrative signals (top
scorer race incentives, must-win match dynamics). Worked the
squad-tweaking conversation through every transfer round, applying
match-observation context to the model outputs before the final pick
was made.

**Specifically credited:**

- The pre-tournament call to include Willian Pacho (Ecuador defender,
  $4.4M) in the MD1 squad came from Diego. Pacho returned 11 fantasy
  points across the group stage at a low ownership of 5.8%, a high
  return per million.
- The rotation-risk flagging across multiple matchdays in the group
  stage came from Diego (e.g. predicting which clinched-group teams
  would rest starters in MD3).
- The Captain Olise switch from Lautaro for MD2, which produced the
  squad's largest captain return (9 raw, 18 captained), came from a
  shared conversation in which Diego's read on Olise's form was the
  deciding input.

## Contribution statement

| Activity | A. Davila | D. Guajardo |
|---|---|---|
| System architecture and software | lead | reviewing |
| Model development and validation | lead | reviewing |
| Football domain knowledge (per-player, per-team) | supporting | lead |
| Transfer and captain decisions | shared | shared |
| Pre-tournament squad construction | shared | shared |
| Writing | lead | reviewing |
| Project administration | lead | none |

The computational and domain expertise are complementary. The decisions
that did well combined both signals; the decisions that did poorly
weighted one over the other unnecessarily.
