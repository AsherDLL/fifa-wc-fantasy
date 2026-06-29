# 08 — Live tournament results

Status: **DRAFT** (grows match-by-match through the tournament)

This section logs every matchday of the WC 2026 along with the model's
recommendation, the user's actual decisions, the realised points, and a
short post-hoc note on what worked and what did not.

## 8.1 Pre-tournament squad selection (group stage MD1)

Squad assembled from a combination of the three model backends'
recommendations and human domain priors. The user took the heuristic
backend as the primary signal, with Poisson and GBM as cross-checks.

Captain: Lautaro Martínez (ARG FWD, $8.8M, 3.2% ownership).

Realised result: **33 points.** Last in the personal league.

Diagnosis:

- The captain blanked (Lautaro 1 point, doubled to 2). Captain hits are
  the single largest variance source.
- The France triple-stack (Doué, Olise, Dembélé at $7.5M, $9.5M, $10M)
  shared output instead of concentrating: 12 total points across three
  premium midfield slots.
- The squad had four defenders, none of whom kept a clean sheet.
  Argentina kept a clean sheet but no premium Argentine defender was
  paired with Emi Martínez at GK.

## 8.2 MD2 transfer round

Two free transfers. The optimiser's best 2-transfer move was to drop
Olise + Pacho and bring in Saka + Ibañez. The user disagreed strongly
with the Olise drop (Olise had been the squad's top MD1 scorer at 6
points and is a tournament-favourite winger). The conversation that
followed produced a counterproposal: Lautaro + Doué OUT, Messi + Barcola
IN, captain Messi.

The user opted for a hybrid: Lautaro OUT, Olise to vice captain, captain
swap to Olise.

Realised result: **102 points.** Massive improvement on MD1.

Highlights:

- Olise captain: 9 raw, 18 captained
- Dembélé: 12 points, justifying the keep
- Gakpo: 19 points (NED was a top group)
- James + Pacho: 9 and 9 (defensive returns when group survival was
  contested)

Diagnosis:

- The model was right that Olise had future ceiling. The user was right
  that dropping him was wrong.
- Form anchoring beats model EV for premium players in form, by exactly
  the amount the captain x2 multiplier suggests it should.

## 8.3 MD3 transfer round

Two free transfers plus the user took an additional hit. Final moves
included bringing in Daniel Muñoz, Alexander Freeman, Ronald Araújo, and
Deniz Undav while keeping Messi (captain). The user's intuition on
rotation in clinched matches was largely right; the model's was too
conservative.

Realised result: **42 points (after the -6 transfer hit).**

Diagnosis:

- Many starters did not play in clinched-group matches (rotation). The
  model did not have a strong rotation-risk signal beyond a country-level
  multiplier.
- The user's "Pacho out because Mexico beats Ecuador" call was correct
  in spirit; Mexico did win and Pacho's defensive return was modest.
- Captain Messi (14 captained, 7 raw vs Cape Verde... wait MD3 was
  Argentina vs Jordan; Messi played, scored modestly) -- captain choice
  was defensible but Olise vice was not in form.

## 8.4 R32 (Wildcard rebuild)

[Update after R32 results are in.]

Wildcard used: unlimited transfers in this round, no point hits. The
squad was rebuilt almost from scratch using the post-MD3 candidate pool.

Final squad:

| Pos | Player | Cty | $M | MD3-eve total pts | Own% |
|---|---|---|---:|---:|---:|
| GK | Rangel | MEX | 3.9 | 21 | 6.2% |
| GK | Beach | AUS | 3.5 | 21 | 3.8% |
| DEF | Laporte | ESP | 5.5 | 28 | 8.0% |
| DEF | Cucurella | ESP | 5.1 | 24 | 25.6% |
| DEF | Muñoz | COL | 4.6 | 24 | 9.1% |
| DEF | Nuno Mendes | POR | 5.8 | 24 | 36.9% |
| DEF | Freeman | USA | 4.0 | 24 | 6.4% |
| MID | Dembélé | FRA | 10.0 | 35 | 19.7% |
| MID | Vinícius | BRA | 10.0 | 35 | 32.2% |
| MID | Bellingham | ENG | 8.3 | 26 | 14.6% |
| MID | Olise | FRA | 9.5 | 17 | 34.1% |
| MID | Manzambi | SUI | 5.6 | 29 | 2.0% |
| FWD | Messi | ARG | 10.0 | 40 | 37.8% |
| FWD | Haaland | NOR | 10.5 | 30 | 28.0% |
| FWD | Undav | GER | 6.6 | 29 | 3.3% |

Total cost: $105.6M / $107.5M.

Captain: **Dembélé** (user's choice; the model marginally preferred
Messi but the differential argument for Dembélé at 19.7% ownership vs
Messi at 37.8% was strong).

Realised result: [PENDING — update after R32 matches conclude].

## 8.5 R16, QF, SF, Final

[To be added as each stage completes.]

## 8.6 Season summary table (to compile at the end)

A final cross-stage table with one row per matchday:

| Round | Model captain pick | User captain | Realised user score | Field median | Captain points |
|---|---|---|---:|---:|---:|

Plus a per-stage accuracy report for the three backends (predicted
ranking correlation with realised ranking).
