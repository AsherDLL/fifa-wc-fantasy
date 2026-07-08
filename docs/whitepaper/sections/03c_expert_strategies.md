# 03c - Expert practitioner strategies and how our approach compares

Status: **DRAFT**

This section surveys what top fantasy-football practitioners and
specialised fantasy-data sites recommend, then maps our methodology
against their strategies. The goal is to position our paper not just
in the academic literature (Section 3b) but against the actual
community of practice.

## 3c.1 Sources consulted

| Source | Type | Focus |
|---|---|---|
| Fantasy Football Scout (`fantasyfootballscout.co.uk`) | Editorial + analytics | FPL + WC fantasy |
| All About FPL (`allaboutfpl.com`) | Editorial | FPL + WC fantasy + booster strategy |
| Fantasy Football Hub (`fantasyfootballhub.co.uk`) | Editorial + tooling | WC ultimate guides |
| Frontier Economics (`frontier-economics.com`) | Data science research | FPL strategy via ML |
| FPL Captain (`fplcaptain.com`) | AI-driven tooling | Per-gameweek captain picks |
| FourFourTwo | Editorial | WC fantasy team building |
| Premier Fantasy Tools | Analytics platform | xPts-driven captain selection |
| FPL Copilot | Editorial | Captain strategy framework |
| Premier League FPL Champion column | Editorial | Tactics from former #1 manager |

## 3c.2 What experts say about captain selection

Captain selection is universally agreed to be the single
highest-leverage decision per round. Consensus expert advice:

1. **Pick by Expected Points (xPts)**: the highest projected player.
2. **Anchor on Form**: a player in scoring/assisting form is safer than
   one with a soft fixture and dry form.
3. **Fixture matters**: a premium attacker vs the league's weakest
   defense at home is the safest captain.
4. **Use advanced stats** (xG, xA, touches in the box) as signal
   beyond raw points.
5. **Check for rotation/injury risk** via pre-match press conferences.
6. **In Double Gameweeks** (or knockout fixtures with two-leg
   features): captain players with two matches.
7. **Adjust to standings position**:
   - If **leading**, captain the consensus pick (defensive, match
     the field).
   - If **chasing**, captain a low-ownership differential (offensive,
     swing the field).
8. **Strategic Vice**: always the next-highest xPts, never "safe".

The Premier League's published "FPL Champion" advice and Frontier
Economics' data-science research both confirm: the consensus pick is
the **statistically right move** for managers near the top of their
league. Differential captaining is a chasing strategy with
diminishing returns past one or two maverick picks.

## 3c.3 What experts say about WC 2026 booster strategy

The five WC 2026 boosters and their popular vs differential timings
(per AllAboutFPL's comprehensive guide):

| Booster | Function | Popular timing | Differential timing |
|---|---|---|---|
| **Wildcard** | Unlimited transfers in one round | Round 3 (resets pre-knockouts) | Round of 16 / Quarter-finals |
| **Maximum Captain** | Doubles highest XI scorer (auto-captain) | Final (small candidate pool) | Semi-final (respond to surprises) |
| **12th Man** | Add an extra player outside budget | Round 1-2 (group stage point pool) | Round 3 (combine with rolled transfers) |
| **Qualification Booster** | +2 pts to advancing players | Round of 32 (clear favourites) | Round of 16 |
| **Mystery Booster (Clean Sheet Shield)** | Defenders don't lose CS bonus until 2+ conceded | Round of 16 | Quarter-finals onward |

We have already used the Wildcard at R32 (the differential timing per
the guide), which respects the principle of holding flexibility for
unpredictable knockout outcomes. The remaining four boosters are
unused at the time of writing.

## 3c.4 What our approach does that experts do

Mapping our methodology against expert practice:

| Expert recommendation | Our system | Status |
|---|---|---|
| Pick by xPts | Yes - three backends each produce `predicted_points` | yes |
| Anchor on form | Monte Carlo backend explicit form multiplier; heuristic via realised data blend in live scripts | yes |
| Fixture matters | Country Elo signal (from martj42) + opponent strength via top-11 price | yes |
| Advanced stats (xG, xA) | Poisson backend uses team xG; we don't have player-level Opta xG | partial |
| Rotation/injury risk | Per-country rotation multiplier in scripts; no team-news scraper | partial (Section 11) |
| Double gameweek captaining | Not applicable in WC group stage; will surface in knockout 2-legged ties if used | N/A |
| Standings-aware differential | Documented framework in Section 10.4; not yet automated in optimiser | partial |
| Vice = next-highest xPts | Yes - lineup solver picks second-highest predicted scorer | yes |
| Booster strategy | Documented in this section; we have used Wildcard | partial |

## 3c.5 What our approach does that experts don't

Five things that distinguish our system from the expert practitioner
content reviewed:

1. **Cross-domain transfer learning** (Section 9). Training a GBM
   model on EPL FPL data and applying it to WC international play.
   None of the expert sites discuss this distribution-shift problem
   explicitly. They use season-specific FPL data only.

2. **Held-out RMSE validation gate** (Section 7). We measure backend
   accuracy against a withheld portion of training data before
   shipping. Expert sites publish predictions without auditing
   accuracy retrospectively in a methodologically rigorous way.

3. **MILP-based optimal squad selection** (Section 5.7). Most expert
   sites pick squads by editorial judgement plus per-position tier
   tables; they do not use mixed-integer linear programming under
   formal stage constraints. The Mishra & Mishra (2025) arXiv paper
   (Section 3b) is the closest academic analogue.

4. **Prediction-market integration via Benter combiner** (Section 11c).
   No expert site we found combines prediction-market data (Polymarket,
   Kalshi) with their fantasy projections. This is genuinely new for
   fantasy football.

5. **Cross-round retrospective backtest** (Section 8b). Expert sites
   publish per-round projections and per-round retrospective
   reviews, but do not systematically measure their cumulative
   accuracy or compare alternate model versions against their own
   recommendations.

## 3c.6 What experts do that we don't (and should consider)

1. **Mid-round captain switching**. FIFA Fantasy permits captain
   changes during a round as long as the new captain has not yet
   played and the previous captain's match is complete. Expert
   editorial content recommends "stick or twist" decision-making
   after the early-window matches finish. We do not have a live
   decision tool that automates this; we mention it in Section 5.8
   as a manual `live` module call.

2. **Press conference monitoring**. Expert sites curate predicted XIs
   from manager press conferences (a day before kickoff). We do
   not. This is Section 11.2 future work.

3. **Tier-based affordability framing**. Editorial content groups
   players into price tiers (Tier 1: $10M+, Tier 2: $8-10M, etc.) and
   discusses cross-tier swaps explicitly. Our MILP optimiser treats
   price as a continuous constraint and does not surface tier
   discussions.

4. **Booster timing as a separate decision layer**. We treat boosters
   reactively (used Wildcard at R32) rather than planning their
   deployment across the tournament in advance. A booster-strategy
   module is Section 11 future work.

5. **Player-level expected stats (xG, xA, big-chances-missed)**.
   Expert sites pull these from Opta or Understat. The football-
   data.co.uk source we use is match-level only; for player-level
   advanced stats we'd need a paid Opta subscription or an Understat
   community mirror.

## 3c.7 The "go template or go differential" debate

This is the central tactical question and the literature gives a
sharp answer (Frontier Economics, 2024):

- For a manager **leading** their league, the optimal captain is
  the **consensus pick** (highest-owned eligible player). Differentials
  add risk without expected value.
- For a manager **modestly behind** (10-30 points), one **maverick**
  pick balances catch-up potential with floor.
- For a manager **substantially behind**, multiple mavericks are
  warranted; "adding a second maverick increased chances of catching
  up by 22 percentage points, vs. an additional safe bet increasing
  chances by 18pp."
- Beyond two mavericks, returns diminish materially.

We have framed this correctly in Section 10.4 (Portfolio construction,
3-4 anchors + 5-7 workhorses + 3-5 differentials). The Frontier
result quantifies it; the rule of thumb is **two mavericks** for
chasing managers.

## 3c.8 Specific takeaway for our paper's positioning

The paper's contribution is **NOT** that we invented expert-grade
fantasy advice. The expert sites we reviewed deliver actionable
per-week recommendations that are excellent for their target
audience. Our contribution is the **methodological framework**:

- Systematic backend comparison via held-out RMSE and live backtest
- Cross-domain transfer learning challenges and their consequences
- Prediction-market integration via the Benter combiner
- MILP-based optimisation under formal constraints
- Documented decision log with model-vs-human comparison

These are research artefacts, not weekly-update content. The
**precedent for future fantasy contests** the user requested is
this methodological scaffolding, validated empirically through one
World Cup. Future practitioners can plug in their own scoring rules,
data sources, and prediction-market access to apply the same
framework to other tournaments or fantasy games.
