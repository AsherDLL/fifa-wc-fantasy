# Opponent-Strength Signals

The heuristic predictor's matchup multiplier combines two independent
signals so picks favour players whose teams are objectively stronger
than their opponents, not just better priced.

## 1. Squad price proxy (already in Phase 2)

`squad_top_n_avg_price`: the mean price of a squad's top-11 most
expensive players in the FIFA Fantasy pool. Encodes the game's pricing
model, which tracks club-league quality.

Per-row: `strength_diff = own_top_11_avg - opp_top_11_avg`. Range
roughly plus or minus 3 across the field.

## 2. FIFA Men's World Ranking (added in Phase 4.7)

`squad_rank_points`: FIFA ranking points loaded from
`data/static/fifa_rankings.csv`. The file is a hand-maintained snapshot
from `https://www.fifa.com/en/rankings/men`. The loader is in
`src/fifa_fantasy/collector/rankings.py`; it returns an empty frame if
the file is missing, so a deleted CSV falls back gracefully to the
price-only path.

Per-row: `rank_diff = own_rank_points - opp_rank_points`. Range roughly
plus or minus 700 across the field.

### Refreshing the snapshot

The CSV is the only ranking source. To update:

1. Open `https://www.fifa.com/en/rankings/men` and locate each WC 2026
   qualifier's ranking points.
2. Edit `data/static/fifa_rankings.csv` in place. Header comments are
   preserved.
3. Re-run `python -m fifa_fantasy.features` and the rest of the
   pipeline.

The country names must match the FIFA Fantasy API spelling exactly
(`IR Iran`, `Türkiye`, `Côte d'Ivoire`, etc.). The current squads list
is dumped from `data/raw/squads_<date>.parquet`.

## Blend in the heuristic

```
z_price   = strength_diff / STRENGTH_DIFF_SCALE       # ~ +/- 1.5
z_rank    = rank_diff     / RANK_DIFF_SCALE           # ~ +/- 3
combined  = 0.35 * z_price + 0.65 * z_rank
matchup   = 1 + 0.40 * tanh(combined)
```

Constants in `src/fifa_fantasy/model/baseline.py`:

| Constant | Value | Reasoning |
|---|---|---|
| `STRENGTH_DIFF_SCALE` | 2.0 | Normalizes the price gap so saturation aligns with the ranking signal. |
| `RANK_DIFF_SCALE` | 250.0 | Set so a 250-point ranking gap looks like a 2-point price gap, which roughly matches the spread of WC 2026 qualifiers. |
| `PRICE_SIGNAL_WEIGHT` | 0.35 | Lower because club-league pricing lags national-team form. |
| `RANK_SIGNAL_WEIGHT` | 0.65 | Higher because national-team rankings track national-team output. |
| `STRENGTH_DIFF_ALPHA` | 0.40 | Saturation cap. A top-vs-bottom matchup now moves the prediction by 40%, up from the prior 25%. |

If a row's `rank_diff` is NaN (no FIFA ranking for that country), the
price signal carries the full weight for that row so the rest of the
pipeline is unaffected.

## Effect on the day-1 MD1 recommendation

Before the upgrade, 14 of 15 squad members were European, driven by
price alone. After:

- Argentina vs Algeria, France vs Senegal, Spain vs Cabo Verde, and
  Germany vs Curaçao all become high-leverage matchups for clean sheets
  and goals.
- Lautaro Martinez's MD1 prediction climbs from 6.91 to 8.05; captain
  doubled output rises from 13.83 to 16.10.
- Total horizon expected points: 263.87 to 301.97 (+38).
- 9 of 15 squad members change; the formation stays 3-4-3.

The change is large by design. The user flagged the matchup signal as
first-class.
