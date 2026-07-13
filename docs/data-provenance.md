# Data provenance and licensing

Every dataset redistributed in this repository, where it comes from, and
why it is committed. Everything else under `data/` is regenerable cache
and is gitignored (notably all scraped news article text, which is never
redistributed). Committed datasets exist to make a specific whitepaper
claim reproducible from a fresh clone without re-scraping.

| Dataset | Paths | Source | Terms | Why committed |
|---|---|---|---|---|
| FIFA Fantasy API snapshots | `data/raw/{players,squads,fixtures}_*.parquet` (whitelisted dates only) | `play.fifa.com/json/fantasy` (public, unauthenticated) | Factual game-state data (prices, points, fixtures); no account data involved | Decision-point snapshots behind whitepaper sections 08 and 05d |
| Derived feature/prediction tables | `data/processed/{features,predictions}_*.parquet` (whitelisted dates only) | Computed by this project from the snapshots above | Project license (GPL-3.0) | End-to-end runs and the registered semifinal pick comparison without retraining |
| FPL player-gameweek history | `data/training/fpl_player_gameweek_*.parquet` | Community mirror github.com/vaastav/Fantasy-Premier-League of the public FPL API | Mirror repo is MIT-licensed; underlying data is factual match statistics | Training and held-out validation (whitepaper 05, 07) |
| Match results and odds | `data/external/fd_matches.parquet` | football-data.co.uk free CSV archive | Free-to-use public archive | GK formula calibration (whitepaper 09c) |
| Club Elo ratings | `data/external/club_elo.csv` | clubelo.com public API | Free public API | Strength features |
| National-team Elo | `data/external/country_elo.csv` | Derived from the public results dataset github.com/martj42/international_results (CC0) | CC0 | Strength features (whitepaper 04) |
| FIFA world rankings | `data/static/fifa_rankings.csv` | FIFA.com public ranking table (facts) | Factual ranking snapshot | Fallback strength signal |
| Prediction-market snapshots | `data/external/prediction_markets/snapshot_*.jsonl` | Polymarket CLOB and Kalshi public market-data APIs | Public market data endpoints, no authentication | Market-integration negative result and Benter combiner study (whitepaper 11b, 11d) |
| Trained model files | `data/models/*.txt` | Trained by this project | Project license (GPL-3.0) | Run the shipped ensemble without retraining |
| Personal squad logs | `data/user_squads/round_*.json` | The authors' own fantasy entries | Authors' own data | Live-results ground truth (whitepaper 08) |

Rule texts: `docs/Fantasy.md` quotes short excerpts of the official game
guidelines (source and retrieval date stated in the file); the
code-aligned scoring contract lives in `docs/scoring-rules.md` and is
pinned by `tests/test_scoring.py`.

Nothing in this repository grants rights over the upstream datasets
themselves; the tables above document origin so downstream users can
honor the upstream terms. FIFA World Cup and FIFA Fantasy are trademarks
of FIFA; this is an independent research project with no affiliation.
