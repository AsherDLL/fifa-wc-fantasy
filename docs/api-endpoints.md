# FIFA Fantasy WC 2026 - Public Data Endpoints

The official site (https://play.fifa.com/fantasy) is a JS-rendered SPA. Its
static gameplay data is served as plain JSON files from the same origin under
`/json/fantasy/`. These endpoints are unauthenticated and the same payloads
the browser fetches for a logged-out user opening the player pool.

| Endpoint | Purpose | Size (Jun 7, 2026) |
|---|---|---|
| `GET /json/fantasy/players.json` | Full player pool with prices, positions, ownership %, status. | ~1.1 MB, 1,481 players |
| `GET /json/fantasy/squads.json` | The 48 national teams: id, name, group, abbr, isEliminated. | ~6 KB, 48 squads |
| `GET /json/fantasy/rounds.json` | The 8 rounds (MD1, MD2, MD3, R32, R16, QF, SF, Final) with all fixtures embedded under `tournaments`. | ~74 KB, 8 rounds |
| `GET /json/fantasy/checksums.json` | MD5 hashes of `players` and `rounds`. Lets us skip a full refetch if nothing changed. | ~100 B |

Base URL: `https://play.fifa.com/json/fantasy`

## Discovery

The client-side JS at `https://play.fifa.com/fantasy/static/js/index-*.js` is
minified, with template literals composed from one- and two-character
variables. The relevant constants are:

- `Rf = "//play.fifa.com/json/"` - JSON content base
- `dR = "fantasy/"` - fantasy-specific path prefix
- `na = new ir({ baseURL: Rf + dR })` - the HTTP client for the fantasy
  static JSON
- The bundle's `na.get(…)` and `Cr.get(…)` calls reveal sibling JSON files
  (`countries.json`, `faq.json`, `settings.json`, `help_pages.json`,
  `checksums.json`). Probing siblings under `fantasy/` surfaced
  `players.json`, `squads.json`, and `rounds.json`.

## Schemas (verbatim payloads)

### `players.json` - list of player records

```jsonc
{
  "id": 1,
  "firstName": "Rayan",
  "lastName": "Aït-Nouri",
  "knownName": null,
  "squadId": 1,
  "position": "DEF",
  "price": 4.9,
  "status": "playing",           // observed: "playing", "transferred"
  "matchStatus": null,
  "percentSelected": 1.2,        // percent, divide by 100 for a fraction
  "roundsSelected": { "1": 1.2 },
  "stats": {
    "totalPoints": 0,
    "avgPoints": 0,
    "form": 0,
    "lastRoundPoints": 0,
    "roundPoints": [],
    "nextFixtureFromActiveRound": null,
    "nextFixtureFromScheduledRound": 19
  },
  "oneToWatch": false,
  "oneToWatchText": null,
  "qualificationRoundIds": [],
  "fifaId": null
}
```

Positions are `GK`, `DEF`, `MID`, `FWD`. Prices are in millions. Squad
distribution: 181 GK, 482 DEF, 512 MID, 306 FWD.

### `squads.json` - list of 48 national teams

```jsonc
{ "id": 1, "name": "Algeria", "group": "j", "abbr": "ALG", "isEliminated": false }
```

`group` is the lowercase group letter for the group stage. Knockout teams
will keep their original group letter.

### `rounds.json` - list of 8 rounds

```jsonc
{
  "id": 1,
  "status": "scheduled",
  "startDate": "2026-06-11T20:00:00+01:00",
  "endDate":   "2026-06-18T05:00:00+01:00",
  "tournaments": [
    {
      "id": 1,
      "period": "pre_match",
      "minutes": 0,
      "extraMinutes": 0,
      "venueName": "Estadio Banorte",
      "venueCity": "Mexico City",
      "venueId": 1,
      "date": "2026-06-11T20:00:00+01:00",
      "status": "scheduled",
      "isSuspended": false,
      "homeSquadId": 28,  "awaySquadId": 40,
      "homeSquadName": "Mexico", "awaySquadName": "South Africa",
      "homeSquadAbbr": "MEX",    "awaySquadAbbr": "RSA",
      "homeScore": null,         "awayScore": null,
      "homePenaltyScore": null,  "awayPenaltyScore": null,
      "homeGoalScorersAssists": null,
      "awayGoalScorersAssists": null
    }
  ]
}
```

Round ids map to stages:

| Round id | Stage |
|---:|---|
| 1 | Group MD1 |
| 2 | Group MD2 |
| 3 | Group MD3 |
| 4 | Round of 32 |
| 5 | Round of 16 |
| 6 | Quarter-finals |
| 7 | Semi-finals |
| 8 | Final |

`tournaments[].date` is the match kickoff time, ISO 8601, timezone-aware.

## Fetching

Plain unauthenticated GET, set a custom User-Agent for politeness:

```
curl -A "fifa-fantasy-collector/0.0.1" https://play.fifa.com/json/fantasy/players.json
```

The collector module (`src/fifa_fantasy/collector/`) wraps this with
`httpx`, validates with Pydantic, and writes Parquet under `data/raw/`.
