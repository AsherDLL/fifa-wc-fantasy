"""Resolve the user's MD1 squad to player_ids and print MD1 actuals."""
import unicodedata
import pandas as pd

players = pd.read_parquet("data/raw/players_2026-06-18.parquet")

def strip_accents(s):
    if not isinstance(s, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()

players["norm_name"] = players["full_name"].map(strip_accents)

wanted = [
    ("martinez", "ARG", "GK",  "Start GK"),     # Emiliano "Dibu"
    ("james",    "ENG", "DEF", "Start DEF"),
    ("pacho",    "ECU", "DEF", "Start DEF"),
    ("tah",      "GER", "DEF", "Start DEF"),
    ("raum",     "GER", "DEF", "Start DEF"),
    ("doue",     "FRA", "MID", "Start MID"),
    ("fernandez","ARG", "MID", "Start MID"),
    ("olise",    "FRA", "MID", "Start MID"),
    ("dembele",  "FRA", "MID", "Start MID"),
    ("gakpo",    "NED", "FWD", "Start FWD"),
    ("lautaro",  "ARG", "FWD", "CAPTAIN FWD"),
    ("penders",  None,  "GK",  "Bench GK"),
    ("dalot",    "POR", "DEF", "Bench DEF"),   # likely Diego Dalot (Portugal)
    ("watkins",  "ENG", "FWD", "Bench FWD"),
    ("williams", "ESP", "MID", "Bench MID"),
]

def md1(rp):
    if rp is None:
        return 0
    L = list(rp)
    return L[0] if L else 0

print(f"{'role':<12} {'name':<28} {'pos':<4} {'cty':<5} {'price':>6} {'own%':>5} {'MD1':>4} {'form':>5} {'status':<10}")
print("-" * 95)
ids = []
for needle, cty, pos, role in wanted:
    m = players[
        players["norm_name"].str.contains(needle, na=False)
        & (players["position"] == pos)
    ]
    if cty:
        m = m[m["country_abbr"] == cty]
    if m.empty:
        print(f"{role:<12} MISS  {needle:<22} {pos:<4} {cty or '?':<5}")
        continue
    for r in m.itertuples():
        ids.append((role, int(r.player_id)))
        print(f"{role:<12} {r.full_name:<28} {r.position:<4} {r.country_abbr:<5} {r.price_millions:>6.1f} {r.ownership_fraction*100:>5.1f} {md1(r.round_points):>4} {r.form:>5.1f} {r.status:<10}")

print()
print("SQUAD IDS:", [i for _, i in ids])
