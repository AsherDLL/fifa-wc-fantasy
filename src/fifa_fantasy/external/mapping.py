"""Country name normalization between sources.

Sources use different country names:
- martj42 international results: full English ("United States", "South Korea")
- FIFA Fantasy API: short names ("USA", "Korea Republic")
- football-data.co.uk: league/club names (not country)

This module holds the canonical lookup table from martj42 names to the
FIFA Fantasy `country` field. The pipeline merges on either name or
country_abbr where it can; this lookup fills the gaps.

Maintain manually as new countries qualify. Missing entries cause the
join to leave country_elo NaN, which the heuristic and GBM handle.
"""
from __future__ import annotations

# martj42 → FIFA Fantasy `country` (full name as it appears in API squads).
# Only entries that differ between the two need to be listed; identical
# names fall through.
MARTJ42_TO_FIFA: dict[str, str] = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Iran": "IR Iran",
    "Republic of Ireland": "Ireland",
    "DR Congo": "Congo DR",
    "Curacao": "Curaçao",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
    # add as needed
}


def to_fifa_country(martj42_name: str) -> str:
    return MARTJ42_TO_FIFA.get(martj42_name, martj42_name)


# FPL `team_name` -> football-data.co.uk club name. Only entries that differ
# need to be listed; identical names fall through.
FPL_TO_FD_CLUB: dict[str, str] = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Sheffield Utd": "Sheffield United",
}


def to_fd_club(fpl_name: str) -> str:
    return FPL_TO_FD_CLUB.get(fpl_name, fpl_name)
