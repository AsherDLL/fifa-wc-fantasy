"""Fuzzy-match scraped player names to FIFA Fantasy player_ids.

Strategy: lowercase + strip accents + match against (country_abbr, position
hint, last name primary) on the latest players_<date>.parquet.

Confidence tiers:
  1.00 - exact full-name match
  0.85 - last-name match within (country, position)
  0.70 - last-name match within country (any position)
  0.50 - last-name match without country (likely false positive)
  0.00 - no match
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    decomposed = unicodedata.normalize("NFD", s)
    no_accents = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return no_accents.lower().strip()


@dataclass(frozen=True)
class MatchResult:
    player_id: int | None
    confidence: float
    notes: str


class PlayerNameMatcher:
    """Resolve raw scraped player names to FIFA Fantasy player_ids.

    Build once from the latest players_<date>.parquet; reuse for every
    parsed lineup in the same run.
    """

    def __init__(self, players: pd.DataFrame):
        self.players = players.copy()
        self.players["full_norm"] = self.players["full_name"].map(_normalize)
        self.players["known_norm"] = (
            self.players.get("known_name", pd.Series([""] * len(self.players)))
            .fillna("").map(_normalize)
        )
        # Last name = the last token of full_name.
        self.players["last_norm"] = (
            self.players["full_norm"].str.split(r"\s+").str[-1]
        )
        self._by_country = self.players.groupby("country_abbr")

    @classmethod
    def from_latest(cls, raw_dir: Path = Path("data/raw")) -> "PlayerNameMatcher":
        files = sorted(raw_dir.glob("players_*.parquet"))
        if not files:
            raise FileNotFoundError(f"no players_*.parquet in {raw_dir}")
        return cls(pd.read_parquet(files[-1]))

    def match(self, raw_name: str, country_abbr: str | None = None,
              position: str | None = None) -> MatchResult:
        norm = _normalize(raw_name)
        if not norm:
            return MatchResult(None, 0.0, "empty name")

        # 1. Exact full-name match (highest confidence).
        exact = self.players[self.players["full_norm"] == norm]
        if country_abbr:
            exact = exact[exact["country_abbr"] == country_abbr]
        if len(exact) == 1:
            return MatchResult(int(exact.iloc[0]["player_id"]), 1.00, "exact full")

        # 2. Known-name match (e.g. "Vinicius Jr").
        known = self.players[self.players["known_norm"] == norm]
        if country_abbr:
            known = known[known["country_abbr"] == country_abbr]
        if len(known) == 1:
            return MatchResult(int(known.iloc[0]["player_id"]), 0.95, "exact known_name")

        # 3. Last-name match within (country, position).
        last = norm.split()[-1]
        if country_abbr and country_abbr in self._by_country.groups:
            by_country = self.players[self.players["country_abbr"] == country_abbr]
            if position:
                in_pos = by_country[by_country["position"] == position]
                last_pos = in_pos[in_pos["last_norm"] == last]
                if len(last_pos) == 1:
                    return MatchResult(int(last_pos.iloc[0]["player_id"]),
                                       0.85, f"last in {country_abbr}/{position}")
            last_country = by_country[by_country["last_norm"] == last]
            if len(last_country) == 1:
                return MatchResult(int(last_country.iloc[0]["player_id"]),
                                   0.70, f"last in {country_abbr}")

        # 4. Last-name match without country (low confidence).
        last_only = self.players[self.players["last_norm"] == last]
        if len(last_only) == 1:
            return MatchResult(int(last_only.iloc[0]["player_id"]),
                               0.50, "last only (no country)")

        # No match.
        return MatchResult(None, 0.0, "no match")
