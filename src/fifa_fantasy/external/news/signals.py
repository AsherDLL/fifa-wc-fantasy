"""Config-driven player-signal extraction from collected news articles.

Turns the article cache (data/external/news_articles/*.parquet) into a
structured per-player signal table: injury, fitness doubt, suspension,
rotation, return-from-injury, player-of-the-match, and whatever else the
vocabulary file defines. The vocabulary lives in `config/signals.json`
(override with NEWS_SIGNALS_CONFIG or --signals-config): the code has no
hardcoded words, so the extractor is reusable in any domain by swapping
the JSON and the entity index.

Matching rule: a signal fires when a pattern and a player-name mention
occur within `proximity_chars` of each other in the article title+body.
Proximity is the false-positive guard: "injury" in paragraph one and a
player listed in paragraph nine is noise; "Dembele limped off with a
hamstring injury" is signal. Short surnames (below min_lastname_len)
must match the player's full name to avoid substring hits like Gill in
Gillingham.

Output: data/external/player_signals/signals_<UTC>.parquet with one row
per (player, article, signal). Downstream policy: these rows are intel
for the analyst and the report. They deliberately do NOT mutate model
predictions or effective points; every signal that should become a
coefficient must first pass the leak-free validation gate like any
other feature (see docs sections 5c and 11h).

    python -m fifa_fantasy.external.news.signals --days 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("news.signals")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "signals.json"
DEFAULT_ARTICLES_DIR = Path("data/external/news_articles")
DEFAULT_OUT_DIR = Path("data/external/player_signals")
DEFAULT_RAW_DIR = Path("data/raw")


@dataclass(frozen=True)
class SignalsConfig:
    proximity_chars: int
    min_lastname_len: int
    # signal name -> (class, tuple of lowercase patterns)
    signals: dict[str, tuple[str, tuple[str, ...]]]


def load_signals_config(path: Path | str | None = None) -> SignalsConfig:
    p = Path(path or os.environ.get("NEWS_SIGNALS_CONFIG") or DEFAULT_CONFIG_PATH)
    raw = json.loads(p.read_text())
    signals = {
        name: (spec.get("class", "info"),
               tuple(s.lower() for s in spec.get("patterns", ())))
        for name, spec in raw.get("signals", {}).items()
    }
    return SignalsConfig(
        proximity_chars=int(raw.get("proximity_chars", 220)),
        min_lastname_len=int(raw.get("min_lastname_len", 5)),
        signals=signals,
    )


def _norm(text: str) -> str:
    """Lowercase, strip accents. Keeps offsets aligned (1 char per char)."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c if not unicodedata.combining(c) else " "
                       for c in decomposed)
    # NFKD can expand some chars; re-normalize length by replacing runs is
    # overkill here because we only need approximate offsets for proximity.
    return stripped.lower()


def build_name_index(players: pd.DataFrame,
                     min_lastname_len: int) -> list[dict]:
    """One entry per player with the normalized name variants to search.

    Variants: full name always; known_name and last name only when long
    enough to be unambiguous as standalone words.
    """
    index = []
    for _, p in players.iterrows():
        variants = set()
        full = _norm(str(p["full_name"]))
        variants.add(full)
        known = p.get("known_name")
        if pd.notna(known) and len(str(known)) >= min_lastname_len:
            variants.add(_norm(str(known)))
        last = p.get("last_name")
        if pd.notna(last) and len(str(last)) >= min_lastname_len:
            variants.add(_norm(str(last)))
        index.append({
            "player_id": int(p["player_id"]),
            "full_name": str(p["full_name"]),
            "country_abbr": str(p.get("country_abbr", "")),
            "position": str(p.get("position", "")),
            "patterns": [re.compile(r"\b" + re.escape(v) + r"\b")
                         for v in variants if v],
        })
    return index


def _load_recent_articles(articles_dir: Path, days: float) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    frames = []
    for f in sorted(articles_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if "collected_at_utc" in df.columns:
            ts = pd.to_datetime(df["collected_at_utc"], utc=True, errors="coerce")
            df = df[ts >= cutoff]
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "url" in out.columns:
        out = out.drop_duplicates("url", keep="last")
    return out


def extract_signals(articles: pd.DataFrame,
                    players: pd.DataFrame,
                    cfg: SignalsConfig) -> pd.DataFrame:
    """One output row per (player, article, signal) proximity hit."""
    index = build_name_index(players, cfg.min_lastname_len)
    rows = []
    for _, art in articles.iterrows():
        text = " ".join(str(art.get(c) or "") for c in
                        ("title", "snippet", "body_text"))
        if len(text) < 20:
            continue
        norm = _norm(text)
        # Find signal-pattern hit positions once per article.
        sig_hits: dict[str, list[int]] = {}
        for name, (cls, patterns) in cfg.signals.items():
            positions = [m.start() for pat in patterns
                         for m in re.finditer(re.escape(pat), norm)]
            if positions:
                sig_hits[name] = positions
        if not sig_hits:
            continue
        for entry in index:
            name_positions = [m.start() for pat in entry["patterns"]
                              for m in pat.finditer(norm)]
            if not name_positions:
                continue
            for sig_name, positions in sig_hits.items():
                near = min((abs(sp - np_) for sp in positions
                            for np_ in name_positions), default=None)
                if near is None or near > cfg.proximity_chars:
                    continue
                cls = cfg.signals[sig_name][0]
                # Evidence window around the closest pairing.
                sp = min(positions,
                         key=lambda s: min(abs(s - n) for n in name_positions))
                lo = max(0, sp - 90)
                evidence = text[lo:sp + 130].strip().replace("\n", " ")
                rows.append({
                    "player_id": entry["player_id"],
                    "full_name": entry["full_name"],
                    "country_abbr": entry["country_abbr"],
                    "position": entry["position"],
                    "signal": sig_name,
                    "signal_class": cls,
                    "proximity_chars": int(near),
                    "evidence": evidence[:220],
                    "article_title": str(art.get("title", ""))[:200],
                    "source_id": art.get("source_id"),
                    "source_confidence": art.get("source_confidence"),
                    "published_at_utc": art.get("published_at_utc"),
                    "collected_at_utc": art.get("collected_at_utc"),
                    "url": art.get("url"),
                })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["signal_class", "signal", "full_name"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(prog="fifa_fantasy.external.news.signals")
    p.add_argument("--days", type=float, default=3.0,
                   help="Only scan articles collected in the last N days")
    p.add_argument("--articles-dir", type=Path, default=DEFAULT_ARTICLES_DIR)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help="Where players_*.parquet lives (the entity index)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--signals-config", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_signals_config(args.signals_config)
    players_files = sorted(args.raw_dir.glob("players_*.parquet"))
    if not players_files:
        log.error("no players_*.parquet under %s", args.raw_dir)
        return 1
    players = pd.read_parquet(players_files[-1])
    articles = _load_recent_articles(args.articles_dir, args.days)
    log.info("scanning %d articles against %d players, %d signal types",
             len(articles), len(players), len(cfg.signals))
    out = extract_signals(articles, players, cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = args.out_dir / f"signals_{ts}.parquet"
    out.to_parquet(path, index=False)
    log.info("%d signal rows -> %s", len(out), path)
    if len(out):
        summary = (out.groupby(["signal_class", "signal"])["player_id"]
                   .nunique().rename("players"))
        for (cls, sig), n in summary.items():
            log.info("  %-6s %-22s %d player(s)", cls, sig, n)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
