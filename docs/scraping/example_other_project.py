"""Demonstration: reuse the `scraping` module in a non-football project.

Shows that `StealthClient` works independently of the FIFA fantasy
project. Fetches the Hacker News front page, prints the top story
titles.

Run from the repo root:
    .venv/bin/python docs/scraping/example_other_project.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add the project's src/ to sys.path so this script runs from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fifa_fantasy.external.scraping import StealthClient
from bs4 import BeautifulSoup


def main() -> int:
    client = StealthClient(
        impersonate="chrome124",
        rate_limit_per_second=1.0,
        cache_dir="/tmp/scraping_demo",
        cache_ttl_hours=1.0,
    )
    response = client.get("https://news.ycombinator.com/")
    if response.status_code != 200:
        print(f"unexpected status: {response.status_code}")
        return 1
    soup = BeautifulSoup(response.text, "lxml")
    titles = [a.get_text() for a in soup.select(".titleline > a")][:10]
    print(f"Top {len(titles)} HN stories ({'cached' if response.from_cache else 'fresh'}):")
    for i, t in enumerate(titles, 1):
        print(f"  {i:>2}. {t}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
