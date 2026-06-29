"""Optional proxy support. Reads from env var or explicit config."""
from __future__ import annotations

import os
from dataclasses import dataclass
from random import choice
from typing import Sequence


@dataclass(frozen=True)
class ProxyConfig:
    """One proxy endpoint."""
    url: str          # e.g. "http://user:pass@proxy.example.com:8000"
    label: str = ""   # human-readable name for logging


class ProxyRotator:
    """Round-robin or random proxy selection.

    Empty pool returns None (no proxy applied). The StealthClient passes
    whatever we return into curl_cffi's `proxies` parameter.
    """

    def __init__(self, proxies: Sequence[ProxyConfig] | None = None,
                 mode: str = "random"):
        self.proxies = list(proxies or [])
        if mode not in ("random", "round_robin"):
            raise ValueError("mode must be 'random' or 'round_robin'")
        self.mode = mode
        self._cursor = 0

    @classmethod
    def from_env(cls) -> "ProxyRotator":
        """Build from SCRAPING_PROXY_URL env var (single proxy).

        For multi-proxy setups, pass an explicit list to __init__.
        """
        url = os.environ.get("SCRAPING_PROXY_URL")
        if not url:
            return cls(proxies=[])
        return cls(proxies=[ProxyConfig(url=url, label="env")])

    def pick(self) -> ProxyConfig | None:
        if not self.proxies:
            return None
        if self.mode == "random":
            return choice(self.proxies)
        # round_robin
        p = self.proxies[self._cursor % len(self.proxies)]
        self._cursor += 1
        return p
