"""Per-host token-bucket rate limiter (thread-safe, sync).

Each host gets its own bucket. When a request would exceed the rate,
the caller sleeps until a slot is available. We track the monotonic
clock so adjustment to wall-clock time does not break it.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class _Bucket:
    """Token-bucket state for one host. Holds the last-permit timestamp
    and a re-entrant lock so concurrent threads serialise correctly."""
    last_permit_monotonic: float = 0.0
    lock: threading.Lock = None  # set in __post_init__

    def __post_init__(self):
        self.lock = threading.Lock()


class PerHostRateLimiter:
    """Wait until the host's bucket has a token, then take one.

    Args:
        requests_per_second: float (e.g. 1.0 = 1 req/s, 0.5 = 1 req/2s).
            Defaults to 1.0 which is conservative for ethical scraping.
    """

    def __init__(self, requests_per_second: float = 1.0):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.min_interval_s = 1.0 / requests_per_second
        self._buckets: dict[str, _Bucket] = defaultdict(_Bucket)
        self._global_lock = threading.Lock()

    def acquire(self, url: str) -> float:
        """Block until a token is available; return seconds slept."""
        host = urlparse(url).netloc or url
        with self._global_lock:
            bucket = self._buckets[host]
        with bucket.lock:
            now = time.monotonic()
            wait = (bucket.last_permit_monotonic + self.min_interval_s) - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            bucket.last_permit_monotonic = now
            return max(0.0, wait)
