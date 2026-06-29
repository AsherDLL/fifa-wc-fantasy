"""On-disk response cache with TTL.

Cache key = SHA256(method + URL + sorted relevant headers). Each entry
is stored as two files:

    <key>.body   raw response body bytes
    <key>.meta   JSON metadata: status_code, headers, stored_at_utc, url

Read path: load .meta, check stored_at_utc + ttl_hours; if expired, miss.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping


@dataclass
class CachedEntry:
    body: bytes
    status_code: int
    headers: dict
    url: str
    stored_at_utc: datetime
    from_cache: bool = True


# Headers that affect the response content; included in the cache key.
_VARYING_HEADERS = ("Accept", "Accept-Language", "Authorization")


class DiskCache:
    """Simple disk-backed cache. Not LRU; relies on TTL + manual cleanup."""

    def __init__(self, cache_dir: Path | str, ttl_hours: float = 6.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _key(self, method: str, url: str, headers: Mapping[str, str] | None) -> str:
        h = hashlib.sha256()
        h.update(method.upper().encode())
        h.update(b"\n")
        h.update(url.encode())
        h.update(b"\n")
        if headers:
            for name in _VARYING_HEADERS:
                v = headers.get(name) or headers.get(name.lower(), "")
                h.update(f"{name}={v}\n".encode())
        return h.hexdigest()

    def get(self, method: str, url: str,
            headers: Mapping[str, str] | None = None) -> CachedEntry | None:
        key = self._key(method, url, headers)
        meta_path = self.cache_dir / f"{key}.meta"
        body_path = self.cache_dir / f"{key}.body"
        if not meta_path.exists() or not body_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        stored_at = datetime.fromisoformat(meta["stored_at_utc"])
        if datetime.now(timezone.utc) - stored_at > self.ttl:
            return None
        try:
            body = body_path.read_bytes()
        except OSError:
            return None
        return CachedEntry(
            body=body,
            status_code=int(meta["status_code"]),
            headers=meta.get("headers", {}),
            url=meta["url"],
            stored_at_utc=stored_at,
            from_cache=True,
        )

    def put(self, method: str, url: str, headers: Mapping[str, str] | None,
            body: bytes, status_code: int, response_headers: Mapping[str, str]) -> None:
        key = self._key(method, url, headers)
        meta_path = self.cache_dir / f"{key}.meta"
        body_path = self.cache_dir / f"{key}.body"
        body_path.write_bytes(body)
        meta_path.write_text(json.dumps({
            "url": url,
            "status_code": status_code,
            "headers": dict(response_headers),
            "stored_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

    def clear(self) -> int:
        """Delete all cached entries; returns count deleted."""
        n = 0
        for p in self.cache_dir.glob("*.meta"):
            try:
                p.unlink()
                (self.cache_dir / f"{p.stem}.body").unlink(missing_ok=True)
                n += 1
            except OSError:
                continue
        return n
