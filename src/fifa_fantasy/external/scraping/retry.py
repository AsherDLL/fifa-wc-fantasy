"""Exponential backoff with jitter, plus body-fingerprint block detection.

A 200 OK is not always a real response. Modern WAFs (Cloudflare, AWS
WAF, DataDome, Akamai, PerimeterX) often return 200 with a challenge
page in the body: "Just a moment...", "Checking your browser", etc.
We sniff for those signatures and treat them as retryable blocks so
the client can rotate identity / IP and try again.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping
from typing import TypeVar

T = TypeVar("T")

# Status codes worth retrying.
RETRY_STATUSES = (408, 429, 500, 502, 503, 504)

# Body substrings that indicate a WAF / bot-mgmt page rather than real
# content. All matched case-insensitively. Sources:
# - Cloudflare interstitial: well-known "Just a moment..." title and
#   challenge-platform JS.
# - AWS WAF: "AwsWafIntegration" appears in the inline JS challenge;
#   the page is served with status 202 plus x-amzn-waf-action: challenge.
# - DataDome: "datadome" in inline JS, plus X-Dd-B / X-DataDome-CID
#   response headers.
# - Akamai Bot Manager: "ak_bmsc" cookie + "Reference&#32;#" pattern.
# - PerimeterX / Human: "_pxhd" cookie, "PerimeterX" inline.
_BLOCK_BODY_NEEDLES: tuple[tuple[str, str], ...] = tuple(
    (waf, needle.lower()) for waf, needle in (
        ("cloudflare", "<title>just a moment"),
        ("cloudflare", "challenge-platform/h/"),
        ("cloudflare", "cf-mitigated"),
        ("aws_waf", "awswafintegration"),
        ("aws_waf", "/aws-waf-token"),
        ("datadome", "datadome.co/captcha"),
        ("datadome", "geo.captcha-delivery.com"),
        ("akamai", "ak_bmsc"),
        ("akamai", "_abck"),
        ("perimeterx", "perimeterx"),
        ("perimeterx", "_pxhd"),
        ("imperva", "incapsula incident id"),
    )
)

# Header tell-tales for the same WAFs.
_BLOCK_HEADER_HINTS: tuple[tuple[str, str, str], ...] = (
    ("aws_waf", "x-amzn-waf-action", "challenge"),
    ("aws_waf", "x-amzn-waf-action", "captcha"),
    ("cloudflare", "cf-mitigated", "challenge"),
    ("datadome", "x-dd-b", ""),
    ("datadome", "x-datadome-cid", ""),
)


def detect_block(status_code: int, headers: Mapping[str, str],
                 body: str) -> str | None:
    """Return the name of the WAF blocking us, or None if response looks real.

    Examines headers first (cheap, often definitive) then a bounded
    substring scan of the body (we only look at the first 16KB; WAF
    challenge pages are always small).
    """
    lower_headers = {k.lower(): (v or "").lower() for k, v in (headers or {}).items()}
    for waf, header, needle in _BLOCK_HEADER_HINTS:
        v = lower_headers.get(header)
        if v is None:
            continue
        if needle == "" or needle in v:
            return waf

    # AWS WAF specifically uses status 202 with a JS body. Generic 202
    # is rare from content sites, so this is a strong signal.
    if status_code == 202:
        return "aws_waf"

    if not body:
        return None
    head = body[:16384].lower()
    for waf, needle in _BLOCK_BODY_NEEDLES:
        if needle in head:
            return waf
    return None


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    jitter: float = 0.25,
    is_retryable: Callable[[Exception | T], bool] | None = None,
) -> T:
    """Call `fn()` up to `max_attempts` times with exponential backoff.

    Args:
        fn: callable returning the desired value
        max_attempts: total attempts (1 = no retry)
        base_delay_s: first sleep after a failure
        max_delay_s: ceiling for the sleep
        jitter: +/- fraction added to each sleep (0.25 = +/-25%)
        is_retryable: predicate; given the result or exception, return True
            to retry. Defaults to retrying on common HTTP status codes
            and any RequestException-class exception.

    Returns:
        The first successful return value.

    Raises:
        Whatever the last attempt raised.
    """
    last_exception: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001
            last_exception = e
            if attempt == max_attempts:
                raise
            if is_retryable is not None and not is_retryable(e):
                raise
            _sleep_with_jitter(attempt, base_delay_s, max_delay_s, jitter)
            continue
        # Success path with optional result-check predicate.
        if is_retryable is not None and is_retryable(result):
            if attempt == max_attempts:
                return result
            _sleep_with_jitter(attempt, base_delay_s, max_delay_s, jitter)
            continue
        return result
    if last_exception:
        raise last_exception
    return result  # unreachable; for the type checker


def _sleep_with_jitter(attempt: int, base: float, ceiling: float,
                      jitter: float) -> None:
    delay = min(ceiling, base * (2 ** (attempt - 1)))
    perturbed = delay * (1.0 + random.uniform(-jitter, jitter))
    time.sleep(max(0.0, perturbed))
