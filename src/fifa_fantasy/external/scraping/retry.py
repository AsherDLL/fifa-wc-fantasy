"""Exponential backoff with jitter for retryable HTTP errors."""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# Status codes worth retrying.
RETRY_STATUSES = (408, 429, 500, 502, 503, 504)


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


def default_retryable(value_or_exc) -> bool:
    """Default predicate: retry on common HTTP errors + network exceptions."""
    if isinstance(value_or_exc, Exception):
        name = type(value_or_exc).__name__.lower()
        return any(s in name for s in ("connectionerror", "timeout", "transport"))
    status = getattr(value_or_exc, "status_code", None)
    return status in RETRY_STATUSES
