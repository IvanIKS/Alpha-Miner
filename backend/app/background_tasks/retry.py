"""Shared retry policy for the Celery tasks.

A transient connection error -- an unstable internet connection, or the exchange / Anthropic
API briefly unreachable -- should pause the task and try again, not fail the whole mining run.
The provider SDKs already retry fast blips internally, so an error only reaches a task when the
network has been down long enough that those retries were exhausted; we therefore back off in
tens of seconds rather than milliseconds.
"""
from __future__ import annotations

from functools import lru_cache

# Wait before retry n (0-based): 30, 60, 120, 240, 480, then capped at RETRY_WAIT_MAX.
RETRY_WAIT_BASE = 30
RETRY_WAIT_MAX = 600


def retry_countdown(retries: int) -> int:
    """Exponential backoff (seconds) for the `retries`-th retry -- pass self.request.retries."""
    return min(RETRY_WAIT_BASE * (2 ** max(retries, 0)), RETRY_WAIT_MAX)


@lru_cache(maxsize=1)
def connection_errors() -> tuple[type[BaseException], ...]:
    """Exception types that mean "the network blinked" across the SDKs the tasks call: ccxt
    (exchange), anthropic / httpx / requests (LLM and HTTP), plus the builtin socket errors.
    Resolved lazily so a missing optional package never breaks the import."""
    errors: list[type[BaseException]] = [ConnectionError, TimeoutError]
    for module, names in (
        ("ccxt", ("NetworkError",)),
        ("anthropic", ("APIConnectionError",)),
        ("httpx", ("TransportError",)),
        ("requests.exceptions", ("ConnectionError", "Timeout")),
    ):
        try:
            mod = __import__(module, fromlist=list(names))
        except Exception:
            continue
        for name in names:
            exc = getattr(mod, name, None)
            if isinstance(exc, type) and issubclass(exc, BaseException):
                errors.append(exc)
    return tuple(errors)
