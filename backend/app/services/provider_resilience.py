"""Shared, process-local resilience controls for calls to the AI provider."""

from __future__ import annotations

import logging
import math
import random
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable

from groq import RateLimitError

from app.config import Settings
from app.services.exceptions import AICircuitOpenError, AITimeoutError
from app.services.processing_budget import ProcessingBudget

logger = logging.getLogger(__name__)

_QUOTA_CODES = {
    "billing_hard_limit_reached",
    "billing_not_active",
    "insufficient_quota",
    "quota_exceeded",
}
_TRANSIENT_RATE_CODES = {"rate_limit_exceeded", "rate_limit_reached"}


def _error_body(error: RateLimitError) -> dict:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        return body
    response = getattr(error, "response", None)
    try:
        parsed = response.json() if response is not None else None
    except Exception:
        parsed = None
    return parsed if isinstance(parsed, dict) else {}


def provider_error_code(error: RateLimitError) -> str | None:
    body = _error_body(error)
    detail = body.get("error", body)
    if not isinstance(detail, dict):
        return None
    code = detail.get("code") or detail.get("type")
    return code.strip().lower() if isinstance(code, str) else None


def _parse_retry_delay(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            matches = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", value.lower().replace(" ", ""))
            if not matches:
                return None
            units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
            return sum(float(amount) * units[unit] for amount, unit in matches)


def extract_retry_after_seconds(error: RateLimitError) -> float | None:
    """Read a provider wait hint without exposing provider response details."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) if response is not None else {}
    get_header = getattr(headers, "get", lambda _key: None)

    for name in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        delay = _parse_retry_delay(get_header(name))
        if delay is not None:
            return delay
    return None


def classify_rate_limit(error: RateLimitError) -> str:
    """Return ``temporary``, ``quota`` or ``unknown`` using structured provider data."""
    body = _error_body(error)
    detail = body.get("error", body)
    code = provider_error_code(error)
    if code in _QUOTA_CODES:
        return "quota"

    message = detail.get("message", "") if isinstance(detail, dict) else ""
    message = message.lower() if isinstance(message, str) else ""
    if any(marker in message for marker in ("insufficient quota", "quota exhausted", "billing limit", "billing exhausted")):
        return "quota"
    if code in _TRANSIENT_RATE_CODES:
        return "temporary"
    if extract_retry_after_seconds(error) is not None:
        return "temporary"
    return "unknown"


def exponential_backoff(
    retry_index: int,
    *,
    initial_seconds: float,
    maximum_seconds: float,
    multiplier: float,
    random_value: Callable[[], float] = random.random,
) -> float:
    """Calculate bounded exponential backoff with deterministic-testable +/-20% jitter."""
    base = min(maximum_seconds, initial_seconds * multiplier**retry_index)
    jitter = 0.8 + (max(0.0, min(1.0, random_value())) * 0.4)
    return min(maximum_seconds, base * jitter)


class SharedRequestPacer:
    """Reserve process-wide provider request starts with a minimum interval."""

    def __init__(
        self,
        interval_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval_seconds = interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_start = 0.0
        self._not_before = 0.0

    def defer_for(self, seconds: float) -> None:
        with self._lock:
            self._not_before = max(self._not_before, self._clock() + max(0.0, seconds))

    def wait_for_slot(self, budget: ProcessingBudget | None = None) -> float:
        while True:
            with self._lock:
                now = self._clock()
                target = max(self._next_start, self._not_before)
                delay = max(0.0, target - now)
                if delay <= 0:
                    self._next_start = now + self.interval_seconds
                    return now

            if budget is not None and budget.remaining_seconds() <= delay + 0.1:
                raise AITimeoutError()
            self._sleep(delay)
            if budget is not None:
                budget.check_deadline(min_remaining=0.1)


class RateLimitCircuitBreaker:
    """Thread-safe circuit breaker that opens only after exhausted transient 429 calls."""

    def __init__(
        self,
        failure_threshold: int,
        recovery_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._state = "closed"
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _transition(self, next_state: str) -> None:
        previous = self._state
        if previous != next_state:
            self._state = next_state
            logger.warning(
                "AI provider circuit state changed",
                extra={"circuit_from": previous, "circuit_to": next_state},
            )

    def begin_call(self) -> None:
        with self._lock:
            now = self._clock()
            if self._state == "open":
                opened_at = self._opened_at if self._opened_at is not None else now
                remaining = self.recovery_seconds - (now - opened_at)
                if remaining > 0:
                    raise AICircuitOpenError(retry_after=math.ceil(remaining))
                self._transition("half_open")
                self._probe_in_flight = True
                return
            if self._state == "half_open":
                if self._probe_in_flight:
                    raise AICircuitOpenError(retry_after=1)
                self._probe_in_flight = True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False
            self._transition("closed")

    def record_rate_limit_failure(self) -> None:
        with self._lock:
            if self._state == "half_open":
                self._failures = self.failure_threshold
            else:
                self._failures += 1
            self._probe_in_flight = False
            if self._failures >= self.failure_threshold:
                self._opened_at = self._clock()
                self._transition("open")

    def record_non_rate_limit_outcome(self) -> None:
        """Release a probe without counting permanent or unrelated failures."""
        with self._lock:
            if self._state == "half_open":
                self._probe_in_flight = False
                self._failures = 0
                self._opened_at = None
                self._transition("closed")


class ProviderRequestCoordinator:
    """Shared pacing and rate-limit circuit state for all Groq operations in a process."""

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.rate_limiter = SharedRequestPacer(
            settings.AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS,
            clock=clock,
            sleep=sleep,
        )
        self.circuit_breaker = RateLimitCircuitBreaker(
            settings.AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            settings.AI_CIRCUIT_BREAKER_RECOVERY_SECONDS,
            clock=clock,
        )
        self._clock = clock
        self._sleep = sleep
        self._random_value = random_value

    def begin_call(self, budget: ProcessingBudget | None) -> None:
        self.circuit_breaker.begin_call()
        self.rate_limiter.wait_for_slot(budget)

    def wait_for_retry(self, delay: float, budget: ProcessingBudget | None, min_call_margin: float = 2.0) -> bool:
        self.rate_limiter.defer_for(delay)
        if budget is not None and budget.remaining_seconds() < delay + min_call_margin:
            return False
        self.rate_limiter.wait_for_slot(budget)
        return True

    def backoff(self, retry_index: int, *, initial: float, maximum: float, multiplier: float) -> float:
        return exponential_backoff(
            retry_index,
            initial_seconds=initial,
            maximum_seconds=maximum,
            multiplier=multiplier,
            random_value=self._random_value,
        )

    def sleep(self, seconds: float) -> None:
        self._sleep(seconds)

    def succeeded(self) -> None:
        self.circuit_breaker.record_success()

    def rate_limited(self) -> None:
        self.circuit_breaker.record_rate_limit_failure()

    def other_outcome(self) -> None:
        self.circuit_breaker.record_non_rate_limit_outcome()
