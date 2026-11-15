"""Deterministic coverage for centralized provider pacing, retries and circuit breaking."""

import threading

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
from unittest.mock import MagicMock

from app.config import Settings
from app.main import create_app
from app.services.exceptions import (
    AIBudgetExceededError,
    AICircuitOpenError,
    AIAuthenticationError,
    AIProviderError,
    AIRateLimitError,
    AIQuotaExceededError,
)
from app.services.generation_service import GenerationService
from app.services.processing_budget import ProcessingBudget
from app.services.groq_service import GroqService, get_groq_service
from app.services.provider_resilience import (
    ProviderRequestCoordinator,
    RateLimitCircuitBreaker,
    SharedRequestPacer,
    classify_rate_limit,
    exponential_backoff,
)


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.lock = threading.Lock()

    def now(self):
        with self.lock:
            return self.value

    def sleep(self, seconds):
        with self.lock:
            self.value += seconds


def rate_limit_error(*, code="rate_limit_exceeded", headers=None):
    request = httpx.Request("POST", "https://api.groq.com")
    body = {"error": {"type": code, "code": code}}
    response = httpx.Response(
        status_code=429,
        headers=headers or {},
        json=body,
        request=request,
    )
    return RateLimitError("rate limit", response=response, body=body)


def completion(text='{"ok": true}'):
    result = MagicMock()
    result.choices[0].message.content = text
    return result


def setup_service(side_effect, *, settings=None, clock=None, random_value=lambda: 0.5):
    settings = settings or Settings(
        GROQ_API_KEY="test-key",
        AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS=0,
        AI_PROVIDER_RATE_LIMIT_MAX_RETRIES=1,
        AI_PROVIDER_TRANSIENT_MAX_RETRIES=1,
        AI_PROVIDER_BACKOFF_INITIAL_SECONDS=1,
        AI_PROVIDER_BACKOFF_MAX_SECONDS=16,
        AI_PROVIDER_TRANSIENT_BACKOFF_INITIAL_SECONDS=0.5,
        AI_PROVIDER_TRANSIENT_BACKOFF_MAX_SECONDS=2,
        AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5,
        AI_CIRCUIT_BREAKER_RECOVERY_SECONDS=5,
    )
    clock = clock or FakeClock()
    coordinator = ProviderRequestCoordinator(
        settings,
        clock=clock.now,
        sleep=clock.sleep,
        random_value=random_value,
    )
    groq = MagicMock()
    groq.model = "test-model"
    groq.client.chat.completions.create.side_effect = side_effect
    service = GenerationService(groq, settings=settings, coordinator=coordinator)
    budget = ProcessingBudget(max_calls=120, max_duration_seconds=300, max_operations=100, clock=clock.now)
    return service, groq, coordinator, budget, clock


def test_retries_transient_429_then_succeeds_and_counts_every_attempt():
    service, groq, _coordinator, budget, clock = setup_service(
        [rate_limit_error(), completion("valid")]
    )
    assert service._call_groq([], budget) == "valid"
    assert groq.client.chat.completions.create.call_count == 2
    assert budget.calls_consumed == 2
    assert clock.now() == pytest.approx(1.0)


def test_retry_after_is_honored_instead_of_replaced_by_fallback_backoff():
    service, groq, _coordinator, budget, clock = setup_service(
        [rate_limit_error(headers={"Retry-After": "3"}), completion("valid")]
    )
    assert service._call_groq([], budget) == "valid"
    assert clock.now() == pytest.approx(3.0)
    assert groq.client.chat.completions.create.call_count == 2


def test_reset_header_duration_is_used_as_provider_wait_hint():
    error = rate_limit_error(headers={"x-ratelimit-reset-requests": "2s"})
    assert classify_rate_limit(error) == "temporary"
    service, _groq, _coordinator, _budget, clock = setup_service(
        [error, completion("valid")]
    )
    assert service._call_groq([]) == "valid"
    assert clock.now() == pytest.approx(2.0)


def test_fallback_exponential_backoff_is_bounded_and_jittered():
    delays = [
        exponential_backoff(i, initial_seconds=1, maximum_seconds=16, multiplier=2, random_value=lambda: 0.5)
        for i in range(5)
    ]
    assert delays == [1, 2, 4, 8, 16]
    assert exponential_backoff(0, initial_seconds=1, maximum_seconds=16, multiplier=2, random_value=lambda: 0) == pytest.approx(0.8)
    assert exponential_backoff(0, initial_seconds=1, maximum_seconds=16, multiplier=2, random_value=lambda: 1) == pytest.approx(1.2)
    assert exponential_backoff(6, initial_seconds=1, maximum_seconds=16, multiplier=2, random_value=lambda: 1) == 16


def test_rate_limit_retries_stop_at_configured_maximum():
    settings = Settings(
        GROQ_API_KEY="test-key",
        AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS=0,
        AI_PROVIDER_RATE_LIMIT_MAX_RETRIES=2,
        AI_PROVIDER_BACKOFF_INITIAL_SECONDS=1,
        AI_PROVIDER_BACKOFF_MAX_SECONDS=16,
        AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5,
    )
    service, groq, _coordinator, budget, _clock = setup_service(
        rate_limit_error(), settings=settings
    )
    with pytest.raises(AIRateLimitError):
        service._call_groq([], budget)
    assert groq.client.chat.completions.create.call_count == 3
    assert budget.calls_consumed == 3


def test_transient_500_retries_once_then_succeeds():
    req = httpx.Request("POST", "https://api.groq.com")
    error = APIStatusError("upstream", response=httpx.Response(500, request=req), body=None)
    service, groq, _coordinator, budget, clock = setup_service(
        [error, completion("valid")]
    )
    assert service._call_groq([], budget) == "valid"
    assert groq.client.chat.completions.create.call_count == 2
    assert budget.calls_consumed == 2
    assert clock.now() == pytest.approx(0.5)


def test_network_failure_uses_separate_bounded_retry_policy():
    request = httpx.Request("POST", "https://api.groq.com")
    service, groq, _coordinator, budget, clock = setup_service(
        [APIConnectionError(request=request), completion("valid")]
    )
    assert service._call_groq([], budget) == "valid"
    assert groq.client.chat.completions.create.call_count == 2
    assert budget.calls_consumed == 2
    assert clock.now() == pytest.approx(0.5)


def test_transient_network_retries_stop_at_configured_maximum():
    settings = Settings(
        GROQ_API_KEY="test-key",
        AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS=0,
        AI_PROVIDER_TRANSIENT_MAX_RETRIES=2,
        AI_PROVIDER_TRANSIENT_BACKOFF_INITIAL_SECONDS=0.5,
        AI_PROVIDER_TRANSIENT_BACKOFF_MAX_SECONDS=2,
    )
    request = httpx.Request("POST", "https://api.groq.com")
    service, groq, _coordinator, budget, clock = setup_service(
        APIConnectionError(request=request), settings=settings
    )
    with pytest.raises(AIProviderError):
        service._call_groq([], budget)
    assert groq.client.chat.completions.create.call_count == 3
    assert budget.calls_consumed == 3
    assert clock.now() == pytest.approx(1.5)


def test_permanent_400_is_not_retried():
    req = httpx.Request("POST", "https://api.groq.com")
    error = APIStatusError("invalid request", response=httpx.Response(400, request=req), body=None)
    service, groq, *_ = setup_service(error)
    with pytest.raises(AIProviderError):
        service._call_groq([])
    assert groq.client.chat.completions.create.call_count == 1


def test_invalid_api_key_is_not_retried():
    req = httpx.Request("POST", "https://api.groq.com")
    error = AuthenticationError("invalid key", response=httpx.Response(401, request=req), body=None)
    service, groq, *_ = setup_service(error)
    with pytest.raises(AIAuthenticationError):
        service._call_groq([])
    assert groq.client.chat.completions.create.call_count == 1


def test_quota_exhaustion_is_distinguished_and_not_retried():
    service, groq, _coordinator, _budget, _clock = setup_service(
        rate_limit_error(code="insufficient_quota")
    )
    with pytest.raises(AIQuotaExceededError):
        service._call_groq([])
    assert groq.client.chat.completions.create.call_count == 1


def test_unclassified_429_without_retry_hint_is_not_retried():
    error = rate_limit_error(code="provider_rejected")
    service, groq, _coordinator, _budget, _clock = setup_service(error)
    with pytest.raises(AIRateLimitError):
        service._call_groq([])
    assert groq.client.chat.completions.create.call_count == 1


def test_circuit_opens_after_threshold_and_rejects_without_provider_call():
    settings = Settings(
        GROQ_API_KEY="test-key",
        AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS=0,
        AI_PROVIDER_RATE_LIMIT_MAX_RETRIES=0,
        AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD=2,
        AI_CIRCUIT_BREAKER_RECOVERY_SECONDS=5,
    )
    error = rate_limit_error()
    service, groq, coordinator, _budget, _clock = setup_service([error, error], settings=settings)
    for _ in range(2):
        with pytest.raises(AIRateLimitError):
            service._call_groq([])
    assert coordinator.circuit_breaker.state == "open"
    with pytest.raises(AICircuitOpenError) as exc_info:
        service._call_groq([])
    assert exc_info.value.retry_after == 5
    assert groq.client.chat.completions.create.call_count == 2


def test_half_open_allows_one_probe_and_success_closes_circuit():
    settings = Settings(
        GROQ_API_KEY="test-key",
        AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS=0,
        AI_PROVIDER_RATE_LIMIT_MAX_RETRIES=0,
        AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD=1,
        AI_CIRCUIT_BREAKER_RECOVERY_SECONDS=5,
    )
    clock = FakeClock()
    service, _groq, coordinator, _budget, _clock = setup_service(
        [rate_limit_error(), completion("probe-success")], settings=settings, clock=clock
    )
    with pytest.raises(AIRateLimitError):
        service._call_groq([])
    clock.sleep(5)
    assert service._call_groq([]) == "probe-success"
    assert coordinator.circuit_breaker.state == "closed"


def test_half_open_rejects_concurrent_probe_until_active_probe_finishes():
    clock = FakeClock()
    breaker = RateLimitCircuitBreaker(1, 5, clock=clock.now)
    breaker.begin_call()
    breaker.record_rate_limit_failure()
    clock.sleep(5)
    breaker.begin_call()
    with pytest.raises(AICircuitOpenError):
        breaker.begin_call()
    breaker.record_success()
    assert breaker.state == "closed"


def test_open_circuit_returns_controlled_503_and_retry_after_header():
    settings = Settings(
        GROQ_API_KEY="test-key",
        AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS=0,
        AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD=1,
        AI_CIRCUIT_BREAKER_RECOVERY_SECONDS=10,
    )
    clock = FakeClock()
    coordinator = ProviderRequestCoordinator(settings, clock=clock.now, sleep=clock.sleep)
    coordinator.circuit_breaker._state = "open"
    coordinator.circuit_breaker._opened_at = clock.now()
    groq = MagicMock(spec=GroqService)
    groq.settings = settings
    groq.coordinator = coordinator
    app = create_app(settings)
    app.dependency_overrides[get_groq_service] = lambda: groq

    with TestClient(app) as client:
        response = client.post("/api/generate/quiz", json={"text": "study notes", "count": 1})

    app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "10"
    assert response.json() == {"detail": "AI service is temporarily unavailable. Please try again shortly."}
    groq.client.chat.completions.create.assert_not_called()


def test_shared_pacer_serializes_concurrent_request_starts():
    clock = FakeClock()
    pacer = SharedRequestPacer(1.0, clock=clock.now, sleep=clock.sleep)
    starts = []
    start_lock = threading.Lock()

    def take_slot():
        start_time = pacer.wait_for_slot()
        with start_lock:
            starts.append(start_time)

    threads = [threading.Thread(target=take_slot) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert all(not thread.is_alive() for thread in threads)
    ordered = sorted(starts)
    assert len(ordered) == 5
    assert all(right - left >= 1 for left, right in zip(ordered, ordered[1:]))


def test_retry_attempt_cannot_exceed_request_call_budget():
    service, groq, _coordinator, budget, _clock = setup_service(
        [rate_limit_error(), completion("should-not-be-called")]
    )
    budget.max_calls = 1
    with pytest.raises(AIBudgetExceededError):
        service._call_groq([], budget)
    assert budget.calls_consumed == 1
    assert groq.client.chat.completions.create.call_count == 1


def test_retry_after_longer_than_remaining_budget_fails_without_sleeping():
    service, groq, _coordinator, budget, clock = setup_service(
        rate_limit_error(headers={"retry-after": "25"})
    )
    budget.max_duration_seconds = 10
    with pytest.raises(AIRateLimitError):
        service._call_groq([], budget)
    assert groq.client.chat.completions.create.call_count == 1
    assert clock.now() == 0
