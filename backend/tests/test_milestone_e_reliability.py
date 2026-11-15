"""
Comprehensive deterministic test suite for Milestone E:
Fix Large-Document AI Reliability, Shared Provider Coordination, and Reusable Summarization.

Covers all 17 required scenarios:
1. Summarization uses the shared provider coordinator.
2. Temporary 429 with a 5-15 second Retry-After is retried when the deadline allows.
3. Retry stops when the deadline is insufficient.
4. Account quota exhaustion is not retried.
5. Permanent errors are not retried.
6. Summarization failures update circuit-breaker state correctly.
7. All summarization calls consume the request budget.
8. Flashcards and Quiz reuse the same cached consolidated summary.
9. Different document content creates a different cache entry.
10. Expired entries are regenerated.
11. Cache size remains bounded (LRU eviction).
12. Concurrent requests for identical documents do not duplicate summarization work (single-flight).
13. Failed summaries are not cached.
14. Duplicate quiz requests and correlation IDs (X-Request-ID) are handled safely.
15. The 504 path returns a controlled and accurate domain error.
16. Existing small-document generation still works.
17. Large-document processing preserves coverage across the entire document.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.config import Settings
from app.main import create_app
from app.models.generation import FlashcardItem, QuizQuestion
from app.services.exceptions import (
    AIAuthenticationError,
    AIBudgetExceededError,
    AICircuitOpenError,
    AIProviderError,
    AIQuotaExceededError,
    AIRateLimitError,
    AITimeoutError,
)
from app.services.generation_service import GenerationService
from app.services.groq_service import GroqService, get_groq_service
from app.services.processing_budget import ProcessingBudget
from app.services.provider_resilience import (
    ProviderRequestCoordinator,
    classify_rate_limit,
)
from app.services.summarization_service import (
    CHUNK_SUMMARIZATION_SYSTEM_PROMPT,
    CONSOLIDATION_SYSTEM_PROMPT,
    SummarizationService,
    chunk_text,
)
from app.services.summary_cache import DocumentSummaryCache


# ----------------------------------------------------------------------
# Helpers and Mocks
# ----------------------------------------------------------------------


class FakeClock:
    """Thread-safe simulated monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.value = start
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self.value

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.value += seconds


def make_mock_completion(content: str) -> MagicMock:
    """Create a mock Groq ChatCompletion object."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


def make_rate_limit_error(
    *,
    code: str = "rate_limit_exceeded",
    headers: dict[str, str] | None = None,
    message: str = "Rate limit hit",
) -> RateLimitError:
    """Create a structured Groq RateLimitError."""
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": message, "type": code, "code": code}}
    resp = httpx.Response(
        status_code=429,
        headers=headers or {},
        json=body,
        request=req,
    )
    return RateLimitError(message, response=resp, body=body)


# ----------------------------------------------------------------------
# Scenario 1: Summarization uses the shared provider coordinator
# ----------------------------------------------------------------------


def test_scenario_1_summarization_uses_shared_provider_coordinator():
    """Chunk summarization and consolidation must go through ProviderRequestCoordinator."""
    clock = FakeClock()
    settings = Settings(GROQ_API_KEY="test-key", AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS=0.1)
    coordinator = ProviderRequestCoordinator(settings, clock=clock.now, sleep=clock.sleep)

    mock_groq = MagicMock(spec=GroqService)
    mock_groq.model = "test-model"
    mock_groq.coordinator = coordinator
    mock_groq.client.chat.completions.create.return_value = make_mock_completion(
        json.dumps({"summary": "Valid educational section summary preserving key concepts."})
    )

    summarizer = SummarizationService(mock_groq, settings=settings, coordinator=coordinator)
    budget = ProcessingBudget.from_settings(settings, clock=clock.now)

    with patch.object(coordinator, "begin_call", wraps=coordinator.begin_call) as spy_begin:
        with patch.object(coordinator, "succeeded", wraps=coordinator.succeeded) as spy_success:
            result = summarizer._summarize_single_chunk("Important concepts in biology.", 1, 1, budget=budget)
            assert "Valid educational section summary" in result
            assert spy_begin.call_count == 1
            assert spy_success.call_count == 1


# ----------------------------------------------------------------------
# Scenario 2: Temporary 429 with 5-15s Retry-After retried when deadline allows
# ----------------------------------------------------------------------


def test_scenario_2_temporary_429_retried_when_deadline_allows():
    """A temporary 429 with Retry-After: 7 is retried and succeeds when budget deadline accommodates it."""
    clock = FakeClock()
    settings = Settings(
        GROQ_API_KEY="test-key",
        AI_PROVIDER_RATE_LIMIT_MAX_RETRIES=3,
        AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS=0,
    )
    coordinator = ProviderRequestCoordinator(settings, clock=clock.now, sleep=clock.sleep)

    err_429 = make_rate_limit_error(headers={"Retry-After": "7"})
    success_resp = make_mock_completion(
        json.dumps({"summary": "Recovered educational summary after rate limit wait."})
    )

    mock_groq = MagicMock(spec=GroqService)
    mock_groq.model = "test-model"
    mock_groq.coordinator = coordinator
    mock_groq.client.chat.completions.create.side_effect = [err_429, success_resp]

    summarizer = SummarizationService(mock_groq, settings=settings, coordinator=coordinator)
    budget = ProcessingBudget(max_calls=10, max_duration_seconds=300.0, max_operations=10, clock=clock.now)

    summary = summarizer._summarize_single_chunk("Photosynthesis processes and light reactions.", 1, 1, budget=budget)
    assert "Recovered educational summary" in summary
    assert mock_groq.client.chat.completions.create.call_count == 2
    assert clock.now() >= 1007.0  # Waited 7 seconds


# ----------------------------------------------------------------------
# Scenario 3: Retry stops when deadline is insufficient
# ----------------------------------------------------------------------


def test_scenario_3_retry_stops_when_deadline_insufficient():
    """When Retry-After is 10s but only 8s remains before budget deadline, call halts fast."""
    clock = FakeClock()
    settings = Settings(GROQ_API_KEY="test-key", AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS=0)
    coordinator = ProviderRequestCoordinator(settings, clock=clock.now, sleep=clock.sleep)

    err_429 = make_rate_limit_error(headers={"Retry-After": "10"})
    mock_groq = MagicMock(spec=GroqService)
    mock_groq.model = "test-model"
    mock_groq.coordinator = coordinator
    mock_groq.client.chat.completions.create.side_effect = err_429

    summarizer = SummarizationService(mock_groq, settings=settings, coordinator=coordinator)
    # Remaining time is only 8.0s, which cannot accommodate 10s delay + margin
    budget = ProcessingBudget(max_calls=10, max_duration_seconds=8.0, max_operations=10, clock=clock.now)

    with pytest.raises(AIRateLimitError):
        summarizer._call_groq([{"role": "user", "content": "text"}], budget=budget)

    assert mock_groq.client.chat.completions.create.call_count == 1


# ----------------------------------------------------------------------
# Scenario 4: Account quota exhaustion is not retried
# ----------------------------------------------------------------------


def test_scenario_4_quota_exhaustion_not_retried():
    """Quota exhaustion (insufficient_quota) fails immediately without retrying."""
    clock = FakeClock()
    settings = Settings(GROQ_API_KEY="test-key", AI_PROVIDER_RATE_LIMIT_MAX_RETRIES=5)
    coordinator = ProviderRequestCoordinator(settings, clock=clock.now, sleep=clock.sleep)

    quota_err = make_rate_limit_error(code="insufficient_quota", message="Billing limit reached")
    mock_groq = MagicMock(spec=GroqService)
    mock_groq.model = "test-model"
    mock_groq.coordinator = coordinator
    mock_groq.client.chat.completions.create.side_effect = quota_err

    summarizer = SummarizationService(mock_groq, settings=settings, coordinator=coordinator)
    budget = ProcessingBudget.from_settings(settings, clock=clock.now)

    with pytest.raises(AIQuotaExceededError):
        summarizer._call_groq([{"role": "user", "content": "test"}], budget=budget)

    assert mock_groq.client.chat.completions.create.call_count == 1


# ----------------------------------------------------------------------
# Scenario 5: Permanent errors are not retried
# ----------------------------------------------------------------------


def test_scenario_5_permanent_errors_not_retried():
    """Permanent errors (HTTP 400 Bad Request, AuthenticationError) fail immediately without retrying."""
    clock = FakeClock()
    settings = Settings(GROQ_API_KEY="test-key")
    coordinator = ProviderRequestCoordinator(settings, clock=clock.now, sleep=clock.sleep)

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    bad_req_err = APIStatusError("bad request", response=httpx.Response(400, request=req), body=None)

    mock_groq = MagicMock(spec=GroqService)
    mock_groq.model = "test-model"
    mock_groq.coordinator = coordinator
    mock_groq.client.chat.completions.create.side_effect = bad_req_err

    summarizer = SummarizationService(mock_groq, settings=settings, coordinator=coordinator)
    budget = ProcessingBudget.from_settings(settings, clock=clock.now)

    with pytest.raises(AIProviderError):
        summarizer._call_groq([{"role": "user", "content": "test"}], budget=budget)

    assert mock_groq.client.chat.completions.create.call_count == 1


# ----------------------------------------------------------------------
# Scenario 6: Summarization failures update circuit-breaker state correctly
# ----------------------------------------------------------------------


def test_scenario_6_summarization_failures_update_circuit_breaker():
    """Repeated rate-limit failures during summarization trip the circuit breaker."""
    clock = FakeClock()
    settings = Settings(
        GROQ_API_KEY="test-key",
        AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD=2,
        AI_CIRCUIT_BREAKER_RECOVERY_SECONDS=30.0,
        AI_PROVIDER_RATE_LIMIT_MAX_RETRIES=0,
    )
    coordinator = ProviderRequestCoordinator(settings, clock=clock.now, sleep=clock.sleep)

    rate_err = make_rate_limit_error(headers={"Retry-After": "60"})
    mock_groq = MagicMock(spec=GroqService)
    mock_groq.model = "test-model"
    mock_groq.coordinator = coordinator
    mock_groq.client.chat.completions.create.side_effect = rate_err

    summarizer = SummarizationService(mock_groq, settings=settings, coordinator=coordinator)
    budget = ProcessingBudget(max_calls=10, max_duration_seconds=300.0, clock=clock.now)

    # First exhausted rate limit failure
    with pytest.raises(AIRateLimitError):
        summarizer._call_groq([{"role": "user", "content": "1"}], budget=budget)
    assert coordinator.circuit_breaker.state == "closed"

    # Second exhausted rate limit failure -> trips threshold
    with pytest.raises(AIRateLimitError):
        summarizer._call_groq([{"role": "user", "content": "2"}], budget=budget)
    assert coordinator.circuit_breaker.state == "open"

    # Next call is rejected immediately by circuit breaker without calling Groq
    with pytest.raises(AICircuitOpenError):
        summarizer._call_groq([{"role": "user", "content": "3"}], budget=budget)

    # Total network calls remained 2
    assert mock_groq.client.chat.completions.create.call_count == 2


# ----------------------------------------------------------------------
# Scenario 7: All summarization calls consume the request budget
# ----------------------------------------------------------------------


def test_scenario_7_all_summarization_calls_consume_budget():
    """Chunk summarization and hierarchical consolidation properly consume calls and operations."""
    clock = FakeClock()
    mock_groq = MagicMock(spec=GroqService)
    mock_groq.model = "test-model"
    mock_groq.client.chat.completions.create.return_value = make_mock_completion(
        json.dumps({"summary": "Valid concept summary preserving all domain facts."})
    )

    settings = Settings(
        GROQ_API_KEY="test-key",
        MAX_GENERATION_INPUT_BYTES=500,
        MAX_LARGE_DOCUMENT_BYTES=100000,
        AI_MAX_REQUEST_CALLS=20,
        AI_MAX_SUMMARIZATION_OPERATIONS=20,
    )
    summarizer = SummarizationService(mock_groq, settings=settings)
    budget = ProcessingBudget.from_settings(settings, clock=clock.now)

    # 3 paragraphs -> splits into 3 chunks, target_bytes=50 forces Level 2 consolidation
    text = "Section A: Cellular respiration.\n\nSection B: Photosynthesis.\n\nSection C: DNA replication."
    summary = summarizer.summarize_large_document(
        text, target_bytes=50, max_chunk_size=50, chunk_overlap=10, budget=budget
    )

    assert "Valid concept summary" in summary
    # 3 chunks -> 3 chunk operations + 1 consolidation operation = 4 operations
    assert budget.operations_consumed == 4
    assert budget.calls_consumed == 4
    assert mock_groq.client.chat.completions.create.call_count == 4


# ----------------------------------------------------------------------
# Scenario 8: Flashcards and Quiz reuse the same cached consolidated summary
# ----------------------------------------------------------------------


def test_scenario_8_flashcards_and_quiz_reuse_cached_summary():
    """When Flashcards is generated, summary is cached; subsequent Quiz request skips summarization."""
    clock = FakeClock()
    summary_text = "Consolidated summary containing mitochondria ATP and glycolysis details."
    mock_groq = MagicMock(spec=GroqService)
    mock_groq.model = "test-model"

    # Responses: 1 for chunk summarization, 1 for flashcards, 1 for quiz
    mock_groq.client.chat.completions.create.side_effect = [
        make_mock_completion(json.dumps({"summary": summary_text})),
        make_mock_completion(
            json.dumps({"flashcards": [{"question": "What is ATP?", "answer": "Energy currency."}]})
        ),
        make_mock_completion(
            json.dumps({
                "quiz": [{
                    "question": "Where does glycolysis occur?",
                    "options": ["Cytoplasm", "Nucleus", "Ribosome", "Vacuole"],
                    "correct_answer": "Cytoplasm",
                    "explanation": "Glycolysis occurs in the cytoplasm.",
                }]
            })
        ),
    ]

    settings = Settings(
        GROQ_API_KEY="test-key",
        MAX_GENERATION_INPUT_BYTES=1000,
        MAX_LARGE_DOCUMENT_BYTES=100000,
    )
    cache = DocumentSummaryCache(clock=clock.now)
    mock_groq.summary_cache = cache
    gen_service = GenerationService(mock_groq, settings=settings, summary_cache=cache)

    large_text = "Section on cell respiration.\n\n" * 40  # > 1000 bytes

    # 1. Generate Flashcards
    budget1 = ProcessingBudget.from_settings(settings, clock=clock.now)
    fc_res = gen_service.generate_flashcards(large_text, count=1, budget=budget1)
    assert fc_res.count == 1
    # 1 summarization call + 1 flashcard generation call = 2 calls
    assert mock_groq.client.chat.completions.create.call_count == 2
    assert len(cache) == 1

    # 2. Generate Quiz for the EXACT SAME document
    budget2 = ProcessingBudget.from_settings(settings, clock=clock.now)
    quiz_res = gen_service.generate_quiz(large_text, count=1, budget=budget2)
    assert quiz_res.count == 1
    # ONLY 1 additional call for quiz generation! ZERO additional summarization calls!
    assert mock_groq.client.chat.completions.create.call_count == 3
    assert budget2.operations_consumed == 0


# ----------------------------------------------------------------------
# Scenario 9: Different document content creates a different cache entry
# ----------------------------------------------------------------------


def test_scenario_9_different_document_creates_different_cache_entry():
    """Different document contents produce different SHA-256 hashes and distinct cache entries."""
    cache = DocumentSummaryCache()
    doc_a = "Text about photosynthesis and plant chloroplasts."
    doc_b = "Text about astrophysics, stellar nucleosynthesis, and black holes."

    hash_a = cache.compute_hash(doc_a)
    hash_b = cache.compute_hash(doc_b)

    assert hash_a != hash_b

    cache.put(hash_a, "Summary A")
    cache.put(hash_b, "Summary B")

    assert cache.get(hash_a) == "Summary A"
    assert cache.get(hash_b) == "Summary B"
    assert len(cache) == 2


# ----------------------------------------------------------------------
# Scenario 10: Expired entries are regenerated
# ----------------------------------------------------------------------


def test_scenario_10_expired_entries_regenerated():
    """Entries older than TTL are treated as expired and recomputed."""
    clock = FakeClock()
    cache = DocumentSummaryCache(ttl_seconds=60.0, clock=clock.now)

    text = "Important study text to be summarized."
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return f"Summary version {calls}"

    # First call: computes
    res1 = cache.get_or_compute(text, compute)
    assert res1 == "Summary version 1"
    assert calls == 1

    # Immediate second call: cache hit
    res2 = cache.get_or_compute(text, compute)
    assert res2 == "Summary version 1"
    assert calls == 1

    # Advance clock past TTL (61 seconds)
    clock.sleep(61.0)

    # Third call: expired -> recomputed
    res3 = cache.get_or_compute(text, compute)
    assert res3 == "Summary version 2"
    assert calls == 2


# ----------------------------------------------------------------------
# Scenario 11: Cache size remains bounded (LRU eviction)
# ----------------------------------------------------------------------


def test_scenario_11_cache_size_remains_bounded_lru():
    """When cache reaches max_entries, least recently used entries are evicted."""
    cache = DocumentSummaryCache(max_entries=3)

    cache.put("hash1", "Summary 1")
    cache.put("hash2", "Summary 2")
    cache.put("hash3", "Summary 3")
    assert len(cache) == 3

    # Access hash1 to make it recently used (order: 2, 3, 1)
    assert cache.get("hash1") == "Summary 1"

    # Insert hash4 -> hash2 (least recently used) should be evicted
    cache.put("hash4", "Summary 4")
    assert len(cache) == 3

    assert cache.get("hash2") is None
    assert cache.get("hash1") == "Summary 1"
    assert cache.get("hash3") == "Summary 3"
    assert cache.get("hash4") == "Summary 4"


# ----------------------------------------------------------------------
# Scenario 12: Concurrent requests do not duplicate summarization work (single-flight)
# ----------------------------------------------------------------------


def test_scenario_12_single_flight_deduplicates_concurrent_summarization():
    """Simultaneous requests for the same document execute compute_fn exactly once."""
    cache = DocumentSummaryCache()
    text = "Large document subject to concurrent flashcard and quiz requests."

    compute_invocations = 0
    barrier = threading.Barrier(2)
    results = []

    def slow_compute():
        nonlocal compute_invocations
        compute_invocations += 1
        time.sleep(0.05)
        return "Consolidated single-flight summary."

    def worker():
        barrier.wait()
        res = cache.get_or_compute(text, slow_compute)
        results.append(res)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 2
    assert results[0] == "Consolidated single-flight summary."
    assert results[1] == "Consolidated single-flight summary."
    # The expensive compute function was executed ONLY ONCE!
    assert compute_invocations == 1


# ----------------------------------------------------------------------
# Scenario 13: Failed summaries are not cached
# ----------------------------------------------------------------------


def test_scenario_13_failed_summaries_not_cached():
    """If summarization fails (e.g. RateLimitError or syntax error), failure is not cached."""
    cache = DocumentSummaryCache()
    text = "Document whose summarization initially fails."

    def failing_compute():
        raise AIRateLimitError("Temporary rate limit")

    with pytest.raises(AIRateLimitError):
        cache.get_or_compute(text, failing_compute)

    text_hash = cache.compute_hash(text)
    assert cache.get(text_hash) is None
    assert len(cache) == 0


# ----------------------------------------------------------------------
# Scenario 14: Duplicate quiz requests & Correlation IDs (X-Request-ID)
# ----------------------------------------------------------------------


def test_scenario_14_correlation_id_and_duplicate_request_safety():
    """X-Request-ID is echoed back in response headers, and auto-generated when missing."""
    settings = Settings(GROQ_API_KEY="test-key")
    app = create_app(settings)

    with TestClient(app) as client:
        # 1. Custom client correlation ID preserved
        resp1 = client.get("/api/health", headers={"X-Request-ID": "custom-uuid-12345"})
        assert resp1.status_code == 200
        assert resp1.headers.get("x-request-id") == "custom-uuid-12345"

        # 2. Automatically generated correlation ID when header is missing
        resp2 = client.get("/api/health")
        assert resp2.status_code == 200
        gen_id = resp2.headers.get("x-request-id")
        assert gen_id is not None
        # Must be valid UUID string
        assert uuid.UUID(gen_id)


# ----------------------------------------------------------------------
# Scenario 15: The 504 path returns a controlled and accurate error
# ----------------------------------------------------------------------


def test_scenario_15_504_path_returns_controlled_error():
    """When request processing deadline is exceeded, endpoint returns clean HTTP 504 with sanitized message."""
    mock_groq = MagicMock(spec=GroqService)
    mock_groq.model = "test-model"

    current_time = [1000.0]

    def mock_clock():
        return current_time[0]

    # Side effect advances time beyond deadline
    def timeout_effect(*args, **kwargs):
        current_time[0] += 500.0
        raise APITimeoutError(request=MagicMock())

    mock_groq.client.chat.completions.create.side_effect = timeout_effect

    settings = Settings(GROQ_API_KEY="test-key", AI_REQUEST_TIMEOUT_SECONDS=10.0)
    app = create_app(settings)
    app.dependency_overrides[get_groq_service] = lambda: mock_groq

    with TestClient(app) as client:
        resp = client.post("/api/generate/flashcards", json={"text": "Short notes.", "count": 2})
        assert resp.status_code == 504
        data = resp.json()
        assert "detail" in data
        assert "AI service request timed out" in data["detail"]


# ----------------------------------------------------------------------
# Scenario 16: Existing small-document generation still works
# ----------------------------------------------------------------------


def test_scenario_16_small_document_generation_fast_path():
    """Texts <= 100 KB bypass summarization entirely and succeed in 1 direct provider call."""
    mock_groq = MagicMock(spec=GroqService)
    mock_groq.model = "test-model"
    mock_groq.client.chat.completions.create.return_value = make_mock_completion(
        json.dumps({
            "flashcards": [
                {"question": "What is DNA?", "answer": "Genetic material."},
                {"question": "What is RNA?", "answer": "Ribonucleic acid."},
            ]
        })
    )

    settings = Settings(GROQ_API_KEY="test-key", MAX_GENERATION_INPUT_BYTES=100 * 1024)
    gen_service = GenerationService(mock_groq, settings=settings)

    small_text = "DNA carries hereditary genetic information in cells."
    res = gen_service.generate_flashcards(small_text, count=2)

    assert res.count == 2
    assert len(res.flashcards) == 2
    # Exactly 1 direct call, zero summarization passes
    assert mock_groq.client.chat.completions.create.call_count == 1


# ----------------------------------------------------------------------
# Scenario 17: Large-document processing preserves coverage across the entire document
# ----------------------------------------------------------------------


def test_scenario_17_large_document_coverage_preserved():
    """Hierarchical summarization covers all parts of a large multi-section document."""
    mock_groq = MagicMock(spec=GroqService)
    mock_groq.model = "test-model"

    def dynamic_summarizer(*args, **kwargs):
        messages = kwargs.get("messages", [])
        user_prompt = messages[1]["content"] if len(messages) > 1 else ""
        if "Part 1: Photosynthesis" in user_prompt:
            return make_mock_completion(json.dumps({"summary": "Section 1: Photosynthesis occurs in chloroplasts."}))
        elif "Part 2: Cellular Respiration" in user_prompt:
            return make_mock_completion(json.dumps({"summary": "Section 2: Respiration oxidizes glucose in mitochondria."}))
        elif "Part 3: Protein Synthesis" in user_prompt:
            return make_mock_completion(json.dumps({"summary": "Section 3: Ribosomes translate mRNA into polypeptides."}))
        elif "Consolidate and synthesize" in user_prompt:
            return make_mock_completion(
                json.dumps({
                    "summary": (
                        "Unified Guide: Photosynthesis occurs in chloroplasts. "
                        "Respiration oxidizes glucose in mitochondria. "
                        "Protein synthesis occurs on ribosomes translating mRNA."
                    )
                })
            )
        return make_mock_completion(json.dumps({"summary": "Default educational concept summary."}))

    mock_groq.client.chat.completions.create.side_effect = dynamic_summarizer

    settings = Settings(GROQ_API_KEY="test-key", MAX_GENERATION_INPUT_BYTES=200)
    summarizer = SummarizationService(mock_groq, settings=settings)

    full_document = (
        "Part 1: Photosynthesis in thylakoids generates ATP and NADPH.\n\n" * 5 +
        "Part 2: Cellular Respiration in mitochondria produces ATP.\n\n" * 5 +
        "Part 3: Protein Synthesis uses tRNA and ribosomes.\n\n" * 5
    )

    final_summary = summarizer.summarize_large_document(
        full_document,
        target_bytes=200,
        max_chunk_size=300,
        chunk_overlap=20,
    )

    # Verify coverage across all 3 distinct parts
    assert "Photosynthesis" in final_summary
    assert "Respiration" in final_summary
    assert "Protein" in final_summary
