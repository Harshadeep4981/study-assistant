"""
In-memory cache for reusable large-document intermediate summaries.

Provides a thread-safe, bounded, process-local summary cache keyed by SHA-256
of extracted document text. Eliminates redundant hierarchical summarization passes
when generating both flashcards and quizzes from the same document, or during
user retries after transient failures.

Note on process boundary:
This cache is process-local and stored in memory. In multi-worker or containerized
deployments, each worker maintains its own bounded cache without external persistent
storage or distributed caching dependencies.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import threading
import time
from typing import Callable

from app.services.processing_budget import ProcessingBudget

logger = logging.getLogger(__name__)


class DocumentSummaryCache:
    """
    Thread-safe, bounded LRU cache with TTL expiration and single-flight synchronization
    for intermediate large-document summaries.

    Keyed by SHA-256 hash of extracted document text.
    """

    def __init__(
        self,
        max_entries: int = 50,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """
        Initialize the summary cache.

        Args:
            max_entries: Maximum number of summaries to store before LRU eviction.
            ttl_seconds: Time-to-live in seconds before an entry expires.
            clock: Monotonic clock source for timing and TTL checks.
        """
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._cache: collections.OrderedDict[str, tuple[str, float]] = collections.OrderedDict()
        self._in_flight: dict[str, threading.Event] = {}

    @staticmethod
    def compute_hash(text: str) -> str:
        """Calculate deterministic SHA-256 hex digest for document text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_locked(self, text_hash: str) -> str | None:
        """Retrieve entry if present and unexpired (internal helper, must hold lock)."""
        if text_hash not in self._cache:
            return None
        summary, timestamp = self._cache[text_hash]
        if self._clock() - timestamp > self.ttl_seconds:
            # Expired
            del self._cache[text_hash]
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(text_hash)
        return summary

    def _put_locked(self, text_hash: str, summary: str) -> None:
        """Store summary in cache with current timestamp, evicting oldest if full."""
        if text_hash in self._cache:
            self._cache.move_to_end(text_hash)
        self._cache[text_hash] = (summary, self._clock())
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def get(self, text_hash: str) -> str | None:
        """
        Retrieve cached summary by text SHA-256 digest.

        Returns None on cache miss or TTL expiration.
        """
        with self._lock:
            return self._get_locked(text_hash)

    def put(self, text_hash: str, summary: str) -> None:
        """Directly insert a summary into the cache."""
        with self._lock:
            self._put_locked(text_hash, summary)

    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[], str],
        budget: ProcessingBudget | None = None,
    ) -> str:
        """
        Retrieve summary from cache, or compute and cache it with single-flight deduplication.

        If another thread is already computing the summary for identical document text,
        subsequent callers wait on the single-flight event rather than executing duplicate
        Groq API calls.

        Args:
            text: Document text to summarize.
            compute_fn: Zero-argument callable producing the summary string on cache miss.
            budget: Optional request processing budget to enforce during single-flight wait.

        Returns:
            Consolidated summary string.
        """
        text_hash = self.compute_hash(text)

        while True:
            with self._lock:
                cached = self._get_locked(text_hash)
                if cached is not None:
                    logger.info(
                        "Document summary cache hit",
                        extra={"document_hash": text_hash[:12]},
                    )
                    return cached

                event = self._in_flight.get(text_hash)
                if event is None:
                    event = threading.Event()
                    self._in_flight[text_hash] = event
                    is_waiter = False
                else:
                    is_waiter = True

            if is_waiter:
                logger.info(
                    "Joining in-flight summarization for document",
                    extra={"document_hash": text_hash[:12]},
                )
                while not event.is_set():
                    if budget is not None:
                        budget.check_deadline(min_remaining=0.1)
                    event.wait(timeout=0.25)
                # Event is set, loop back to re-check cache or take over if failed
                continue

            # Computation owner
            logger.info(
                "Document summary cache miss; executing summarization",
                extra={"document_hash": text_hash[:12]},
            )
            try:
                result = compute_fn()
                with self._lock:
                    self._put_locked(text_hash, result)
                return result
            finally:
                with self._lock:
                    self._in_flight.pop(text_hash, None)
                    event.set()

    def clear(self) -> None:
        """Clear all cached summaries."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return count of cached entries (including potentially unpruned expired ones)."""
        with self._lock:
            return len(self._cache)
