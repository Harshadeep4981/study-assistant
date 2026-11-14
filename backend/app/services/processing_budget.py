"""
Request-level processing budget manager.

Tracks and enforces global limits on Groq API calls, summarization/consolidation
operations, and total execution time across the entire request lifecycle.
"""

import time
from typing import Any, Callable

from app.config import Settings
from app.services.exceptions import AIBudgetExceededError, AITimeoutError


class ProcessingBudget:
    """
    Request-level processing budget tracking API calls, operations, and deadlines.

    Ensures that processing limits apply across the entire workflow (chunking,
    Map summarization passes, Reduce consolidation passes, and final generation)
    without resetting per-chunk or per-level.
    """

    def __init__(
        self,
        max_calls: int = 120,
        max_duration_seconds: float = 300.0,
        max_operations: int = 100,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.max_calls = max_calls
        self.max_duration_seconds = max_duration_seconds
        self.max_operations = max_operations
        self._clock = clock or time.monotonic
        self.start_time = self._clock()
        self.calls_consumed = 0
        self.operations_consumed = 0

    @classmethod
    def from_settings(
        cls, settings: Settings, clock: Callable[[], float] | None = None
    ) -> "ProcessingBudget":
        """Instantiate budget directly from application Settings."""
        return cls(
            max_calls=settings.AI_MAX_REQUEST_CALLS,
            max_duration_seconds=settings.AI_REQUEST_TIMEOUT_SECONDS,
            max_operations=settings.AI_MAX_SUMMARIZATION_OPERATIONS,
            clock=clock,
        )

    def elapsed_seconds(self) -> float:
        """Calculate elapsed seconds since request processing started."""
        return self._clock() - self.start_time

    def remaining_seconds(self) -> float:
        """Calculate remaining seconds before request deadline expires."""
        return max(0.0, self.max_duration_seconds - self.elapsed_seconds())

    def check_deadline(self, min_remaining: float = 0.0) -> None:
        """
        Verify that request deadline has not expired and sufficient time remains.

        Args:
            min_remaining: Minimum remaining seconds required. Defaults to 0.0.

        Raises:
            AITimeoutError: If remaining seconds is <= min_remaining.
        """
        if self.remaining_seconds() <= min_remaining:
            raise AITimeoutError(
                f"Request processing timed out: exceeded maximum duration of {self.max_duration_seconds:.1f} seconds."
            )

    def consume_call(self, cost: int = 1, min_remaining: float = 0.1) -> None:
        """
        Record and validate an API call attempt against the request budget.

        Args:
            cost: Number of calls to record against budget. Defaults to 1.
            min_remaining: Minimum remaining seconds required to initiate call. Defaults to 0.1.
                           Avoids initiating provider calls when less than 0.1s remains.

        Raises:
            AITimeoutError: If remaining request duration is <= min_remaining.
            AIBudgetExceededError: If the maximum request call budget is exhausted.
        """
        self.check_deadline(min_remaining=min_remaining)
        if self.calls_consumed + cost > self.max_calls:
            raise AIBudgetExceededError(
                f"Request processing budget exceeded: attempted call {self.calls_consumed + cost} "
                f"exceeds maximum allowance of {self.max_calls} API calls per request."
            )
        self.calls_consumed += cost

    def consume_operation(self, cost: int = 1) -> None:
        """
        Record and validate a summarization or consolidation operation.

        Raises:
            AITimeoutError: If the request deadline has expired.
            AIBudgetExceededError: If the maximum operation budget is exhausted.
        """
        self.check_deadline()
        if self.operations_consumed + cost > self.max_operations:
            raise AIBudgetExceededError(
                f"Request processing budget exceeded: attempted operation {self.operations_consumed + cost} "
                f"exceeds maximum allowance of {self.max_operations} summarization/consolidation operations."
            )
        self.operations_consumed += cost
