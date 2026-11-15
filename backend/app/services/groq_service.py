"""
Groq AI service.

Initializes and provides the Groq client for AI generation calls.
The client is created once at startup and reused across requests.
"""

from fastapi import HTTPException, Request
from groq import Groq

from app.config import Settings
from app.services.provider_resilience import ProviderRequestCoordinator
from app.services.summary_cache import DocumentSummaryCache


class GroqService:
    """Manages the Groq API client lifecycle."""

    def __init__(
        self,
        settings: Settings,
        *,
        coordinator: ProviderRequestCoordinator | None = None,
        summary_cache: DocumentSummaryCache | None = None,
    ) -> None:
        self._client = Groq(api_key=settings.GROQ_API_KEY)
        self._model = settings.GROQ_MODEL
        self._settings = settings
        self._coordinator = coordinator or ProviderRequestCoordinator(settings)
        self._summary_cache = summary_cache or DocumentSummaryCache(
            max_entries=getattr(settings, "AI_SUMMARY_CACHE_MAX_ENTRIES", 50),
            ttl_seconds=getattr(settings, "AI_SUMMARY_CACHE_TTL_SECONDS", 3600.0),
        )

    @property
    def client(self) -> Groq:
        """Return the initialized Groq client."""
        return self._client

    @property
    def model(self) -> str:
        """Return the configured model name."""
        return self._model

    @property
    def coordinator(self) -> ProviderRequestCoordinator:
        """Return process-shared pacing and circuit state for provider requests."""
        return self._coordinator

    @property
    def summary_cache(self) -> DocumentSummaryCache:
        """Return process-shared in-memory document summary cache."""
        return self._summary_cache

    @property
    def settings(self) -> Settings:
        """Return the settings shared by generation and provider resilience policies."""
        return self._settings


def get_groq_service(request: Request) -> GroqService:
    """
    FastAPI dependency that provides the GroqService instance.

    Raises HTTP 503 if the Groq API key was not configured at startup.
    Use with ``Depends(get_groq_service)`` in endpoint signatures.
    """
    service = getattr(request.app.state, "groq_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Groq API key is not configured. "
                "Set the GROQ_API_KEY environment variable."
            ),
        )
    return service
