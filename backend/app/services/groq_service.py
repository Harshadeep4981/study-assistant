"""
Groq AI service.

Initializes and provides the Groq client for AI generation calls.
The client is created once at startup and reused across requests.
"""

from fastapi import HTTPException, Request
from groq import Groq

from app.config import Settings


class GroqService:
    """Manages the Groq API client lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self._client = Groq(api_key=settings.GROQ_API_KEY)
        self._model = settings.GROQ_MODEL

    @property
    def client(self) -> Groq:
        """Return the initialized Groq client."""
        return self._client

    @property
    def model(self) -> str:
        """Return the configured model name."""
        return self._model


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
