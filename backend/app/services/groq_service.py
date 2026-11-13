"""
Groq AI service.

Initializes and provides the Groq client for AI generation calls.
The client is created once at startup and reused across requests.
"""

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
