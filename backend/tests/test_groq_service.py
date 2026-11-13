"""Tests for the Groq service and its FastAPI dependency."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.services.groq_service import GroqService, get_groq_service


class TestGetGroqServiceDependency:
    """The get_groq_service dependency must guard against missing config."""

    def test_raises_503_when_not_configured(self):
        """Requesting Groq when the key is missing must raise HTTP 503."""
        mock_request = MagicMock()
        mock_request.app.state.groq_service = None

        with pytest.raises(HTTPException) as exc_info:
            get_groq_service(mock_request)

        assert exc_info.value.status_code == 503
        assert "GROQ_API_KEY" in exc_info.value.detail

    def test_returns_service_when_configured(self):
        """When Groq is configured the dependency must return the service."""
        sentinel_service = MagicMock(spec=GroqService)
        mock_request = MagicMock()
        mock_request.app.state.groq_service = sentinel_service

        result = get_groq_service(mock_request)
        assert result is sentinel_service

    def test_raises_503_when_state_attribute_missing(self):
        """If groq_service was never set on app.state, raise HTTP 503."""
        mock_request = MagicMock()
        # Simulate app.state having no groq_service attribute at all
        del mock_request.app.state.groq_service

        with pytest.raises(HTTPException) as exc_info:
            get_groq_service(mock_request)

        assert exc_info.value.status_code == 503


class TestGroqServiceInit:
    """GroqService initializes correctly from settings."""

    def test_stores_model_name(self):
        """The configured model name must be accessible."""
        from app.config import Settings

        settings = Settings(GROQ_API_KEY="test-key", GROQ_MODEL="test-model")
        service = GroqService(settings)
        assert service.model == "test-model"

    def test_client_is_created(self):
        """A Groq client object must be created."""
        from app.config import Settings

        settings = Settings(GROQ_API_KEY="test-key")
        service = GroqService(settings)
        assert service.client is not None
