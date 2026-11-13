"""Tests for CORS middleware configuration."""

from app.config import Settings, get_settings
from app.main import create_app
from fastapi.testclient import TestClient


class TestDefaultCORS:
    """Default CORS origins (localhost:3000 and localhost:5173) must work."""

    def test_allows_localhost_5173(self, client_no_key):
        response = client_no_key.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_allows_localhost_3000(self, client_no_key):
        response = client_no_key.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_rejects_unconfigured_origin(self, client_no_key):
        """An origin not in the allowed list must not receive CORS headers."""
        response = client_no_key.options(
            "/api/health",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") is None


class TestCustomCORS:
    """CORS_ORIGINS configuration must change the allowed origins."""

    def test_custom_origin_allowed(self):
        """A custom origin specified in CORS_ORIGINS must be allowed."""
        custom = "http://custom-app.example.com:9000"
        settings = Settings(GROQ_API_KEY=None, CORS_ORIGINS=[custom])
        with TestClient(create_app(settings)) as client:
            response = client.options(
                "/api/health",
                headers={
                    "Origin": custom,
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert response.headers.get("access-control-allow-origin") == custom

    def test_default_origin_rejected_when_overridden(self):
        """Default origins must not be allowed when CORS_ORIGINS is overridden."""
        settings = Settings(
            GROQ_API_KEY=None,
            CORS_ORIGINS=["http://only-this.example.com"],
        )
        with TestClient(create_app(settings)) as client:
            response = client.options(
                "/api/health",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert response.headers.get("access-control-allow-origin") is None

    def test_cors_origins_from_env_variable(self, monkeypatch):
        """Setting CORS_ORIGINS via environment variable controls middleware behavior."""
        custom = "http://from-env.example.com:3000"
        monkeypatch.setenv("CORS_ORIGINS", f'["{custom}"]')
        get_settings.cache_clear()
        with TestClient(create_app()) as client:
            response = client.options(
                "/api/health",
                headers={
                    "Origin": custom,
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert response.headers.get("access-control-allow-origin") == custom
