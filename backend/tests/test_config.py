"""Tests for configuration loading."""

from app.config import Settings, get_settings
from app.main import create_app
from fastapi.testclient import TestClient


class TestSettingsDefaults:
    """Settings must have sensible defaults and make GROQ_API_KEY optional."""

    def test_api_key_defaults_to_none(self):
        """Without GROQ_API_KEY in the environment, it defaults to None."""
        settings = Settings()
        assert settings.GROQ_API_KEY is None

    def test_default_cors_origins(self):
        """Default CORS origins include both localhost:3000 and localhost:5173."""
        settings = Settings()
        assert "http://localhost:3000" in settings.CORS_ORIGINS
        assert "http://localhost:5173" in settings.CORS_ORIGINS

    def test_custom_cors_origins(self):
        """Explicit constructor argument overrides defaults."""
        custom = ["http://example.com:4000"]
        settings = Settings(CORS_ORIGINS=custom)
        assert settings.CORS_ORIGINS == custom
        assert "http://localhost:3000" not in settings.CORS_ORIGINS

    def test_default_model(self):
        """Default model is llama-3.1-8b-instant."""
        settings = Settings()
        assert settings.GROQ_MODEL == "llama-3.1-8b-instant"


class TestGetSettingsCache:
    """get_settings must be cached and clearable."""

    def test_returns_same_instance(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_returns_new_instance(self):
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # After clearing, a new object is created (may be equal, but not same id)
        assert s1 is not s2


class TestEnvironmentVariableLoading:
    """Settings must read configuration from genuine environment variables."""

    def test_groq_api_key_read_from_environment(self, monkeypatch):
        """GROQ_API_KEY is read from the genuine environment variable."""
        expected_key = "gsk_env_test_key_abc123"
        monkeypatch.setenv("GROQ_API_KEY", expected_key)
        get_settings.cache_clear()

        settings = get_settings()
        assert settings.GROQ_API_KEY == expected_key

    def test_cors_origins_parsed_from_json_environment_variable(self, monkeypatch):
        """CORS_ORIGINS is correctly parsed from a JSON list in the environment."""
        custom_origins = '["http://env-origin-a.com:3000", "http://env-origin-b.com:8080"]'
        monkeypatch.setenv("CORS_ORIGINS", custom_origins)
        get_settings.cache_clear()

        settings = get_settings()
        assert settings.CORS_ORIGINS == [
            "http://env-origin-a.com:3000",
            "http://env-origin-b.com:8080",
        ]

    def test_get_settings_picks_up_changed_env_after_cache_clear(self, monkeypatch):
        """get_settings() returns cached values until cache_clear() is invoked."""
        monkeypatch.setenv("GROQ_MODEL", "model-version-1")
        get_settings.cache_clear()
        s1 = get_settings()
        assert s1.GROQ_MODEL == "model-version-1"

        # Change environment variable; cached instance should persist
        monkeypatch.setenv("GROQ_MODEL", "model-version-2")
        assert get_settings().GROQ_MODEL == "model-version-1"

        # After clearing cache, new environment value must be loaded
        get_settings.cache_clear()
        s2 = get_settings()
        assert s2.GROQ_MODEL == "model-version-2"
        assert s1 is not s2

    def test_cors_origins_from_env_used_by_create_app_middleware(self, monkeypatch):
        """create_app() with default settings applies CORS origins loaded from environment."""
        allowed_origin = "http://env-front-app.example.com:4000"
        monkeypatch.setenv("CORS_ORIGINS", f'["{allowed_origin}"]')
        get_settings.cache_clear()

        # create_app() with no arguments calls get_settings(), which reads env
        app = create_app()
        with TestClient(app) as client:
            # 1. Configured origin receives CORS header
            resp_allowed = client.options(
                "/api/health",
                headers={
                    "Origin": allowed_origin,
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert resp_allowed.headers.get("access-control-allow-origin") == allowed_origin

            # 2. Unconfigured origin (e.g. localhost:5173 which is no longer in origins) is rejected
            resp_rejected = client.options(
                "/api/health",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert resp_rejected.headers.get("access-control-allow-origin") is None
