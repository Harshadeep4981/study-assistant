"""Tests for configuration loading."""

from app.config import Settings, get_settings


class TestSettingsDefaults:
    """Settings must have sensible defaults and make GROQ_API_KEY optional."""

    def test_api_key_defaults_to_none(self):
        settings = Settings(GROQ_API_KEY=None)
        assert settings.GROQ_API_KEY is None

    def test_default_cors_origins(self):
        settings = Settings(GROQ_API_KEY=None)
        assert "http://localhost:3000" in settings.CORS_ORIGINS
        assert "http://localhost:5173" in settings.CORS_ORIGINS

    def test_custom_cors_origins(self):
        custom = ["http://example.com:4000"]
        settings = Settings(GROQ_API_KEY=None, CORS_ORIGINS=custom)
        assert settings.CORS_ORIGINS == custom
        assert "http://localhost:3000" not in settings.CORS_ORIGINS

    def test_default_model(self):
        settings = Settings(GROQ_API_KEY=None)
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
