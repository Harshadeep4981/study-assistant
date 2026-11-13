"""Shared fixtures for the test suite."""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """
    Ensure complete test isolation from ambient environment variables and external .env.

    Clears settings cache before and after every test, disables loading of any external
    .env file, and strips ambient environment variables so tests run predictably regardless
    of developer machine state.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings_no_key():
    """Settings instance with no Groq API key."""
    return Settings(GROQ_API_KEY=None)


@pytest.fixture
def settings_with_key():
    """Settings instance with a dummy Groq API key (no network calls)."""
    return Settings(GROQ_API_KEY="test-key-not-real")


@pytest.fixture
def app_no_key(settings_no_key):
    """FastAPI application created without a Groq API key."""
    return create_app(settings_no_key)


@pytest.fixture
def app_with_key(settings_with_key):
    """FastAPI application created with a dummy Groq API key."""
    return create_app(settings_with_key)


@pytest.fixture
def client_no_key(app_no_key):
    """Test client for an app without a Groq API key."""
    with TestClient(app_no_key) as client:
        yield client


@pytest.fixture
def client_with_key(app_with_key):
    """Test client for an app with a dummy Groq API key."""
    with TestClient(app_with_key) as client:
        yield client
