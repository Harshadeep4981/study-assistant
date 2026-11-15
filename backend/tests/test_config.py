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
        """Default model is verified working model qwen/qwen3.8-27b."""
        settings = Settings()
        assert settings.GROQ_MODEL == "qwen/qwen3.8-27b"

    def test_provider_resilience_defaults(self):
        settings = Settings()
        assert settings.AI_PROVIDER_RATE_LIMIT_MAX_RETRIES == 5
        assert settings.AI_PROVIDER_TRANSIENT_MAX_RETRIES == 2
        assert settings.AI_PROVIDER_BACKOFF_INITIAL_SECONDS == 1.0
        assert settings.AI_PROVIDER_BACKOFF_MULTIPLIER == 2.0
        assert settings.AI_PROVIDER_BACKOFF_MAX_SECONDS == 16.0
        assert settings.AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS == 0.25
        assert settings.AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD == 3
        assert settings.AI_CIRCUIT_BREAKER_RECOVERY_SECONDS == 30.0

    def test_provider_backoff_maximum_must_not_be_less_than_initial(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="cannot be less than"):
            Settings(AI_PROVIDER_BACKOFF_INITIAL_SECONDS=3, AI_PROVIDER_BACKOFF_MAX_SECONDS=2)

        with pytest.raises(ValidationError, match="cannot be less than"):
            Settings(
                AI_PROVIDER_TRANSIENT_BACKOFF_INITIAL_SECONDS=3,
                AI_PROVIDER_TRANSIENT_BACKOFF_MAX_SECONDS=2,
            )

    def test_default_max_large_document_bytes(self):
        """Default MAX_LARGE_DOCUMENT_BYTES is 500 KB (512,000 bytes) when input bytes is default."""
        settings = Settings()
        assert settings.MAX_LARGE_DOCUMENT_BYTES == 500 * 1024

    def test_max_large_document_bytes_falls_back_when_input_bytes_overridden(self):
        """When MAX_LARGE_DOCUMENT_BYTES is unset and MAX_GENERATION_INPUT_BYTES is customized, it falls back safely."""
        settings = Settings(MAX_GENERATION_INPUT_BYTES=500)
        assert settings.MAX_LARGE_DOCUMENT_BYTES == 500

    def test_explicit_max_large_document_bytes_accepted(self):
        """Explicit MAX_LARGE_DOCUMENT_BYTES exceeding MAX_GENERATION_INPUT_BYTES is accepted."""
        settings = Settings(MAX_GENERATION_INPUT_BYTES=500, MAX_LARGE_DOCUMENT_BYTES=2000)
        assert settings.MAX_LARGE_DOCUMENT_BYTES == 2000

    def test_max_large_document_bytes_cannot_be_less_than_generation_input_bytes(self):
        """MAX_LARGE_DOCUMENT_BYTES cannot be less than MAX_GENERATION_INPUT_BYTES."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="cannot be less than"):
            Settings(MAX_GENERATION_INPUT_BYTES=1000, MAX_LARGE_DOCUMENT_BYTES=500)

    def test_max_large_document_bytes_must_be_positive(self):
        """MAX_LARGE_DOCUMENT_BYTES must be positive (> 0)."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(MAX_LARGE_DOCUMENT_BYTES=0)

        with pytest.raises(ValidationError):
            Settings(MAX_LARGE_DOCUMENT_BYTES=-500)

    def test_default_max_pdf_pages(self):
        """Default MAX_PDF_PAGES is 100."""
        settings = Settings()
        assert settings.MAX_PDF_PAGES == 100

    def test_max_pdf_pages_must_be_positive(self):
        """MAX_PDF_PAGES must be positive (> 0)."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(MAX_PDF_PAGES=0)

        with pytest.raises(ValidationError):
            Settings(MAX_PDF_PAGES=-10)

    def test_max_pdf_pages_cannot_exceed_100(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="cannot exceed the Studywise limit of 100 pages"):
            Settings(MAX_PDF_PAGES=101)

    def test_max_upload_size_bytes_must_be_positive(self):
        """MAX_UPLOAD_SIZE_BYTES must be positive (> 0)."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(MAX_UPLOAD_SIZE_BYTES=0)

        with pytest.raises(ValidationError):
            Settings(MAX_UPLOAD_SIZE_BYTES=-1024)

    def test_default_max_generation_input_bytes(self):
        """Default MAX_GENERATION_INPUT_BYTES is 100 KB (102,400 bytes)."""
        settings = Settings()
        assert settings.MAX_GENERATION_INPUT_BYTES == 100 * 1024

    def test_max_generation_input_bytes_must_be_positive(self):
        """MAX_GENERATION_INPUT_BYTES must be positive (> 0)."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(MAX_GENERATION_INPUT_BYTES=0)

        with pytest.raises(ValidationError):
            Settings(MAX_GENERATION_INPUT_BYTES=-100)

    def test_default_max_generation_request_body_bytes(self):
        """Default MAX_GENERATION_REQUEST_BODY_BYTES is safely derived as (6 * MAX_LARGE_DOCUMENT_BYTES) + 32 KB."""
        settings = Settings()
        expected = (500 * 1024 * 6) + (32 * 1024)
        assert settings.MAX_GENERATION_REQUEST_BODY_BYTES == expected
        assert settings.MAX_GENERATION_REQUEST_BODY_BYTES == 3104768

    def test_changing_only_input_bytes_derives_body_limit_when_unset(self):
        """When MAX_GENERATION_REQUEST_BODY_BYTES is unset, changing MAX_GENERATION_INPUT_BYTES automatically re-derives a safe limit."""
        settings = Settings(MAX_GENERATION_INPUT_BYTES=200000)
        expected_derived = (200000 * 6) + (32 * 1024)
        assert settings.MAX_GENERATION_INPUT_BYTES == 200000
        assert settings.MAX_GENERATION_REQUEST_BODY_BYTES == expected_derived
        assert settings.MAX_GENERATION_REQUEST_BODY_BYTES == 1232768

    def test_explicit_safe_request_body_limits_accepted(self):
        """Explicit values meeting or exceeding the safe derived minimum are accepted."""
        # Minimum safe for 1,000 bytes input is (1,000 * 6) + 32,768 = 38,768
        s_exact = Settings(
            MAX_GENERATION_INPUT_BYTES=1000,
            MAX_GENERATION_REQUEST_BODY_BYTES=38768,
        )
        assert s_exact.MAX_GENERATION_REQUEST_BODY_BYTES == 38768

        s_generous = Settings(
            MAX_GENERATION_INPUT_BYTES=1000,
            MAX_GENERATION_REQUEST_BODY_BYTES=50000,
        )
        assert s_generous.MAX_GENERATION_REQUEST_BODY_BYTES == 50000

    def test_explicit_request_body_limits_below_safe_minimum_rejected(self):
        """Explicit request-body limits below the safe derived minimum are rejected by Settings validation."""
        import pytest
        from pydantic import ValidationError

        # 38,767 is 1 byte below safe minimum (38,768)
        with pytest.raises(ValidationError, match="cannot be less than the safe derived minimum"):
            Settings(
                MAX_GENERATION_INPUT_BYTES=1000,
                MAX_GENERATION_REQUEST_BODY_BYTES=38767,
            )

        # Equal to MAX_GENERATION_INPUT_BYTES (1,000) is rejected because it leaves no room for JSON overhead/escaping
        with pytest.raises(ValidationError, match="cannot be less than the safe derived minimum"):
            Settings(
                MAX_GENERATION_INPUT_BYTES=1000,
                MAX_GENERATION_REQUEST_BODY_BYTES=1000,
            )

        # Legacy 2x limit (2,500 bytes) is rejected because it does not cover worst-case 6x escaping
        with pytest.raises(ValidationError, match="cannot be less than the safe derived minimum"):
            Settings(
                MAX_GENERATION_INPUT_BYTES=1000,
                MAX_GENERATION_REQUEST_BODY_BYTES=2500,
            )

    def test_max_generation_request_body_bytes_must_be_positive(self):
        """MAX_GENERATION_REQUEST_BODY_BYTES must be positive (> 0)."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(MAX_GENERATION_REQUEST_BODY_BYTES=0)

        with pytest.raises(ValidationError):
            Settings(MAX_GENERATION_REQUEST_BODY_BYTES=-500)

    def test_default_ai_max_retries(self):
        """Default AI_MAX_RETRIES is 2."""
        settings = Settings()
        assert settings.AI_MAX_RETRIES == 2

    def test_ai_max_retries_cannot_be_negative(self):
        """AI_MAX_RETRIES cannot be negative."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(AI_MAX_RETRIES=-1)

    def test_default_processing_budget_settings(self):
        """Default request-level processing budget configuration accommodates MAX_LARGE_DOCUMENT_BYTES (500 KB)."""
        settings = Settings()
        assert settings.AI_MAX_REQUEST_CALLS == 120
        assert settings.AI_REQUEST_TIMEOUT_SECONDS == 300.0
        assert settings.AI_MAX_SUMMARIZATION_OPERATIONS == 100

    def test_processing_budget_covers_max_large_document_capacity(self):
        """Processing budget defaults must accommodate the actual hierarchical consolidation algorithm for MAX_LARGE_DOCUMENT_BYTES (500 KB)."""
        from app.services.summarization_service import SummarizationService, chunk_text

        settings = Settings()

        # Test both multi-paragraph text and dense sentence text filling full 500 KB capacity (512,000 bytes)
        sample_texts = [
            ("Section discussing key cellular mechanisms and metabolic pathways in detail.\n\n" * 6500)[
                : settings.MAX_LARGE_DOCUMENT_BYTES
            ],
            ("Short sentence about cellular biology and genetics. More concepts.\n\n" * 15000)[
                : settings.MAX_LARGE_DOCUMENT_BYTES
            ],
            ("Word " * 105000)[: settings.MAX_LARGE_DOCUMENT_BYTES],
        ]

        for text in sample_texts:
            chunks = chunk_text(text, max_chunk_size=8000, chunk_overlap=500)
            assert len(chunks) <= 70

            # Calculate operations strictly matching the actual summarize_large_document() algorithm:
            # 1. Level 1 (Map): 1 operation per chunk
            total_ops = len(chunks)
            current_count = len(chunks)
            depth = 1

            # 2. Level 2..MAX_HIERARCHICAL_DEPTH (Reduce): bundling by 4
            while depth < SummarizationService.MAX_HIERARCHICAL_DEPTH:
                depth += 1
                bundle_ops = 0
                for start_idx in range(0, current_count, 4):
                    bundle_len = min(4, current_count - start_idx)
                    if bundle_len > 1:
                        bundle_ops += 1
                total_ops += bundle_ops
                current_count = (current_count + 3) // 4

            # 3. Final synthesis (while-else block): if current_count > 1, 1 consolidation operation
            if current_count > 1:
                total_ops += 1

            # Total operations must be within configured operation budget (100)
            assert total_ops <= settings.AI_MAX_SUMMARIZATION_OPERATIONS

            # Total base calls (operations + 1 final generation call) must be within request call budget (120)
            base_calls = total_ops + 1
            assert base_calls <= settings.AI_MAX_REQUEST_CALLS
            # Guaranteed headroom of at least 20 correction retries across the request
            assert settings.AI_MAX_REQUEST_CALLS - base_calls >= 20

    def test_processing_budget_settings_must_be_positive(self):
        """Processing budget settings must be strictly positive."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(AI_MAX_REQUEST_CALLS=0)
        with pytest.raises(ValidationError):
            Settings(AI_MAX_REQUEST_CALLS=-5)

        with pytest.raises(ValidationError):
            Settings(AI_REQUEST_TIMEOUT_SECONDS=0)
        with pytest.raises(ValidationError):
            Settings(AI_REQUEST_TIMEOUT_SECONDS=-10.0)

        with pytest.raises(ValidationError):
            Settings(AI_MAX_SUMMARIZATION_OPERATIONS=0)
        with pytest.raises(ValidationError):
            Settings(AI_MAX_SUMMARIZATION_OPERATIONS=-2)

    def test_custom_processing_budget_overrides_accepted(self):
        """Custom processing budget overrides are accepted."""
        settings = Settings(
            AI_MAX_REQUEST_CALLS=25,
            AI_REQUEST_TIMEOUT_SECONDS=45.0,
            AI_MAX_SUMMARIZATION_OPERATIONS=20,
        )
        assert settings.AI_MAX_REQUEST_CALLS == 25
        assert settings.AI_REQUEST_TIMEOUT_SECONDS == 45.0
        assert settings.AI_MAX_SUMMARIZATION_OPERATIONS == 20


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

    def test_max_pdf_pages_read_from_environment(self, monkeypatch):
        """MAX_PDF_PAGES is read from the genuine environment variable."""
        monkeypatch.setenv("MAX_PDF_PAGES", "42")
        get_settings.cache_clear()

        settings = get_settings()
        assert settings.MAX_PDF_PAGES == 42

    def test_max_generation_input_bytes_read_from_environment(self, monkeypatch):
        """MAX_GENERATION_INPUT_BYTES is read from environment and automatically derives body limit when unset."""
        monkeypatch.setenv("MAX_GENERATION_INPUT_BYTES", "204800")
        get_settings.cache_clear()

        settings = get_settings()
        assert settings.MAX_GENERATION_INPUT_BYTES == 204800
        # Confirms changing only MAX_GENERATION_INPUT_BYTES in env safely recomputes the body limit
        expected_derived = (204800 * 6) + (32 * 1024)
        assert settings.MAX_GENERATION_REQUEST_BODY_BYTES == expected_derived
        assert settings.MAX_GENERATION_REQUEST_BODY_BYTES == 1261568

    def test_max_large_document_bytes_read_from_environment(self, monkeypatch):
        """MAX_LARGE_DOCUMENT_BYTES is read from environment variable."""
        monkeypatch.setenv("MAX_GENERATION_INPUT_BYTES", "100000")
        monkeypatch.setenv("MAX_LARGE_DOCUMENT_BYTES", "800000")
        get_settings.cache_clear()

        settings = get_settings()
        assert settings.MAX_LARGE_DOCUMENT_BYTES == 800000

    def test_max_generation_request_body_bytes_read_from_environment(self, monkeypatch):
        """MAX_GENERATION_REQUEST_BODY_BYTES is read from the genuine environment variable."""
        monkeypatch.setenv("MAX_GENERATION_INPUT_BYTES", "100000")
        # Safe minimum for 100,000 is (100,000 * 6) + 32,768 = 632,768
        monkeypatch.setenv("MAX_GENERATION_REQUEST_BODY_BYTES", "700000")
        get_settings.cache_clear()

        settings = get_settings()
        assert settings.MAX_GENERATION_INPUT_BYTES == 100000
        assert settings.MAX_GENERATION_REQUEST_BODY_BYTES == 700000

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
