"""
Application configuration using Pydantic Settings.

Loads environment variables from a .env file placed in the backend/ directory.
The GROQ_API_KEY is optional; the application starts without it, but AI
features will be unavailable until it is configured.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the backend/ directory so the .env file is found regardless of
# whether the application is launched from the project root or backend/.
_BACKEND_DIR = Path(__file__).resolve().parent.parent

# Multiplier accounting for worst-case JSON serialization expansion:
# 1-byte control characters (e.g. \x00-\x1F) or non-ASCII characters serialized
# with ensure_ascii=True expand to 6 ASCII characters (\u00XX).
JSON_ESCAPING_EXPANSION_FACTOR: int = 6

# Conservative envelope buffer for JSON keys, formatting, and parameters (32 KB)
JSON_ENVELOPE_OVERHEAD_BYTES: int = 32 * 1024


def calculate_safe_generation_request_body_bytes(max_input_bytes: int) -> int:
    """
    Calculate the conservative minimum request body size for generation endpoints.

    Accounts for worst-case JSON serialization expansion where single-byte control
    characters or non-ASCII characters expand to 6 ASCII characters (\\u00XX),
    plus a 32 KB envelope buffer for JSON structure and parameters.
    """
    return (max_input_bytes * JSON_ESCAPING_EXPANSION_FACTOR) + JSON_ENVELOPE_OVERHEAD_BYTES


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Optional: Groq API key for AI generation.
    # The application starts without it; AI features are unavailable until set.
    GROQ_API_KEY: str | None = None

    # Groq model to use for generation
    GROQ_MODEL: str = "qwen/qwen3.8-27b"

    # CORS: allowed origins for the React dev server
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Maximum upload size in bytes (default: 10 MB)
    MAX_UPLOAD_SIZE_BYTES: int = Field(default=10 * 1024 * 1024, gt=0)

    # Maximum AI generation direct text input size in bytes (default: 100 KB = 102,400 bytes).
    # Texts up to this size proceed directly to generation without summarization.
    MAX_GENERATION_INPUT_BYTES: int = Field(default=100 * 1024, gt=0)

    # Maximum supported document text size for large-document processing in bytes (default: 500 KB = 512,000 bytes).
    # Inputs between MAX_GENERATION_INPUT_BYTES and MAX_LARGE_DOCUMENT_BYTES undergo semantic chunking
    # and hierarchical summarization before generation. Inputs exceeding this limit are rejected with HTTP 413.
    # When unset, defaults to 500 KB when MAX_GENERATION_INPUT_BYTES is 100 KB, or matches MAX_GENERATION_INPUT_BYTES
    # for explicit small-budget test overrides.
    MAX_LARGE_DOCUMENT_BYTES: int | None = Field(default=None, gt=0)

    # Maximum raw request body size for generation endpoints (default: None -> derived as 6 * MAX_LARGE_DOCUMENT_BYTES + 32 KB)
    # Safely accounts for worst-case JSON escaping (\u00XX) plus JSON envelope.
    MAX_GENERATION_REQUEST_BODY_BYTES: int | None = Field(default=None, gt=0)

    # Maximum allowed PDF page count (default: 100)
    MAX_PDF_PAGES: int = Field(default=100, gt=0)

    # Maximum AI generation retry attempts on malformed output (default: 2 retries -> up to 3 total attempts)
    AI_MAX_RETRIES: int = Field(default=2, ge=0, le=5)

    # Request-level processing budget limits
    # Maximum total Groq API attempts per user request (across chunks, retries, consolidation, and generation).
    # Sized to accommodate MAX_LARGE_DOCUMENT_BYTES (500 KB produces <=69 chunks: Level 1 has 69 ops,
    # Level 2 has 17 ops, Level 3 has 5 ops, and final synthesis has 1 op -> 92 base summarization ops + 1 generation = 93 base calls,
    # leaving at least 25 retry attempts within the 120-call budget).
    AI_MAX_REQUEST_CALLS: int = Field(default=120, gt=0)

    # Maximum total processing duration per user request in seconds (default: 300.0s = 5 minutes).
    AI_REQUEST_TIMEOUT_SECONDS: float = Field(default=300.0, gt=0)

    # Maximum summarization and consolidation operations per user request.
    # Sized to accommodate MAX_LARGE_DOCUMENT_BYTES (500 KB: 69 chunk ops + 17 L2 + 5 L3 + 1 final synthesis = 92 operations, <= 100).
    AI_MAX_SUMMARIZATION_OPERATIONS: int = Field(default=100, gt=0)

    # Provider retries are separate from AI_MAX_RETRIES, which retries malformed model output.
    AI_PROVIDER_RATE_LIMIT_MAX_RETRIES: int = Field(default=5, ge=0, le=10)
    AI_PROVIDER_TRANSIENT_MAX_RETRIES: int = Field(default=2, ge=0, le=5)
    AI_PROVIDER_BACKOFF_INITIAL_SECONDS: float = Field(default=1.0, ge=0)
    AI_PROVIDER_BACKOFF_MULTIPLIER: float = Field(default=2.0, ge=1.0)
    AI_PROVIDER_BACKOFF_MAX_SECONDS: float = Field(default=16.0, gt=0)
    AI_PROVIDER_TRANSIENT_BACKOFF_INITIAL_SECONDS: float = Field(default=0.5, ge=0)
    AI_PROVIDER_TRANSIENT_BACKOFF_MAX_SECONDS: float = Field(default=2.0, gt=0)
    AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS: float = Field(default=0.25, ge=0)
    AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=3, gt=0)
    AI_CIRCUIT_BREAKER_RECOVERY_SECONDS: float = Field(default=30.0, gt=0)

    # In-memory document summary cache configuration
    AI_SUMMARY_CACHE_MAX_ENTRIES: int = Field(default=50, gt=0)
    AI_SUMMARY_CACHE_TTL_SECONDS: float = Field(default=3600.0, gt=0)

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @field_validator("MAX_UPLOAD_SIZE_BYTES")
    @classmethod
    def validate_max_upload_size_bytes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_BYTES must be positive.")
        return v

    @field_validator("MAX_GENERATION_INPUT_BYTES")
    @classmethod
    def validate_max_generation_input_bytes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MAX_GENERATION_INPUT_BYTES must be positive.")
        return v

    @field_validator("MAX_LARGE_DOCUMENT_BYTES")
    @classmethod
    def validate_max_large_document_bytes(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("MAX_LARGE_DOCUMENT_BYTES must be positive.")
        return v

    @field_validator("MAX_GENERATION_REQUEST_BODY_BYTES")
    @classmethod
    def validate_max_generation_request_body_bytes(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("MAX_GENERATION_REQUEST_BODY_BYTES must be positive.")
        return v

    @model_validator(mode="after")
    def resolve_generation_request_body_bytes(self) -> "Settings":
        # Resolve MAX_LARGE_DOCUMENT_BYTES
        if self.MAX_LARGE_DOCUMENT_BYTES is None:
            if self.MAX_GENERATION_INPUT_BYTES == 100 * 1024:
                self.MAX_LARGE_DOCUMENT_BYTES = 500 * 1024
            else:
                self.MAX_LARGE_DOCUMENT_BYTES = self.MAX_GENERATION_INPUT_BYTES
        elif self.MAX_LARGE_DOCUMENT_BYTES < self.MAX_GENERATION_INPUT_BYTES:
            raise ValueError(
                f"MAX_LARGE_DOCUMENT_BYTES ({self.MAX_LARGE_DOCUMENT_BYTES}) cannot be less than "
                f"MAX_GENERATION_INPUT_BYTES ({self.MAX_GENERATION_INPUT_BYTES})."
            )

        safe_minimum = calculate_safe_generation_request_body_bytes(self.MAX_LARGE_DOCUMENT_BYTES)
        if self.MAX_GENERATION_REQUEST_BODY_BYTES is None:
            self.MAX_GENERATION_REQUEST_BODY_BYTES = safe_minimum
        elif self.MAX_GENERATION_REQUEST_BODY_BYTES < safe_minimum:
            raise ValueError(
                f"MAX_GENERATION_REQUEST_BODY_BYTES ({self.MAX_GENERATION_REQUEST_BODY_BYTES}) "
                f"cannot be less than the safe derived minimum of {safe_minimum} bytes "
                f"((MAX_LARGE_DOCUMENT_BYTES * {JSON_ESCAPING_EXPANSION_FACTOR}) + "
                f"{JSON_ENVELOPE_OVERHEAD_BYTES // 1024} KB) to guarantee safe JSON serialization."
            )
        return self

    @model_validator(mode="after")
    def validate_provider_retry_bounds(self) -> "Settings":
        if self.AI_PROVIDER_BACKOFF_MAX_SECONDS < self.AI_PROVIDER_BACKOFF_INITIAL_SECONDS:
            raise ValueError("AI_PROVIDER_BACKOFF_MAX_SECONDS cannot be less than the initial backoff.")
        if self.AI_PROVIDER_TRANSIENT_BACKOFF_MAX_SECONDS < self.AI_PROVIDER_TRANSIENT_BACKOFF_INITIAL_SECONDS:
            raise ValueError("AI_PROVIDER_TRANSIENT_BACKOFF_MAX_SECONDS cannot be less than the transient initial backoff.")
        return self

    @field_validator("MAX_PDF_PAGES")
    @classmethod
    def validate_max_pdf_pages(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MAX_PDF_PAGES must be positive.")
        if v > 100:
            raise ValueError("MAX_PDF_PAGES cannot exceed the Studywise limit of 100 pages.")
        return v

    @field_validator("AI_MAX_RETRIES")
    @classmethod
    def validate_ai_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("AI_MAX_RETRIES cannot be negative.")
        return v

    @field_validator("AI_MAX_REQUEST_CALLS")
    @classmethod
    def validate_ai_max_request_calls(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("AI_MAX_REQUEST_CALLS must be positive.")
        return v

    @field_validator("AI_REQUEST_TIMEOUT_SECONDS")
    @classmethod
    def validate_ai_request_timeout_seconds(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("AI_REQUEST_TIMEOUT_SECONDS must be positive.")
        return v

    @field_validator("AI_MAX_SUMMARIZATION_OPERATIONS")
    @classmethod
    def validate_ai_max_summarization_operations(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("AI_MAX_SUMMARIZATION_OPERATIONS must be positive.")
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Create and return a cached Settings instance.

    The result is cached so that configuration is loaded once and reused
    across the application lifetime. Call ``get_settings.cache_clear()``
    to force a reload (useful in tests).
    """
    return Settings()
