"""
Application configuration using Pydantic Settings.

Loads environment variables from a .env file placed in the backend/ directory.
The GROQ_API_KEY is optional; the application starts without it, but AI
features will be unavailable until it is configured.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the backend/ directory so the .env file is found regardless of
# whether the application is launched from the project root or backend/.
_BACKEND_DIR = Path(__file__).resolve().parent.parent


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
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # CORS: allowed origins for the React dev server
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Maximum upload size in bytes (default: 10 MB)
    MAX_UPLOAD_SIZE_BYTES: int = Field(default=10 * 1024 * 1024, gt=0)

    # Maximum allowed PDF page count (default: 100)
    MAX_PDF_PAGES: int = Field(default=100, gt=0)

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @field_validator("MAX_UPLOAD_SIZE_BYTES")
    @classmethod
    def validate_max_upload_size_bytes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_BYTES must be positive.")
        return v

    @field_validator("MAX_PDF_PAGES")
    @classmethod
    def validate_max_pdf_pages(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MAX_PDF_PAGES must be positive.")
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
