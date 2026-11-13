"""
Application configuration using Pydantic Settings.

Loads environment variables from a .env file placed in the backend/ directory.
The GROQ_API_KEY is required; the application will fail to start without it.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required: Groq API key for AI generation
    GROQ_API_KEY: str

    # Groq model to use for generation
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # CORS: allowed origins for the React dev server
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000


def get_settings() -> Settings:
    """
    Create and return a Settings instance.

    Raises ValidationError if required environment variables are missing.
    """
    return Settings()
