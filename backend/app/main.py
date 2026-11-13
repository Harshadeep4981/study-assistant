"""
FastAPI application entry point.

Provides a ``create_app`` factory that builds the fully-configured
application.  The module-level ``app`` object is created by calling
the factory with default settings for production use.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.services.groq_service import GroqService


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Application factory.

    Creates and configures the FastAPI application instance.

    Args:
        settings: Optional Settings override.  When ``None`` (the default),
                  ``get_settings()`` is called, which reads from environment
                  variables and the ``.env`` file.  Pass a custom Settings
                  instance in tests to control configuration without touching
                  the environment.
    """
    if settings is None:
        settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialise the Groq service on startup if an API key is present."""
        app.state.settings = settings
        if settings.GROQ_API_KEY:
            app.state.groq_service = GroqService(settings)
        else:
            app.state.groq_service = None
        yield

    application = FastAPI(
        title="AI Study Assistant API",
        description="Backend API for generating flashcards and quizzes from study materials.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — configured at app-creation time from Settings.CORS_ORIGINS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @application.get("/api/health")
    async def health_check():
        """
        Lightweight health-check endpoint.

        Returns backend availability and Groq configuration state.
        Does **not** make any network calls to Groq.
        """
        groq_service = getattr(application.state, "groq_service", None)
        return {
            "status": "healthy",
            "groq_client": "initialized" if groq_service is not None else "not_configured",
        }

    return application


# Production application instance
app = create_app()
