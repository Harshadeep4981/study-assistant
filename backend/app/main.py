"""
FastAPI application entry point.

Provides a ``create_app`` factory that builds the fully-configured
application. The module-level ``app`` object is created by calling
the factory with default settings for production use.
"""

from contextlib import asynccontextmanager

from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.middleware import RequestSizeLimitMiddleware
from app.routers.documents import router as documents_router
from app.routers.generate import router as generate_router
from app.services.exceptions import (
    AIBudgetExceededError,
    AIGenerationError,
    AIRateLimitError,
    DocumentProcessingError,
)
from app.services.groq_service import GroqService


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Application factory.

    Creates and configures the FastAPI application instance.

    Args:
        settings: Optional Settings override. When ``None`` (the default),
                  ``get_settings()`` is called, which reads from environment
                  variables and the ``.env`` file. Pass a custom Settings
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
        version="0.3.0",
        lifespan=lifespan,
    )

    # Global handler for document processing domain exceptions
    @application.exception_handler(DocumentProcessingError)
    async def document_error_handler(request: Request, exc: DocumentProcessingError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    # Global handler for AI generation domain exceptions
    @application.exception_handler(AIGenerationError)
    async def ai_generation_error_handler(request: Request, exc: AIGenerationError):
        headers: dict[str, str] = {}
        if isinstance(exc, AIRateLimitError) and exc.retry_after is not None:
            headers["Retry-After"] = str(exc.retry_after)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
            headers=headers if headers else None,
        )

    # Handler for StarletteHTTPException, preserving status code when caused by domain exceptions
    @application.exception_handler(StarletteHTTPException)
    async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, DocumentProcessingError):
            return JSONResponse(
                status_code=cause.status_code,
                content={"detail": cause.message},
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )

    # Request body size limit — enforced on protected POST endpoints before parsing
    application.add_middleware(
        RequestSizeLimitMiddleware,
        max_upload_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
        max_generation_bytes=settings.MAX_GENERATION_REQUEST_BODY_BYTES,
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
    # Routers
    # ------------------------------------------------------------------
    application.include_router(documents_router)
    application.include_router(generate_router)

    # ------------------------------------------------------------------
    # Health check route
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
