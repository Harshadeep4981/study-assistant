"""
FastAPI application entry point.

Sets up the application, CORS middleware, and routes.
The Groq service is initialized at startup and stored in app state.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.services.groq_service import GroqService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Initializes the Groq service on startup and attaches it to app state.
    This ensures the API key is validated before serving any requests.
    """
    settings = get_settings()
    app.state.settings = settings
    app.state.groq_service = GroqService(settings)
    yield


app = FastAPI(
    title="AI Study Assistant API",
    description="Backend API for generating flashcards and quizzes from study materials.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: default origins for React dev servers (CRA on 3000, Vite on 5173).
# These are safe defaults; they can be overridden via the CORS_ORIGINS env var
# once settings load at startup.  The middleware is added here so that even
# the health endpoint works cross-origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """
    Health-check endpoint.

    Returns the service status and confirms the Groq client is initialised.
    """
    groq_ready = hasattr(app.state, "groq_service") and app.state.groq_service is not None
    return {
        "status": "healthy",
        "groq_client": "connected" if groq_ready else "not initialized",
    }
