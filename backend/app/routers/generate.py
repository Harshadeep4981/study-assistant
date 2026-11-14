"""
AI Study Material Generation Router.

Provides endpoints for generating flashcards and multiple-choice quizzes
from study text using the Groq AI service.
"""

from fastapi import APIRouter, Depends, Request, status

from app.config import Settings, get_settings
from app.middleware import format_byte_size
from app.models.generation import (
    FlashcardGenerateRequest,
    FlashcardGenerateResponse,
    QuizGenerateRequest,
    QuizGenerateResponse,
)
from app.services.exceptions import EmptyTextError, OversizedFileError
from app.services.generation_service import GenerationService
from app.services.groq_service import GroqService, get_groq_service

router = APIRouter(prefix="/api/generate", tags=["Generation"])


def get_generation_service(
    request: Request,
    groq_service: GroqService = Depends(get_groq_service),
) -> GenerationService:
    """Dependency providing a configured GenerationService instance."""
    settings = getattr(request.app.state, "settings", None) or get_settings()
    return GenerationService(groq_service=groq_service, settings=settings)


@router.post(
    "/flashcards",
    response_model=FlashcardGenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate study flashcards",
    description=(
        "Generate concept-focused study flashcards from provided study text. "
        "Validates that output strictly matches the requested schema and count, "
        "with automatic bounded retries for malformed AI responses."
    ),
    responses={
        200: {"description": "Flashcards successfully generated."},
        400: {"description": "Empty or whitespace-only text input."},
        413: {"description": "Text or request body exceeds maximum generation size limit."},
        422: {"description": "Invalid count parameter (must be 1-30)."},
        429: {"description": "AI provider rate limit exceeded."},
        502: {"description": "AI provider error or invalid output after retries."},
        503: {"description": "Groq API key not configured on server."},
        504: {"description": "AI provider request timed out."},
    },
)
def generate_flashcards(
    payload: FlashcardGenerateRequest,
    service: GenerationService = Depends(get_generation_service),
) -> FlashcardGenerateResponse:
    """Validate input text and generate flashcards."""
    # 1. Empty or whitespace text check
    if not payload.text or not payload.text.strip():
        raise EmptyTextError("Text input cannot be empty or contain only whitespace.")

    # 2. Maximum input size check against supported capacity
    raw_bytes_len = len(payload.text.encode("utf-8"))
    if raw_bytes_len > service.settings.MAX_LARGE_DOCUMENT_BYTES:
        limit_str = format_byte_size(service.settings.MAX_LARGE_DOCUMENT_BYTES)
        raise OversizedFileError(
            f"Text input ({raw_bytes_len} bytes) exceeds the maximum limit of {limit_str}."
        )

    return service.generate_flashcards(text=payload.text, count=payload.count)


@router.post(
    "/quiz",
    response_model=QuizGenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate multiple-choice quiz",
    description=(
        "Generate multiple-choice quiz questions with 4 distinct options and explanations "
        "from provided study text. Validates schema and options consistency, with automatic "
        "bounded retries for malformed AI responses."
    ),
    responses={
        200: {"description": "Quiz questions successfully generated."},
        400: {"description": "Empty or whitespace-only text input."},
        413: {"description": "Text or request body exceeds maximum generation size limit."},
        422: {"description": "Invalid count parameter (must be 1-20)."},
        429: {"description": "AI provider rate limit exceeded."},
        502: {"description": "AI provider error or invalid output after retries."},
        503: {"description": "Groq API key not configured on server."},
        504: {"description": "AI provider request timed out."},
    },
)
def generate_quiz(
    payload: QuizGenerateRequest,
    service: GenerationService = Depends(get_generation_service),
) -> QuizGenerateResponse:
    """Validate input text and generate multiple-choice quiz questions."""
    # 1. Empty or whitespace text check
    if not payload.text or not payload.text.strip():
        raise EmptyTextError("Text input cannot be empty or contain only whitespace.")

    # 2. Maximum input size check against supported capacity
    raw_bytes_len = len(payload.text.encode("utf-8"))
    if raw_bytes_len > service.settings.MAX_LARGE_DOCUMENT_BYTES:
        limit_str = format_byte_size(service.settings.MAX_LARGE_DOCUMENT_BYTES)
        raise OversizedFileError(
            f"Text input ({raw_bytes_len} bytes) exceeds the maximum limit of {limit_str}."
        )

    return service.generate_quiz(text=payload.text, count=payload.count)
