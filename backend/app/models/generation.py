"""
Pydantic models for AI study material generation.

Defines request and response schemas for flashcards and multiple-choice quizzes,
including strict validation for options, counts, and non-empty fields.
"""

from pydantic import BaseModel, Field, field_validator, model_validator


class FlashcardItem(BaseModel):
    """A single flashcard testing a core concept."""

    question: str = Field(..., description="Concept question testing understanding")
    answer: str = Field(..., description="Accurate, concise, and understandable answer")

    @field_validator("question", "answer")
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Flashcard '{info.field_name}' cannot be empty or whitespace.")
        return v.strip()


class FlashcardGenerateRequest(BaseModel):
    """Request payload for flashcard generation."""

    text: str = Field(..., description="Source study notes or text")
    count: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Number of flashcards to generate (1 to 30, default 5)",
    )


class FlashcardGenerateResponse(BaseModel):
    """Response payload containing generated flashcards."""

    flashcards: list[FlashcardItem]
    count: int


class QuizQuestion(BaseModel):
    """A single multiple-choice quiz question."""

    question: str = Field(..., description="Clear question testing an important concept")
    options: list[str] = Field(..., description="Exactly four distinct choices")
    correct_answer: str = Field(
        ..., description="The single correct choice (must match one of the options verbatim)"
    )
    explanation: str = Field(
        ..., description="Brief explanation of why the correct answer is right"
    )

    @field_validator("question", "explanation")
    @classmethod
    def validate_non_empty_strings(cls, v: str, info) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Quiz '{info.field_name}' cannot be empty or whitespace.")
        return v.strip()

    @field_validator("options")
    @classmethod
    def validate_options(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list) or len(v) != 4:
            raise ValueError(
                f"Quiz question must have exactly 4 options; received {len(v) if isinstance(v, list) else 0}."
            )
        cleaned = [opt.strip() for opt in v if isinstance(opt, str) and opt.strip()]
        if len(cleaned) != 4:
            raise ValueError("All 4 quiz options must be non-empty strings.")
        if len(set(cleaned)) != 4:
            raise ValueError("All 4 quiz options must be distinct (no duplicate choices).")
        return cleaned

    @field_validator("correct_answer")
    @classmethod
    def validate_correct_answer(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Quiz 'correct_answer' cannot be empty or whitespace.")
        return v.strip()

    @model_validator(mode="after")
    def validate_correct_answer_in_options(self) -> "QuizQuestion":
        """Verify that correct_answer exactly matches one of the four validated options."""
        if self.correct_answer not in self.options:
            raise ValueError(
                f"Quiz 'correct_answer' ('{self.correct_answer}') must match one of the options: {self.options}."
            )
        return self


class QuizGenerateRequest(BaseModel):
    """Request payload for quiz generation."""

    text: str = Field(..., description="Source study notes or text")
    count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of quiz questions to generate (1 to 20, default 5)",
    )


class QuizGenerateResponse(BaseModel):
    """Response payload containing generated quiz questions."""

    quiz: list[QuizQuestion]
    count: int
