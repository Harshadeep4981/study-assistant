"""
Automated tests for Milestone 3: AI Generation Pipeline.

Verifies:
- Successful flashcard and quiz generation
- Parsing of markdown-wrapped JSON and edge-case formatting
- Strict Pydantic validation (4 unique options, correct_answer in options, non-empty fields)
- Bounded retries with correction prompts
- Provider failure mappings (auth 502, rate limit 429, timeout 504, connection 502, unconfigured 503)
- Input validation (empty text 400, invalid counts 422, oversized text 413)
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app
from app.models.generation import QuizQuestion
from app.services.exceptions import AIGenerationValidationError
from app.services.generation_service import GenerationService
from app.services.groq_service import GroqService, get_groq_service


# ----------------------------------------------------------------------
# Helpers for mocking Groq completions
# ----------------------------------------------------------------------


def make_mock_completion(content: str):
    """Build a mock Groq ChatCompletion object."""
    mock = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock.choices = [mock_choice]
    return mock


@pytest.fixture
def mock_groq():
    """Provides a mocked GroqService instance with a mock client."""
    mock_service = MagicMock(spec=GroqService)
    mock_service.model = "llama-3.1-8b-instant"
    mock_client = MagicMock()
    mock_service.client = mock_client
    return mock_service


@pytest.fixture
def client_with_mock_groq(mock_groq):
    """TestClient wired with mock GroqService dependency."""
    settings = Settings(GROQ_API_KEY="test-mock-key")
    app = create_app(settings)
    app.dependency_overrides[get_groq_service] = lambda: mock_groq
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# 1. Successful Generation Tests
# ----------------------------------------------------------------------


class TestSuccessfulGeneration:
    """Verifies successful flashcard and quiz generation when AI returns valid JSON."""

    def test_valid_flashcard_generation(self, client_with_mock_groq, mock_groq):
        """Valid AI output produces 200 OK with correct flashcard schema and count."""
        raw_json = json.dumps(
            {
                "flashcards": [
                    {
                        "question": "What is photosynthesis?",
                        "answer": "The process by which green plants synthesize nutrients using sunlight.",
                    },
                    {
                        "question": "Where does photosynthesis occur in plant cells?",
                        "answer": "In the chloroplasts containing chlorophyll.",
                    },
                ]
            }
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(raw_json)

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Photosynthesis occurs in chloroplasts and uses sunlight to produce nutrients.", "count": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["flashcards"]) == 2
        assert data["flashcards"][0]["question"] == "What is photosynthesis?"
        assert "chloroplasts" in data["flashcards"][1]["answer"]

    def test_valid_quiz_generation(self, client_with_mock_groq, mock_groq):
        """Valid AI output produces 200 OK with 4 distinct options and explanation."""
        raw_json = json.dumps(
            {
                "quiz": [
                    {
                        "question": "Which organelle is known as the powerhouse of the cell?",
                        "options": ["Nucleus", "Mitochondria", "Ribosome", "Endoplasmic Reticulum"],
                        "correct_answer": "Mitochondria",
                        "explanation": "Mitochondria generate most of the chemical energy needed by the cell.",
                    }
                ]
            }
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(raw_json)

        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "Mitochondria generate cellular energy in the form of ATP.", "count": 1},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["quiz"]) == 1
        # Check that response contains 'quiz' and does NOT contain 'questions'
        assert "quiz" in data
        assert "questions" not in data
        assert "questions" not in response.text
        q = data["quiz"][0]
        assert q["question"] == "Which organelle is known as the powerhouse of the cell?"
        assert len(q["options"]) == 4
        assert q["correct_answer"] == "Mitochondria"
        assert q["correct_answer"] in q["options"]
        assert "chemical energy" in q["explanation"]

    def test_custom_count_honored(self, client_with_mock_groq, mock_groq):
        """Generating with custom count returns the requested number of items."""
        items = [
            {"question": f"Question {i}?", "answer": f"Answer {i}."}
            for i in range(1, 4)
        ]
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(
            json.dumps({"flashcards": items})
        )

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Study text for custom count test.", "count": 3},
        )

        assert response.status_code == 200
        assert response.json()["count"] == 3
        assert len(response.json()["flashcards"]) == 3


# ----------------------------------------------------------------------
# 2. Malformed AI Responses Tests
# ----------------------------------------------------------------------


class TestMalformedAIResponses:
    """Verifies parsing of edge cases (markdown fences, conversational text) and schema validation."""

    def test_markdown_wrapped_json_parsed_cleanly(self, client_with_mock_groq, mock_groq):
        """JSON enclosed in ```json ... ``` markdown code fences is parsed without retry."""
        wrapped = (
            "```json\n"
            '{\n  "flashcards": [\n    {"question": "What is DNA?", "answer": "Deoxyribonucleic acid."}\n  ]\n}\n'
            "```"
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(wrapped)

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "DNA carries genetic information.", "count": 1},
        )

        assert response.status_code == 200
        assert response.json()["count"] == 1
        assert response.json()["flashcards"][0]["question"] == "What is DNA?"
        # Verified: only 1 Groq call was made (no retry needed)
        assert mock_groq.client.chat.completions.create.call_count == 1

    def test_conversational_text_around_json_parsed_cleanly(self, client_with_mock_groq, mock_groq):
        """JSON surrounded by conversational prose is extracted and parsed cleanly."""
        prose_wrapped = (
            "Here is the study material you requested:\n\n"
            '{"flashcards": [{"question": "What is RNA?", "answer": "Ribonucleic acid."}]}\n\n'
            "Hope this helps with your studies!"
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(prose_wrapped)

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "RNA plays an essential role in protein synthesis.", "count": 1},
        )

        assert response.status_code == 200
        assert response.json()["flashcards"][0]["answer"] == "Ribonucleic acid."

    def test_missing_fields_in_flashcard_rejected(self, client_with_mock_groq, mock_groq):
        """Flashcard missing required 'answer' field fails validation and returns 502 after retries."""
        bad_json = json.dumps({"flashcards": [{"question": "Only question here?"}]})
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(bad_json)

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Sample text", "count": 1},
        )

        assert response.status_code == 502
        assert "failed to generate valid flashcards" in response.json()["detail"].lower()

    def test_incorrect_data_types_rejected(self, client_with_mock_groq, mock_groq):
        """Options given as string instead of list fails validation."""
        bad_quiz = json.dumps(
            {
                "quiz": [
                    {
                        "question": "What is H2O?",
                        "options": "Water, Hydrogen, Oxygen, Carbon",  # String instead of list!
                        "correct_answer": "Water",
                        "explanation": "Chemical formula for water.",
                    }
                ]
            }
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(bad_quiz)

        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "Water is composed of hydrogen and oxygen.", "count": 1},
        )

        assert response.status_code == 502

    def test_empty_questions_or_answers_rejected(self, client_with_mock_groq, mock_groq):
        """Flashcard with empty or whitespace question is rejected."""
        empty_q = json.dumps(
            {"flashcards": [{"question": "   ", "answer": "Valid answer."}]}
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(empty_q)

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Sample text", "count": 1},
        )

        assert response.status_code == 502

    def test_invalid_quiz_options_count_rejected(self, client_with_mock_groq, mock_groq):
        """Quiz question with only 3 options (instead of exactly 4) fails validation."""
        three_options = json.dumps(
            {
                "quiz": [
                    {
                        "question": "What is the speed of light?",
                        "options": ["300,000 km/s", "150,000 km/s", "1,000 km/s"],  # Only 3!
                        "correct_answer": "300,000 km/s",
                        "explanation": "Light travels at approximately 300,000 km/s in a vacuum.",
                    }
                ]
            }
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(three_options)

        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "Light travels at 300,000 km/s in a vacuum.", "count": 1},
        )

        assert response.status_code == 502

    def test_duplicate_quiz_options_rejected(self, client_with_mock_groq, mock_groq):
        """Quiz question with duplicate choices fails validation."""
        duplicate_options = json.dumps(
            {
                "quiz": [
                    {
                        "question": "Which planet is closest to the Sun?",
                        "options": ["Mercury", "Mercury", "Venus", "Mars"],  # Duplicate!
                        "correct_answer": "Mercury",
                        "explanation": "Mercury is the closest planet to the Sun.",
                    }
                ]
            }
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(duplicate_options)

        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "Mercury orbits closest to the Sun.", "count": 1},
        )

        assert response.status_code == 502

    def test_quiz_model_valid_correct_answer_matches_option(self):
        """QuizQuestion instantiates cleanly when correct_answer matches one of the options."""
        question = QuizQuestion(
            question="What is the capital of France?",
            options=["London", "Berlin", "Paris", "Rome"],
            correct_answer="Paris",
            explanation="Paris is the capital of France.",
        )
        assert question.correct_answer == "Paris"
        assert question.correct_answer in question.options

    def test_quiz_model_invalid_correct_answer_raises_validation_error(self):
        """QuizQuestion raises Pydantic ValidationError when correct_answer is not in options."""
        with pytest.raises(ValidationError) as exc_info:
            QuizQuestion(
                question="What is the capital of France?",
                options=["London", "Berlin", "Madrid", "Rome"],
                correct_answer="Paris",  # Not in options
                explanation="Paris is the capital of France.",
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "must match one of the options" in errors[0]["msg"]
        assert errors[0]["type"] == "value_error"

    def test_correct_answer_not_in_options_rejected(self, client_with_mock_groq, mock_groq):
        """Quiz question whose correct_answer is not in options fails validation."""
        mismatched_answer = json.dumps(
            {
                "quiz": [
                    {
                        "question": "What is the capital of France?",
                        "options": ["London", "Berlin", "Madrid", "Rome"],
                        "correct_answer": "Paris",  # Not among options!
                        "explanation": "Paris is the capital of France.",
                    }
                ]
            }
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(mismatched_answer)

        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "Paris is the capital of France.", "count": 1},
        )

        assert response.status_code == 502
        assert "must match one of the options" in response.json()["detail"]

    def test_wrong_item_count_rejected(self, client_with_mock_groq, mock_groq):
        """AI returning fewer or more items than requested triggers retry and fails if uncorrected."""
        three_items = json.dumps(
            {
                "flashcards": [
                    {"question": "Q1", "answer": "A1"},
                    {"question": "Q2", "answer": "A2"},
                    {"question": "Q3", "answer": "A3"},
                ]
            }
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(three_items)

        # Request asked for 5 items, AI only returned 3
        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Study material here.", "count": 5},
        )

        assert response.status_code == 502
        assert "failed to generate valid flashcards" in response.json()["detail"].lower()


# ----------------------------------------------------------------------
# 3. Retry Handling Tests
# ----------------------------------------------------------------------


class TestRetryHandling:
    """Verifies bounded retry loop behavior with correction prompts."""

    def test_retry_succeeds_on_second_attempt(self, client_with_mock_groq, mock_groq):
        """When first attempt is malformed, correction prompt triggers valid second response (200 OK)."""
        invalid_first = "Invalid JSON {{{ not closing"
        valid_second = json.dumps(
            {
                "flashcards": [
                    {"question": "What is gravity?", "answer": "A fundamental force of attraction."}
                ]
            }
        )

        mock_groq.client.chat.completions.create.side_effect = [
            make_mock_completion(invalid_first),
            make_mock_completion(valid_second),
        ]

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Gravity attracts objects with mass.", "count": 1},
        )

        assert response.status_code == 200
        assert response.json()["count"] == 1
        assert response.json()["flashcards"][0]["question"] == "What is gravity?"
        # Verified: exactly 2 calls were made
        assert mock_groq.client.chat.completions.create.call_count == 2

    def test_retries_are_bounded_at_configured_maximum(self, client_with_mock_groq, mock_groq):
        """When all attempts fail, retries stop at 1 + AI_MAX_RETRIES without infinite loops."""
        invalid_output = "Always invalid non-json response."
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(invalid_output)

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Study material.", "count": 1},
        )

        assert response.status_code == 502
        # Default AI_MAX_RETRIES is 2, so total calls = 1 initial + 2 retries = 3 calls
        assert mock_groq.client.chat.completions.create.call_count == 3

    def test_correction_prompt_contains_specific_validation_error(self, mock_groq):
        """Correction prompt sent on retry details the exact schema violation."""
        invalid_quiz_first = json.dumps(
            {
                "quiz": [
                    {
                        "question": "What is boiling point of water?",
                        "options": ["100C", "50C", "0C"],  # 3 options!
                        "correct_answer": "100C",
                        "explanation": "Water boils at 100C.",
                    }
                ]
            }
        )
        valid_quiz_second = json.dumps(
            {
                "quiz": [
                    {
                        "question": "What is boiling point of water?",
                        "options": ["100C", "50C", "0C", "200C"],
                        "correct_answer": "100C",
                        "explanation": "Water boils at 100C at standard atmospheric pressure.",
                    }
                ]
            }
        )

        mock_groq.client.chat.completions.create.side_effect = [
            make_mock_completion(invalid_quiz_first),
            make_mock_completion(valid_quiz_second),
        ]

        service = GenerationService(mock_groq, Settings(AI_MAX_RETRIES=2))
        result = service.generate_quiz(text="Water boils at 100C.", count=1)

        assert result.count == 1
        assert len(result.quiz[0].options) == 4

        # Inspect the messages passed to Groq on the second call
        second_call_messages = mock_groq.client.chat.completions.create.call_args_list[1][1]["messages"]
        assert len(second_call_messages) == 4
        # Message 0: system, Message 1: user, Message 2: assistant (invalid), Message 3: user correction
        correction_msg = second_call_messages[3]["content"]
        assert "validation errors" in correction_msg.lower()
        assert "4 options" in correction_msg.lower()

    def test_quiz_retry_triggered_on_invalid_correct_answer(self, client_with_mock_groq, mock_groq):
        """When AI returns correct_answer not matching options on attempt 1, correction prompt leads to valid attempt 2 (200 OK)."""
        invalid_first = json.dumps(
            {
                "quiz": [
                    {
                        "question": "What is the capital of France?",
                        "options": ["London", "Berlin", "Madrid", "Rome"],
                        "correct_answer": "Paris",  # Not in options
                        "explanation": "Paris is the capital of France.",
                    }
                ]
            }
        )
        valid_second = json.dumps(
            {
                "quiz": [
                    {
                        "question": "What is the capital of France?",
                        "options": ["London", "Berlin", "Madrid", "Paris"],
                        "correct_answer": "Paris",  # Now in options
                        "explanation": "Paris is the capital of France.",
                    }
                ]
            }
        )

        mock_groq.client.chat.completions.create.side_effect = [
            make_mock_completion(invalid_first),
            make_mock_completion(valid_second),
        ]

        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "Paris is the capital of France.", "count": 1},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["quiz"][0]["correct_answer"] == "Paris"
        assert "Paris" in data["quiz"][0]["options"]
        assert mock_groq.client.chat.completions.create.call_count == 2

        # Verify the correction message in attempt 2 specified the error
        second_call_messages = mock_groq.client.chat.completions.create.call_args_list[1][1]["messages"]
        assert len(second_call_messages) == 4
        correction_msg = second_call_messages[3]["content"]
        assert "must match one of the options" in correction_msg

    def test_quiz_retries_exhausted_on_persistent_invalid_correct_answer(self, client_with_mock_groq, mock_groq):
        """When AI persistently returns mismatched correct_answer, service exhausts retries and raises AIGenerationValidationError (HTTP 502)."""
        invalid_output = json.dumps(
            {
                "quiz": [
                    {
                        "question": "What is the capital of France?",
                        "options": ["London", "Berlin", "Madrid", "Rome"],
                        "correct_answer": "Paris",  # Persistent mismatch
                        "explanation": "Paris is the capital of France.",
                    }
                ]
            }
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(invalid_output)

        # 1. Direct service call raises AIGenerationValidationError (never unhandled exception)
        service = GenerationService(mock_groq, Settings(AI_MAX_RETRIES=2))
        with pytest.raises(AIGenerationValidationError) as exc_info:
            service.generate_quiz(text="Paris is the capital of France.", count=1)
        assert "must match one of the options" in exc_info.value.message
        assert "after 3 attempts" in exc_info.value.message

        # 2. Router API call returns clean HTTP 502 (never 500 unhandled)
        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "Paris is the capital of France.", "count": 1},
        )
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert "failed to generate valid quiz questions" in detail.lower()
        assert "must match one of the options" in detail


# ----------------------------------------------------------------------
# 4. Provider Failures Tests
# ----------------------------------------------------------------------


class TestProviderFailures:
    """Verifies that provider exceptions are mapped to clean, sanitized client responses."""

    def test_groq_authentication_error_returns_502(self, client_with_mock_groq, mock_groq):
        """Authentication failure with Groq returns 502 Bad Gateway without leaking keys."""
        mock_resp = httpx.Response(status_code=401, request=httpx.Request("POST", "https://api.groq.com"))
        mock_groq.client.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API Key provided",
            response=mock_resp,
            body=None,
        )

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Some text", "count": 1},
        )

        assert response.status_code == 502
        detail = response.json()["detail"]
        assert "authentication failed" in detail.lower()
        assert "Invalid API Key provided" not in detail

    def test_groq_rate_limit_returns_429(self, client_with_mock_groq, mock_groq):
        """Rate limit exceeded returns HTTP 429 Too Many Requests."""
        mock_resp = httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.groq.com"))
        mock_groq.client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit reached for model",
            response=mock_resp,
            body=None,
        )

        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "Some text", "count": 1},
        )

        assert response.status_code == 429
        assert "rate limit exceeded" in response.json()["detail"].lower()

    def test_groq_timeout_returns_504(self, client_with_mock_groq, mock_groq):
        """API timeout returns HTTP 504 Gateway Timeout."""
        mock_req = MagicMock(spec=httpx.Request)
        mock_groq.client.chat.completions.create.side_effect = APITimeoutError(request=mock_req)

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Some text", "count": 1},
        )

        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()

    def test_groq_connection_error_returns_502(self, client_with_mock_groq, mock_groq):
        """Connection failure to Groq returns HTTP 502 Bad Gateway."""
        mock_req = MagicMock(spec=httpx.Request)
        mock_groq.client.chat.completions.create.side_effect = APIConnectionError(request=mock_req)

        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "Some text", "count": 1},
        )

        assert response.status_code == 502
        assert "communicating with the ai service" in response.json()["detail"].lower()

    def test_groq_api_status_error_returns_502(self, client_with_mock_groq, mock_groq):
        """Groq internal 500 error returns HTTP 502 Bad Gateway."""
        mock_resp = httpx.Response(status_code=500, request=httpx.Request("POST", "https://api.groq.com"))
        mock_groq.client.chat.completions.create.side_effect = APIStatusError(
            message="Internal Server Error from upstream",
            response=mock_resp,
            body=None,
        )

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Some text", "count": 1},
        )

        assert response.status_code == 502
        assert "communicating with the ai service" in response.json()["detail"].lower()

    def test_unconfigured_groq_key_returns_503(self):
        """Calling generation endpoints when GROQ_API_KEY is missing returns HTTP 503."""
        app = create_app(Settings(GROQ_API_KEY=None))
        with TestClient(app) as client:
            resp_flash = client.post("/api/generate/flashcards", json={"text": "Notes", "count": 1})
            assert resp_flash.status_code == 503
            assert "GROQ_API_KEY" in resp_flash.json()["detail"]

            resp_quiz = client.post("/api/generate/quiz", json={"text": "Notes", "count": 1})
            assert resp_quiz.status_code == 503
            assert "GROQ_API_KEY" in resp_quiz.json()["detail"]


# ----------------------------------------------------------------------
# 5. Input Validation Tests
# ----------------------------------------------------------------------


class TestInputValidation:
    """Verifies client input validation for empty input, counts, and size limits."""

    def test_empty_text_rejected(self, client_with_mock_groq):
        """Empty string returns HTTP 400."""
        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "", "count": 5},
        )
        assert response.status_code == 400
        assert "empty or contain only whitespace" in response.json()["detail"]

    def test_whitespace_text_rejected(self, client_with_mock_groq):
        """Whitespace-only text returns HTTP 400."""
        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "   \n\t  ", "count": 5},
        )
        assert response.status_code == 400
        assert "empty or contain only whitespace" in response.json()["detail"]

    def test_missing_text_field_rejected(self, client_with_mock_groq):
        """Missing 'text' field in JSON returns HTTP 422 Unprocessable Entity."""
        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"count": 5},
        )
        assert response.status_code == 422

    def test_invalid_count_below_minimum(self, client_with_mock_groq):
        """count < 1 returns HTTP 422."""
        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Valid notes here.", "count": 0},
        )
        assert response.status_code == 422

    def test_invalid_count_above_maximum_flashcards(self, client_with_mock_groq):
        """count > 30 for flashcards returns HTTP 422."""
        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Valid notes here.", "count": 31},
        )
        assert response.status_code == 422

    def test_invalid_count_above_maximum_quiz(self, client_with_mock_groq):
        """count > 20 for quiz returns HTTP 422."""
        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "Valid notes here.", "count": 21},
        )
        assert response.status_code == 422

    def test_invalid_count_type_rejected(self, client_with_mock_groq):
        """count as string returns HTTP 422."""
        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Valid notes here.", "count": "five"},
        )
        assert response.status_code == 422

    def test_oversized_text_rejected(self, mock_groq):
        """Input text exceeding MAX_GENERATION_INPUT_BYTES returns HTTP 413."""
        small_settings = Settings(GROQ_API_KEY="test-key", MAX_GENERATION_INPUT_BYTES=500)
        app = create_app(small_settings)
        app.dependency_overrides[get_groq_service] = lambda: mock_groq

        with TestClient(app) as client:
            large_text = "A" * 1000
            response = client.post(
                "/api/generate/flashcards",
                json={"text": large_text, "count": 2},
            )
            assert response.status_code == 413
            assert "exceeds the maximum limit" in response.json()["detail"]
            assert "500 bytes" in response.json()["detail"]


# ----------------------------------------------------------------------
# 6. Pre-Parsing Request Size Limits & Boundary Tests
# ----------------------------------------------------------------------


class TestPreParsingGenerationLimits:
    """Verifies RequestSizeLimitMiddleware and text input limit enforcement on generation routes."""

    @pytest.fixture
    def small_limit_client(self, mock_groq):
        """
        TestClient configured with:
        - MAX_GENERATION_INPUT_BYTES = 1000 (text UTF-8 byte limit enforced in router)
        - MAX_GENERATION_REQUEST_BODY_BYTES = None -> safely derived as (1000 * 6) + 32 KB = 38,768 bytes
        - MAX_UPLOAD_SIZE_BYTES = 10 MB (document ingestion limit)
        """
        settings = Settings(
            GROQ_API_KEY="test-key",
            MAX_GENERATION_INPUT_BYTES=1000,
            MAX_UPLOAD_SIZE_BYTES=10 * 1024 * 1024,
        )
        assert settings.MAX_GENERATION_REQUEST_BODY_BYTES == 38768
        app = create_app(settings)
        app.dependency_overrides[get_groq_service] = lambda: mock_groq
        with TestClient(app) as client:
            yield client
        app.dependency_overrides.clear()

    @staticmethod
    def _mock_flashcards():
        return json.dumps(
            {"flashcards": [{"question": "What is ATP?", "answer": "Cellular energy currency."}]}
        )

    @staticmethod
    def _mock_quiz():
        return json.dumps(
            {
                "quiz": [
                    {
                        "question": "What is ATP?",
                        "options": ["Energy currency", "DNA base", "Lipid", "Vitamin"],
                        "correct_answer": "Energy currency",
                        "explanation": "ATP provides energy for cellular processes.",
                    }
                ]
            }
        )

    def test_exact_boundary_text_field_accepted(self, small_limit_client, mock_groq):
        """Text field of exactly 1,000 UTF-8 bytes is accepted on both generation endpoints."""
        mock_groq.client.chat.completions.create.side_effect = [
            make_mock_completion(self._mock_flashcards()),
            make_mock_completion(self._mock_quiz()),
        ]

        text_exact = "A" * 1000
        assert len(text_exact.encode("utf-8")) == 1000

        # 1. Flashcards endpoint
        resp_flash = small_limit_client.post(
            "/api/generate/flashcards",
            json={"text": text_exact, "count": 1},
        )
        assert resp_flash.status_code == 200
        assert resp_flash.json()["count"] == 1

        # 2. Quiz endpoint
        resp_quiz = small_limit_client.post(
            "/api/generate/quiz",
            json={"text": text_exact, "count": 1},
        )
        assert resp_quiz.status_code == 200
        assert resp_quiz.json()["count"] == 1

    def test_one_byte_overflow_text_field_rejected(self, small_limit_client):
        """Text field of 1,001 UTF-8 bytes (1 byte above limit) returns HTTP 413 on both endpoints."""
        text_overflow = "A" * 1001
        assert len(text_overflow.encode("utf-8")) == 1001

        # 1. Flashcards endpoint
        resp_flash = small_limit_client.post(
            "/api/generate/flashcards",
            json={"text": text_overflow, "count": 1},
        )
        assert resp_flash.status_code == 413
        assert "exceeds the maximum limit of 1000 bytes" in resp_flash.json()["detail"]
        assert "1001 bytes" in resp_flash.json()["detail"]

        # 2. Quiz endpoint
        resp_quiz = small_limit_client.post(
            "/api/generate/quiz",
            json={"text": text_overflow, "count": 1},
        )
        assert resp_quiz.status_code == 413
        assert "exceeds the maximum limit of 1000 bytes" in resp_quiz.json()["detail"]
        assert "1001 bytes" in resp_quiz.json()["detail"]

    def test_non_ascii_utf8_exact_boundary_and_overflow(self, small_limit_client, mock_groq):
        """Non-ASCII UTF-8 text at exact 1,000 bytes is accepted; 1,001 bytes returns HTTP 413."""
        mock_groq.client.chat.completions.create.side_effect = [
            make_mock_completion(self._mock_flashcards()),
            make_mock_completion(self._mock_quiz()),
        ]

        # In UTF-8, Japanese 'あ' is 3 bytes. 333 * 3 = 999 bytes + 1 ASCII byte = 1000 bytes.
        exact_non_ascii = ("あ" * 333) + "X"
        assert len(exact_non_ascii.encode("utf-8")) == 1000

        resp_flash_ok = small_limit_client.post(
            "/api/generate/flashcards",
            json={"text": exact_non_ascii, "count": 1},
        )
        assert resp_flash_ok.status_code == 200

        resp_quiz_ok = small_limit_client.post(
            "/api/generate/quiz",
            json={"text": exact_non_ascii, "count": 1},
        )
        assert resp_quiz_ok.status_code == 200

        # 1 byte overflow: 333 * 3 + 2 = 1001 bytes
        overflow_non_ascii = ("あ" * 333) + "XX"
        assert len(overflow_non_ascii.encode("utf-8")) == 1001

        resp_flash_err = small_limit_client.post(
            "/api/generate/flashcards",
            json={"text": overflow_non_ascii, "count": 1},
        )
        assert resp_flash_err.status_code == 413
        assert "1001 bytes" in resp_flash_err.json()["detail"]

        resp_quiz_err = small_limit_client.post(
            "/api/generate/quiz",
            json={"text": overflow_non_ascii, "count": 1},
        )
        assert resp_quiz_err.status_code == 413
        assert "1001 bytes" in resp_quiz_err.json()["detail"]

    def test_control_characters_requiring_uXXXX_escaping_at_limit_accepted(
        self, small_limit_client, mock_groq
    ):
        """
        Text field of exactly 1,000 UTF-8 bytes containing control characters (e.g. \\x01)
        which JSON serializes into 6-character escape sequences (\\u0001).
        The serialized JSON body expands to > 6,000 bytes, but since the raw body limit
        accounts for worst-case 6x escaping (38,768 bytes), the request is accepted on both endpoints.
        """
        mock_groq.client.chat.completions.create.side_effect = [
            make_mock_completion(self._mock_flashcards()),
            make_mock_completion(self._mock_quiz()),
        ]

        # 1,000 SOH control characters: each \x01 escapes to '\u0001' (6 ASCII bytes) in JSON
        ctrl_text = "\x01" * 1000
        assert len(ctrl_text.encode("utf-8")) == 1000

        # Verify serialized JSON payload size
        serialized_payload = json.dumps({"text": ctrl_text, "count": 1}).encode("utf-8")
        assert b"\\u0001" in serialized_payload
        assert len(serialized_payload) >= 6000
        assert len(serialized_payload) < 38768  # within safe derived request body limit

        # 1. Flashcards endpoint accepts
        resp_flash = small_limit_client.post(
            "/api/generate/flashcards",
            content=serialized_payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp_flash.status_code == 200
        assert resp_flash.json()["count"] == 1

        # 2. Quiz endpoint accepts
        resp_quiz = small_limit_client.post(
            "/api/generate/quiz",
            content=serialized_payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp_quiz.status_code == 200
        assert resp_quiz.json()["count"] == 1

    def test_valid_maximum_size_text_field_serialized_with_ensure_ascii(
        self, small_limit_client, mock_groq
    ):
        """
        Valid maximum-size text field (1,000 UTF-8 bytes) containing non-ASCII characters,
        serialized with ensure_ascii=True (which encodes non-ASCII as \\uXXXX 6-byte sequences).
        The serialized body expands to > 3,000 bytes, which exceeds 2x input size but is accepted
        within the safe derived request body limit on both endpoints.
        """
        mock_groq.client.chat.completions.create.side_effect = [
            make_mock_completion(self._mock_flashcards()),
            make_mock_completion(self._mock_quiz()),
        ]

        # In UTF-8, 'é' is 2 bytes (0xC3 0xA9). 500 * 2 = 1,000 UTF-8 bytes.
        non_ascii_text = "é" * 500
        assert len(non_ascii_text.encode("utf-8")) == 1000

        # Serialized with ensure_ascii=True: each 'é' becomes '\u00e9' (6 bytes).
        serialized_payload = json.dumps(
            {"text": non_ascii_text, "count": 1},
            ensure_ascii=True,
        ).encode("utf-8")
        assert b"\\u00e9" in serialized_payload
        assert len(serialized_payload) >= 3000
        assert len(serialized_payload) < 38768  # within safe derived request body limit

        # 1. Flashcards endpoint accepts
        resp_flash = small_limit_client.post(
            "/api/generate/flashcards",
            content=serialized_payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp_flash.status_code == 200
        assert resp_flash.json()["count"] == 1

        # 2. Quiz endpoint accepts
        resp_quiz = small_limit_client.post(
            "/api/generate/quiz",
            content=serialized_payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp_quiz.status_code == 200
        assert resp_quiz.json()["count"] == 1

    def test_json_escaped_quotes_and_newlines_at_text_limit_accepted(
        self, small_limit_client, mock_groq
    ):
        """
        JSON-escaped characters (e.g. quotes) cause serialized JSON to be ~2,025 bytes (> 1,000 bytes),
        and are accepted on both endpoints.
        """
        mock_groq.client.chat.completions.create.side_effect = [
            make_mock_completion(self._mock_flashcards()),
            make_mock_completion(self._mock_quiz()),
        ]

        # 1000 double quotes: each quote escapes to '\"' in JSON (doubling the string length)
        escaped_text = '"' * 1000
        assert len(escaped_text.encode("utf-8")) == 1000

        # Verify serialized JSON body size
        serialized_payload = json.dumps({"text": escaped_text, "count": 1}).encode("utf-8")
        assert len(serialized_payload) > 2000
        assert len(serialized_payload) < 38768

        # 1. Flashcards endpoint accepts
        resp_flash = small_limit_client.post(
            "/api/generate/flashcards",
            content=serialized_payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp_flash.status_code == 200

        # 2. Quiz endpoint accepts
        resp_quiz = small_limit_client.post(
            "/api/generate/quiz",
            content=serialized_payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp_quiz.status_code == 200

    def test_excessively_large_body_rejected_before_parsing_with_content_length(
        self, small_limit_client
    ):
        """
        Oversized raw request body exceeding MAX_GENERATION_REQUEST_BODY_BYTES (38,768 bytes)
        with Content-Length header is rejected by middleware with HTTP 413 before JSON parsing.
        """
        malformed_oversized = b"NOT_VALID_JSON_BUT_EXCEEDS_38768_BYTES:" + (b"X" * 40000)
        assert len(malformed_oversized) > 38768

        # 1. Flashcards endpoint
        resp_flash = small_limit_client.post(
            "/api/generate/flashcards",
            content=malformed_oversized,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(malformed_oversized)),
            },
        )
        # HTTP 413 returned (not 422 JSON decode error), proving pre-parsing rejection
        assert resp_flash.status_code == 413
        assert "exceeds the maximum limit of 37.9 KB" in resp_flash.json()["detail"]

        # 2. Quiz endpoint
        resp_quiz = small_limit_client.post(
            "/api/generate/quiz",
            content=malformed_oversized,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(malformed_oversized)),
            },
        )
        assert resp_quiz.status_code == 413
        assert "exceeds the maximum limit of 37.9 KB" in resp_quiz.json()["detail"]

    def test_excessively_large_streaming_body_rejected_without_content_length(
        self, small_limit_client
    ):
        """
        Streaming request without Content-Length exceeding MAX_GENERATION_REQUEST_BODY_BYTES
        is aborted in real time by middleware with HTTP 413 on both endpoints.
        """
        def stream_chunks():
            yield b'{"text": "'
            yield b"A" * 20000
            yield b"B" * 20000
            yield b'", "count": 1}'

        # 1. Flashcards endpoint
        resp_flash = small_limit_client.post(
            "/api/generate/flashcards",
            content=stream_chunks(),
            headers={"Content-Type": "application/json"},
        )
        assert resp_flash.status_code == 413
        assert "exceeds the maximum limit of 37.9 KB" in resp_flash.json()["detail"]

        # 2. Quiz endpoint
        resp_quiz = small_limit_client.post(
            "/api/generate/quiz",
            content=stream_chunks(),
            headers={"Content-Type": "application/json"},
        )
        assert resp_quiz.status_code == 413
        assert "exceeds the maximum limit of 37.9 KB" in resp_quiz.json()["detail"]

    def test_document_ingestion_uses_own_limit(self, small_limit_client):
        """Document ingestion endpoint uses MAX_UPLOAD_SIZE_BYTES (10 MB), unaffected by generation limits."""
        # 50,000 bytes document exceeds generation limits (1,000 and 38,768), but is well under 10 MB
        larger_doc = "Photosynthesis notes. " * 2300  # ~50,600 bytes
        assert len(larger_doc.encode("utf-8")) > 38768

        resp = small_limit_client.post(
            "/api/documents/ingest",
            json={"text": larger_doc},
        )
        assert resp.status_code == 200
        assert resp.json()["source_type"] == "pasted_text"

    def test_unrelated_routes_unaffected(self, small_limit_client):
        """Unrelated routes pass through without generation size limits."""
        health_resp = small_limit_client.get("/api/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"


# ----------------------------------------------------------------------
# 7. Groq Invocation Parameter Verification
# ----------------------------------------------------------------------


class TestGroqCallParameters:
    """Verifies model name and response_format parameters passed to the Groq API."""

    def test_groq_called_with_configured_model_and_json_format_flashcards(
        self, client_with_mock_groq, mock_groq
    ):
        """Flashcards call specifies configured model name and json_object response_format."""
        raw_json = json.dumps(
            {"flashcards": [{"question": "Q1?", "answer": "A1."}]}
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(raw_json)

        response = client_with_mock_groq.post(
            "/api/generate/flashcards",
            json={"text": "Photosynthesis is the process green plants use.", "count": 1},
        )
        assert response.status_code == 200

        mock_groq.client.chat.completions.create.assert_called_once()
        call_kwargs = mock_groq.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.1-8b-instant"
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["temperature"] == 0.3

    def test_groq_called_with_configured_model_and_json_format_quiz(
        self, client_with_mock_groq, mock_groq
    ):
        """Quiz call specifies configured model name and json_object response_format."""
        raw_json = json.dumps(
            {
                "quiz": [
                    {
                        "question": "What is the powerhouse of the cell?",
                        "options": ["Nucleus", "Mitochondria", "Ribosome", "Chloroplast"],
                        "correct_answer": "Mitochondria",
                        "explanation": "Mitochondria generate ATP.",
                    }
                ]
            }
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(raw_json)

        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "Mitochondria produce ATP energy.", "count": 1},
        )
        assert response.status_code == 200

        mock_groq.client.chat.completions.create.assert_called_once()
        call_kwargs = mock_groq.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.1-8b-instant"
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["temperature"] == 0.3


# ----------------------------------------------------------------------
# 8. Quiz Response Schema Verification
# ----------------------------------------------------------------------


class TestQuizSchemaNoQuestionsField:
    """Verifies that QuizGenerateResponse returns solely 'quiz' and 'count' without 'questions'."""

    def test_quiz_response_excludes_questions_computed_field(
        self, client_with_mock_groq, mock_groq
    ):
        """Quiz API response must only contain 'quiz' and 'count', not 'questions'."""
        raw_json = json.dumps(
            {
                "quiz": [
                    {
                        "question": "What is DNA?",
                        "options": ["Protein", "Genetic code", "Carbohydrate", "Lipid"],
                        "correct_answer": "Genetic code",
                        "explanation": "DNA contains hereditary genetic instructions.",
                    }
                ]
            }
        )
        mock_groq.client.chat.completions.create.return_value = make_mock_completion(raw_json)

        response = client_with_mock_groq.post(
            "/api/generate/quiz",
            json={"text": "DNA stores hereditary genetic instructions.", "count": 1},
        )
        assert response.status_code == 200
        data = response.json()

        assert "quiz" in data
        assert "count" in data
        assert "questions" not in data
        assert "questions" not in response.text
        assert set(data.keys()) == {"quiz", "count"}
