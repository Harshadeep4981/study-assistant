# AI Study Assistant

An AI-powered study tool that generates interactive flashcards and quizzes from study materials. Users can paste notes, enter topics, or upload documents (TXT / text-based PDF) and receive structured study content powered by the Groq AI API.

> **Status**: Backend foundation implemented. Frontend and AI generation features are under development.

---

## Features (Planned)

- **Text & topic input** — paste study notes or enter a topic.
- **File upload** — upload `.txt` and text-based `.pdf` files.
- **AI flashcards** — generate interactive flip-cards from study material.
- **AI quizzes** — generate multiple-choice quizzes with scoring.
- **Retest mode** — retry only incorrectly answered questions.

## Architecture

```
┌─────────────┐        HTTP        ┌──────────────────┐       SDK       ┌──────────┐
│   React     │  ◄──────────────►  │  FastAPI Backend  │  ◄───────────►  │  Groq AI │
│   Frontend  │    JSON / CORS     │  (Python)         │   Structured    │  API     │
└─────────────┘                    └──────────────────┘   responses      └──────────┘
```

**Key decisions:**
- Hierarchical summarization for large documents (no RAG / vector DB).
- Fixed output budgets with AI response validation.
- Groq API key is never exposed to the frontend.
- Text-based PDFs only; image-only PDFs are rejected with a clear message.
- Decoupled startup: backend starts without an API key; future AI endpoints will depend on `get_groq_service` to return HTTP 503 if the key is not configured.

## Tech Stack

| Layer    | Technology                                   |
|----------|----------------------------------------------|
| Frontend | React                                        |
| Backend  | Python, FastAPI, Uvicorn                     |
| AI       | Groq Python SDK (LLaMA 3.1 8B Instant)      |
| PDF      | PyMuPDF (text extraction)                    |
| Config   | Pydantic Settings, python-dotenv             |
| Testing  | pytest, httpx                                |

---

## Getting Started

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/keys) (optional for startup; required for AI features)

### Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (includes pytest)
pip install -r requirements-dev.txt
```

### Environment Variables

Copy the example file and add your API key:

```bash
cp .env.example .env
```

Edit `backend/.env`:

```env
# Required for AI features (optional for startup)
GROQ_API_KEY=your_groq_api_key_here

# Optional (defaults shown)
GROQ_MODEL=llama-3.1-8b-instant
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]
HOST=0.0.0.0
PORT=8000
```

| Variable       | Required | Default                                              | Description                          |
|----------------|----------|------------------------------------------------------|--------------------------------------|
| `GROQ_API_KEY` | No*      | `None`                                               | Your Groq API key                    |
| `GROQ_MODEL`   | No       | `llama-3.1-8b-instant`                               | Groq model for AI generation         |
| `CORS_ORIGINS` | No       | `["http://localhost:3000", "http://localhost:5173"]`  | Allowed frontend origins (JSON list) |
| `HOST`         | No       | `0.0.0.0`                                            | Server bind address                  |
| `PORT`         | No       | `8000`                                               | Server port                          |

> \* The backend starts without `GROQ_API_KEY`. Groq client initialization and the `get_groq_service` dependency are implemented. Flashcard and quiz generation endpoints are planned for future milestones and are not yet implemented; once created, those endpoints will inject `get_groq_service` to return HTTP 503 if the Groq API key is unconfigured.

> **Security**: The `.env` file is listed in `.gitignore` and must never be committed.

### Running the Backend

```bash
cd backend
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### Running Tests

```bash
cd backend
pytest tests/ -v
```

### API Endpoints

| Method | Path           | Description                          | Status      |
|--------|----------------|--------------------------------------|-------------|
| GET    | `/api/health`  | Health check and Groq client status  | Implemented |
| GET    | `/docs`        | Interactive Swagger UI               | Automatic   |
| GET    | `/redoc`       | Alternative API documentation        | Automatic   |

> **Planned AI Endpoints**: Generation endpoints (such as `POST /api/flashcards` and `POST /api/quiz`) are planned for upcoming milestones and are currently not implemented. They will rely on the `get_groq_service` FastAPI dependency to ensure requests return an HTTP 503 error if the server is running without a configured `GROQ_API_KEY`.

#### Health Endpoint Response

The health endpoint distinguishes between backend availability and Groq configuration:

```json
// Without GROQ_API_KEY configured
{ "status": "healthy", "groq_client": "not_configured" }

// With GROQ_API_KEY configured
{ "status": "healthy", "groq_client": "initialized" }
```

The endpoint does not make network calls to Groq; `"initialized"` means the SDK client object has been created with the provided key.

---

## Project Structure

```
Study_Assistant/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings (env vars, @lru_cache)
│   │   ├── main.py                # App factory, CORS, routes
│   │   └── services/
│   │       ├── __init__.py
│   │       └── groq_service.py    # Groq client + get_groq_service dependency
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py            # Shared fixtures with environment isolation
│   │   ├── test_config.py         # Configuration & env-variable tests
│   │   ├── test_cors.py           # CORS middleware tests
│   │   ├── test_groq_service.py   # Groq service & dependency tests
│   │   └── test_health.py         # Health endpoint & startup tests
│   ├── .env.example               # Environment variable template
│   ├── pyproject.toml             # pytest configuration
│   ├── requirements.txt           # Production dependencies
│   └── requirements-dev.txt       # Development dependencies (pytest, httpx)
├── .gitignore
└── README.md
```

---

## Implementation Status

- [x] FastAPI application factory with lifespan management
- [x] Pydantic Settings with `@lru_cache` and reliable `.env` path
- [x] Optional `GROQ_API_KEY` — backend starts without it
- [x] Groq client initialization as a service (only when key is present)
- [x] `get_groq_service` dependency implemented (ready for future AI endpoints to return HTTP 503 when unconfigured)
- [x] CORS configured from `Settings.CORS_ORIGINS` (default: localhost:3000, localhost:5173)
- [x] Health check endpoint (`GET /api/health`) with accurate Groq status
- [x] `.gitignore` with rules for secrets, venv, and build artifacts
- [x] Automated test suite with environment isolation
- [ ] Document processing (TXT and PDF upload) — Planned
- [ ] Hierarchical summarization — Planned
- [ ] Flashcard generation endpoint — Planned
- [ ] Quiz generation endpoint — Planned
- [ ] React frontend — Planned
- [ ] Retest incorrect answers — Planned

---

## License

This project was created as a frontend internship assignment.
