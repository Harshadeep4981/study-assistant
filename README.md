# Studywise

Studywise turns pasted notes or uploaded TXT/PDF documents into interactive flashcards and multiple-choice quizzes. The React frontend calls a FastAPI backend for document extraction and AI generation. AI requests use a server-side Groq key; the key is never sent to the browser.

## Technology stack

- Frontend: React 18, Vite 5, Tailwind CSS 3, Vitest, React Testing Library, and Lucide icons.
- Backend: Python 3.11+, FastAPI, Pydantic Settings, PyMuPDF, and the Groq Python SDK.
- Tests: Vitest/jsdom for the frontend; pytest/httpx for the backend.

## Features

- Paste study notes or a topic, or upload one `.txt` or `.pdf` file.
- Extract and normalize text in the backend; review the source and processing notices in a Study Workspace.
- Generate five flashcards on demand, flip and navigate cards, mark them **Know it** or **Need Practice**, and revisit practice cards.
- Generate a five-question quiz on demand, answer one question at a time, see explanations, review mistakes, and retry incorrect questions.
- Keep one active Study Set and all learning progress in React state while the page is open.

## Architecture

```text
Browser (React + Vite)
  | JSON / multipart HTTP; CORS
  v
FastAPI
  | text extraction and normalization
  | validated generation requests
  v
Groq API (server-side key)
```

The browser first sends material to `POST /api/documents/ingest` and keeps the backend-returned `extracted_text` in the active Study Set. Flashcards and Quiz are generated only when selected in the Workspace; generated responses are validated and held in that same in-memory state. Navigation between the Workspace and learning screens does not repeat successful generation requests. Starting a new Study Set clears the prior set. A page refresh clears the active set and progress; there is no database, browser storage, account system, or study history.

## Project structure

```text
frontend/
  src/
    api/                  Fetch client and ingestion/generation helpers
    components/           Shared layout and brand illustration
    features/             Document input, study workspace, flashcards, quiz/review
    hooks/                Scroll presentation hook
    state/                Active Study Set and learning state
    styles/               Global theme and responsive styles
    test/                 Frontend interaction/API contract tests
  index.html
  package.json            Frontend scripts and dependencies
  vite.config.js
  tailwind.config.js
  postcss.config.js
  .env.example
backend/
  app/
    routers/               Ingestion and generation HTTP endpoints
    models/                Request/response schemas
    services/              Extraction, normalization, summarization, cache, Groq/resilience
    config.py              Environment-backed settings
  tests/                   Backend unit and integration tests
README.md
```

Frontend state is held in the `useStudyComposer` React hook. The UI submits only the active source to ingestion and uses the normalized extracted text for later generation requests.

## Requirements

- Node.js 20 or newer and npm.
- Python 3.11 or newer.
- A Groq API key for live flashcard and quiz generation. The backend can start and ingest documents without one, but generation returns HTTP 503 until configured.

## Setup

### Backend

From the repository root:

```powershell
cd backend
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Edit `backend/.env` and set `GROQ_API_KEY` for generation. Start the server:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API health check is at `http://localhost:8000/api/health`; Swagger UI is at `http://localhost:8000/docs`.

### Frontend

In a second terminal:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Open the Vite URL printed by the command (normally `http://localhost:5173`). The frontend defaults to `http://localhost:8000` for the API. Set `VITE_API_BASE_URL` in the `frontend/.env.local` to override it. The backend `CORS_ORIGINS` must include the frontend origin.

## Environment variables

`frontend/.env.example` contains the frontend setting:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `http://localhost:8000` | FastAPI base URL used by Vite |

Backend settings are read from `backend/.env`; see `backend/.env.example` for defaults and resource limits:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | unset | Server-only key required for AI generation |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Model selected by the backend |
| `CORS_ORIGINS` | localhost:3000 and localhost:5173 | Allowed browser origins (JSON list) |
| `MAX_UPLOAD_SIZE_BYTES` | 10 MiB | Ingestion request/file size limit |
| `MAX_PDF_PAGES` | 100 | Configurable PDF page cap (maximum allowed: 100 pages) |
| `MAX_GENERATION_INPUT_BYTES` | 100 KiB | Direct generation input threshold |
| `MAX_LARGE_DOCUMENT_BYTES` | 500 KiB | Maximum supported extracted material size |
| `MAX_GENERATION_REQUEST_BODY_BYTES` | 3,104,768 bytes (2.96 MiB) | Safe generation JSON body limit; derived as `(6 × MAX_LARGE_DOCUMENT_BYTES) + 32 KiB` when unset |
| `AI_MAX_RETRIES` | 2 | Bounded retries for invalid AI output |
| `AI_MAX_REQUEST_CALLS` | 120 | Per-request AI call budget |
| `AI_REQUEST_TIMEOUT_SECONDS` | 300 | Per-request processing time budget |
| `AI_MAX_SUMMARIZATION_OPERATIONS` | 100 | Per-request summarization budget |
| `AI_PROVIDER_RATE_LIMIT_MAX_RETRIES` | 5 | Maximum extra calls for explicitly transient provider 429s |
| `AI_PROVIDER_TRANSIENT_MAX_RETRIES` | 2 | Maximum extra calls for timeouts, connection errors, and transient 5xx/408 responses |
| `AI_PROVIDER_BACKOFF_INITIAL_SECONDS` | 1.0 | Initial fallback rate-limit backoff; it doubles up to the configured maximum |
| `AI_PROVIDER_BACKOFF_MULTIPLIER` | 2.0 | Exponential backoff multiplier |
| `AI_PROVIDER_BACKOFF_MAX_SECONDS` | 16.0 | Maximum fallback backoff; 20% jitter is applied within the cap |
| `AI_PROVIDER_TRANSIENT_BACKOFF_INITIAL_SECONDS` | 0.5 | Initial delay for transient network/server failures |
| `AI_PROVIDER_TRANSIENT_BACKOFF_MAX_SECONDS` | 2.0 | Maximum delay for transient network/server failures |
| `AI_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS` | 0.25 | Minimum interval between provider request starts in this process |
| `AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | 3 | Exhausted temporary rate-limit calls before opening the breaker |
| `AI_CIRCUIT_BREAKER_RECOVERY_SECONDS` | 30 | Open-circuit recovery delay before a single probe is allowed |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | Backend bind settings |

Provider retries are bounded and separate from `AI_MAX_RETRIES`, which corrects malformed model output. `Retry-After` and supported Groq reset headers take precedence over fallback backoff. Quota/billing exhaustion and invalid credentials are not retried. Each actual provider attempt consumes the existing per-request call budget and all waits are bounded by the existing request deadline. The rate limiter and circuit breaker are in-memory and process-local; deployments with multiple Uvicorn workers or backend instances need a shared distributed limiter to coordinate provider traffic globally.

Do not put `GROQ_API_KEY` in a `VITE_` variable or frontend environment file. `.env` and local environment variants are ignored by Git; the example files are intended to be committed.

## API endpoints

| Method and path | Request | Success response |
| --- | --- | --- |
| `GET /api/health` | None | Health and Groq configuration status |
| `POST /api/documents/ingest` | JSON `{ "text": "..." }`, or multipart field `file` with TXT/PDF | `{ extracted_text, source_type, metadata, warnings }` |
| `POST /api/generate/flashcards` | JSON `{ "text": "...", "count": 5 }` (`count`: 1-30) | `{ flashcards: [{ question, answer }], count }` |
| `POST /api/generate/quiz` | JSON `{ "text": "...", "count": 5 }` (`count`: 1-20) | `{ quiz: [{ question, options, correct_answer, explanation }], count }`; exactly four distinct options per question |

The ingestion response `source_type` is `pasted_text`, `txt`, or `pdf`. For quiz questions, `correct_answer` exactly matches one of the four `options`. Generation uses the extracted text returned by ingestion, not the original browser input.

## Document handling and limitations

- Only `.txt` and `.pdf` uploads are supported. The backend validates signatures/content; an unusual browser MIME type alone does not invalidate a supported extension.
- PDF ingestion is limited to 100 pages (a lower `MAX_PDF_PAGES` value can be configured; values above 100 are rejected). A PDF over the configured limit returns HTTP 413 before page text extraction. TXT uploads use byte-size limits and are not subject to page-count rules.
- Text-based PDFs within the page limit are extracted page by page. Mixed text/image PDFs return extractable text with a warning. Image-only/scanned PDFs return HTTP 422 because OCR is not implemented.
- Encrypted, corrupted, empty, oversized, or unsupported files return client-safe errors. The UI does not parse PDFs in the browser.
- Documents are not stored permanently. Multipart handling may use framework-managed temporary spooling; uploaded file handles are closed after ingestion.
- Extracted inputs above the direct generation threshold can use synchronous hierarchical summarization before generation and can take longer than small requests. The backend enforces document, request-call, operation, and time budgets.
- Summaries are cached in a bounded, process-local in-memory cache and reused by flashcard and quiz generation when eligible. Cache entries are not shared across multiple backend processes and disappear on restart.
- Generated answers depend on the source quality and AI provider output. The frontend validates response structure and does not substitute fabricated cards or questions on failures.
- State is in memory for one active Study Set only. Refreshing or closing the page clears the Study Set and progress.
- No authentication, multi-user data isolation, database, persistence, OCR, or study history is implemented.

## Testing and build

From the repository root:

```bash
cd frontend
npm test
npm run build
cd ..
git diff --check
```

Backend tests (with the backend virtual environment active):

```bash
cd backend
pytest tests/ -v
```

The frontend test suite uses Vitest, jsdom, and React Testing Library. Backend tests use pytest and httpx; AI-dependent tests use test doubles rather than sending provider requests.

These commands run the complete frontend and backend test suites. The targeted checks performed during the latest project verification were not a substitute for running both complete suites.

## Docker

Dockerfiles and a Compose configuration are included for running the frontend and backend together. Docker Compose builds a static production frontend image and the FastAPI backend image.

1. Install Docker Desktop (or Docker Engine with the Compose plugin).
2. From the repository root, optionally create a `.env` file containing `GROQ_API_KEY=your_groq_api_key_here`. Ingestion and health checks work without the key; flashcard and quiz generation require it.
3. Start both services:

   ```bash
   docker compose up --build
   ```

4. Open `http://localhost:8080`. The backend health endpoint is `http://localhost:8000/api/health`, and Swagger UI is `http://localhost:8000/docs`.

The frontend image receives `VITE_API_BASE_URL` at build time and defaults to `http://localhost:8000`, which is the API address reachable by the user's browser. If the browser accesses the app from another host, set `VITE_API_BASE_URL` in the root `.env` to an API URL reachable from that browser, then rebuild with `docker compose up --build`. Compose allows the frontend origin `http://localhost:8080` through backend CORS. Uploaded documents and study progress are not persisted by the containers; the application has no database or permanent document storage.

Stop the services with `Ctrl+C`, or run `docker compose down` in a second terminal.
