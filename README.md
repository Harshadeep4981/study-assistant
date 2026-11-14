# AI Study Assistant

An AI-powered study tool that generates interactive flashcards and quizzes from study materials. Users can paste notes, enter topics, or upload documents (TXT / text-based PDF) and receive structured study content powered by the Groq AI API.

> **Status**: Milestone 1 (Backend Foundation), Milestone 2 (Document Ingestion), and Milestone 3 (AI Generation Pipeline) implemented. Frontend is under development.

---

## Features

- **Text & topic input** — paste study notes or enter a topic via JSON or form submission.
- **File upload** — upload `.txt` and text-based / mixed `.pdf` files.
- **Normalization pipeline** — standardize line endings (LF), clean horizontal whitespace, and preserve paragraph breaks without excessive vertical gaps.
- **Safe PDF extraction** — PyMuPDF-based page-by-page extraction, distinguishing text-based, image-only, and mixed PDFs.
- **AI flashcards** — generate concept-focused study flashcards (`question`, `answer`) with configurable counts.
- **AI quizzes** — generate multiple-choice quizzes with 4 distinct options, unambiguous correct answers, and explanations.
- **Resilient AI parsing & retries** — robust JSON extraction (markdown fence stripping) and bounded retries with correction prompts for schema violations.
- **Retest mode** — *(Planned)* retry only incorrectly answered questions.

---

## Architecture

```
┌─────────────┐        HTTP        ┌──────────────────┐       SDK       ┌──────────┐
│   React     │  ◄──────────────►  │  FastAPI Backend  │  ◄───────────►  │  Groq AI │
│   Frontend  │    JSON / CORS     │  (Python)         │   Structured    │  API     │
└─────────────┘                    └──────────────────┘   responses      └──────────┘
```

**Key decisions:**
- **Hierarchical summarization** for large documents rather than RAG, embeddings, or vector databases.
- **Fixed output budgets** and validated AI responses.
- **Decoupled Groq startup**: backend starts without an API key; future AI generation endpoints will use the `get_groq_service` dependency to return HTTP 503 if unconfigured.
- **Text-based PDFs only**: image-only PDFs are rejected with HTTP 422 explaining that OCR is out of scope. Mixed PDFs extract available text and surface a warning that text embedded in images may be missed.
- **Security & resource bounds**: streaming uploads enforce `MAX_UPLOAD_SIZE_BYTES` in memory without writing temporary files to disk; file signatures and encodings are strictly validated.

---

## Tech Stack

| Layer    | Technology                                   |
|----------|----------------------------------------------|
| Frontend | React *(Planned)*                            |
| Backend  | Python 3.11+, FastAPI, Uvicorn               |
| AI       | Groq Python SDK (`qwen/qwen3.8-27b` default, configurable via `GROQ_MODEL`) |
| PDF      | PyMuPDF / fitz (page-by-page text extraction)|
| Uploads  | python-multipart                             |
| Config   | Pydantic Settings, python-dotenv             |
| Testing  | pytest, httpx                                |

---

## Getting Started

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/keys) (optional for startup; required for downstream AI features)

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

Copy the example file and add your configuration:

```bash
cp .env.example .env
```

Edit `backend/.env`:

```env
# Optional for startup, required for AI endpoints
GROQ_API_KEY=your_groq_api_key_here

# Optional (defaults shown)
GROQ_MODEL=qwen/qwen3.8-27b
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]
MAX_UPLOAD_SIZE_BYTES=10485760
MAX_PDF_PAGES=100
MAX_GENERATION_INPUT_BYTES=102400
MAX_LARGE_DOCUMENT_BYTES=512000
# Optional: Raw request body limit for generation endpoints.
# When unset (recommended), automatically derived as (6 * MAX_LARGE_DOCUMENT_BYTES + 32 KB).
# Only uncomment and set if deliberately overriding the derived default. Any explicit value
# must be >= (6 * MAX_LARGE_DOCUMENT_BYTES + 32 KB) to guarantee safe JSON serialization.
# MAX_GENERATION_REQUEST_BODY_BYTES=3104768
AI_MAX_RETRIES=2
AI_MAX_REQUEST_CALLS=120
AI_REQUEST_TIMEOUT_SECONDS=300.0
AI_MAX_SUMMARIZATION_OPERATIONS=100
HOST=0.0.0.0
PORT=8000
```

| Variable                            | Required | Default                                              | Description                                  |
|-------------------------------------|----------|------------------------------------------------------|----------------------------------------------|
| `GROQ_API_KEY`                      | No*      | `None`                                               | Your Groq API key                            |
| `GROQ_MODEL`                        | No       | `qwen/qwen3.8-27b`                                   | Groq model for AI generation                 |
| `CORS_ORIGINS`                      | No       | `["http://localhost:3000", "http://localhost:5173"]`  | Allowed frontend origins (JSON list)         |
| `MAX_UPLOAD_SIZE_BYTES`             | No       | `10485760` (10 MB)                                   | Maximum document file upload & text size limit in bytes |
| `MAX_GENERATION_INPUT_BYTES`        | No       | `102400` (100 KB)                                    | Maximum AI generation direct study text UTF-8 byte limit (small-document fast path) |
| `MAX_LARGE_DOCUMENT_BYTES`          | No       | `512000` (500 KB)                                    | Maximum supported study material size for large-document processing |
| `MAX_GENERATION_REQUEST_BODY_BYTES` | No       | Derived: `(6 * MAX_LARGE_DOCUMENT_BYTES) + 32 KB` (default: 3,104,768 bytes / ~3.03 MB) | Raw JSON request body limit for generation endpoints. Automatically derived from `MAX_LARGE_DOCUMENT_BYTES` to ensure safe JSON escaping (`\u00XX` worst-case expansion for 500 KB Unicode inputs) without premature rejection. Any explicit override must be $\ge$ safe derived minimum. |
| `MAX_PDF_PAGES`                     | No       | `100`                                                | Maximum allowed PDF page count               |
| `AI_MAX_RETRIES`                    | No       | `2`                                                  | Maximum AI generation correction retry attempts |
| `AI_MAX_REQUEST_CALLS`              | No       | `120`                                                | Maximum total Groq API calls per user request across chunking, consolidation, and generation. Sized to accommodate 500 KB actual algorithm (69 chunk ops + 17 L2 + 5 L3 + 1 final synthesis = 92 ops + 1 generation = 93 base calls, leaving 27 retries within 120). |
| `AI_REQUEST_TIMEOUT_SECONDS`        | No       | `300.0` (5 min)                                      | Maximum wall-clock seconds for a complete request before early budget termination. Sized for 500 KB sequential inference. |
| `AI_MAX_SUMMARIZATION_OPERATIONS`   | No       | `100`                                                | Maximum summarization and consolidation operations allowed per request. Reconciled with `MAX_LARGE_DOCUMENT_BYTES` (500 KB actual algorithm requires 92 operations, $\le 100$). |
| `HOST`                              | No       | `0.0.0.0`                                            | Server bind address                          |
| `PORT`                              | No       | `8000`                                               | Server port                                  |

> \* The backend starts without `GROQ_API_KEY`. The health check (`GET /api/health`) reports `groq_client: "not_configured"`. AI generation endpoints (`POST /api/generate/flashcards` and `POST /api/generate/quiz`) inject the `get_groq_service` dependency and return HTTP 503 Service Unavailable if the Groq API key is unconfigured.

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

All 246 automated tests run in seconds without external network dependencies.

---

## Document Ingestion & Text Extraction

### Supported Input Formats & Content Types

The endpoint accepts two content types:
1. `application/json`: accepts study notes in a JSON object with a required `text` field (`{"text": "..."}`).
2. `multipart/form-data`: accepts file uploads (`.txt`, `.pdf`) via the `file` field, or form-encoded text via the `text` field.

> **Validation Constraint**: Requests providing both `file` and `text` simultaneously are rejected with **HTTP 400 Bad Request** to prevent ambiguous input handling.

### Upload Limits, Security & Storage Lifecycle

- **Pre-Parsing Request-Body Size Enforcement**: Enforced at the ASGI middleware layer (`RequestSizeLimitMiddleware`) before JSON deserialization or multipart form parsing:
  - `POST /api/documents/ingest`: Enforces `MAX_UPLOAD_SIZE_BYTES` (default: 10 MB).
  - `POST /api/generate/flashcards`: Enforces `MAX_GENERATION_REQUEST_BODY_BYTES` (derived automatically as `6 * MAX_LARGE_DOCUMENT_BYTES + 32 KB`; default: 3,104,768 bytes / ~3.03 MB). When left unset (recommended), it automatically adapts whenever `MAX_LARGE_DOCUMENT_BYTES` is adjusted without requiring manual synchronization. Safely accommodates worst-case JSON serialization expansion where 1-byte control characters or non-ASCII characters with `ensure_ascii=True` expand into 6-byte escape sequences (`\u00XX`), plus a 32 KB envelope buffer. For example, a 500 KB document with Unicode or escaped characters serialized to JSON will not be prematurely rejected by the middleware.
  - `POST /api/generate/quiz`: Enforces `MAX_GENERATION_REQUEST_BODY_BYTES` (same automatically derived limit).
  - If a `Content-Length` header is present and exceeds the route limit, the request is rejected immediately with **HTTP 413 Payload Too Large** before reading or parsing any body bytes.
  - If `Content-Length` is missing or misleading (such as in chunked transfer encoding), incoming ASGI body chunks are counted in real-time as they stream in; reading terminates and aborts with **HTTP 413** the moment accumulated bytes exceed the limit. This prevents unbounded memory allocation for JSON and stops oversized uploads before they are parsed or spooled to temporary storage.
  - Unrelated routes (such as `GET /api/health`) pass through without size-limit interception.
- **Two-Tier Generation Input Processing & Large-Document Architecture**:
  - **Tier 1 (Direct Generation Path)**: Inputs up to `MAX_GENERATION_INPUT_BYTES` (default: 100 KB = 102,400 bytes) proceed directly to flashcard or quiz prompt generation without summarization overhead.
  - **Tier 2 (Large-Document Hierarchical Summarization Path)**: Inputs between `MAX_GENERATION_INPUT_BYTES` (100 KB) and `MAX_LARGE_DOCUMENT_BYTES` (default: 500 KB = 512,000 bytes) are automatically processed by the `SummarizationService`:
    1. *Structure-Aware Chunking with Guaranteed Bounds*: Splits text into cohesive chunks of up to 8,000 characters with 500-character overlaps. Preferred split boundaries follow document structure: paragraph breaks (`\n\n+`), line breaks (`\n`), abbreviation-aware sentence terminators (protecting common patterns like `e.g.`, `i.e.`, `Fig. 1`, `Dr. Watson`, `et al.`), and word breaks. If an unspaced token exceeds `max_chunk_size` (e.g. 15,000 chars without whitespace), a deterministic character-slicing fallback ensures **no chunk exceeds `max_chunk_size` under any circumstances**. Overlap is preserved across adjacent chunks even when individual paragraphs exceed `chunk_overlap`.
    2. *Map Phase (Section Summaries)*: Distills key educational concepts, definitions, formulas, and relationships from each chunk using the configured Groq model.
    3. *Intermediate Summary Validation*: Validates that AI-generated summaries contain meaningful educational content (minimum 20 characters and 10 alphanumeric characters), rejecting trivial placeholders (`.`, `None`, `N/A`, or punctuation). Malformed outputs trigger bounded correction retries (`AI_MAX_RETRIES`).
    4. *Reduce Phase (Hierarchical Synthesis)*: If leaf summaries exceed the downstream generation budget, bounded multi-level consolidation synthesizes section summaries into a unified, high-density study guide within budget (`MAX_HIERARCHICAL_DEPTH = 3`).
    5. *Downstream Generation*: The synthesized study guide is fed directly into flashcard or quiz generation.
  - **Rate Limit & Error Propagation**:
    - Groq `RateLimitError` (HTTP 429) extracts `Retry-After` response headers if provided by the provider and raises `AIRateLimitError`, returning HTTP 429 with the `Retry-After` response header to the client.
    - Groq provider status errors (HTTP 500/502/503) and timeouts map cleanly to HTTP 502 (`AIProviderError`) and HTTP 504 (`AITimeoutError`) with complete exception chaining (`from exc`).
  - **Synchronous Processing Performance & Architectural Boundary**:
    - *Processing Duration Estimates*:
      - 110 KB document: ~15 chunks $\rightarrow$ 15 Map calls, 1 Gen call $\approx 16$–19 seconds.
      - 250 KB document: ~34 chunks $\rightarrow$ 34 Map calls, Level 2 bundles $\approx 40$–50 seconds.
      - 500 KB document: ~68 chunks $\rightarrow$ 68 Map calls, Level 2/3 bundles $\approx 90$–100 seconds.
    - *Recommended Client Timeout*: Configure HTTP client timeouts to at least **180 seconds** when requesting generation on large documents (> 100 KB).
    - *Architectural Boundary*: The current synchronous architecture processes large documents safely up to the 500 KB limit within standard HTTP timeouts. For production scale involving documents larger than 500 KB or concurrent multi-user workloads, synchronous HTTP requests become untenable; an asynchronous worker / background job queue (e.g. Celery / ARQ with Redis) with job status polling or WebSocket progress updates is required.
  - **Capacity Limit Enforcement**: Inputs exceeding `MAX_LARGE_DOCUMENT_BYTES` (default: 500 KB) are cleanly rejected with **HTTP 413 Payload Too Large** before any AI calls are made.
  - **Prompt-Injection Defenses**: Source material in large-document processing is framed with strict security boundaries (`--- UNTRUSTED STUDY MATERIAL SECTION ---`). System prompts explicitly instruct the model to treat source text purely as passive data, never execute commands, and never leak system prompts.
- **Secondary Stream & Text Safeguards**: In addition to the ASGI layer, `DocumentService.read_upload_file_safely` streams files in bounded chunks of at most the remaining allowance plus one byte, and `ingest_pasted_text` verifies string byte length as defense-in-depth safeguards.
- **Page Count Limit**: PDFs are checked against `MAX_PDF_PAGES` (default: 100 pages, must be positive). PDFs exceeding this limit are rejected before page extraction with **HTTP 413 Payload Too Large**.
- **Storage Lifecycle & Framework Temporary Storage**:
  - No permanent document storage is implemented.
  - Multipart uploads may use framework-managed temporary storage (such as `SpooledTemporaryFile` used by Starlette when upload sizes exceed memory buffer thresholds).
  - All uploaded resources and file handles are closed and cleaned up immediately after processing (`UploadFile.close()`, `fitz.Document.close()`).
- **Content Authoritativeness**: File content signatures are authoritative. Files with unexpected client MIME types (e.g. `application/octet-stream`) are accepted if their content matches the expected format. Disguised files (e.g. PDF bytes named `.txt` or plain text named `.pdf`) are rejected with **HTTP 400 Bad Request**.

### PDF Handling & Structural Classification

PyMuPDF inspects each PDF page by page to distinguish four document classes:

1. **Text-based PDFs**: Extractable text detected, no embedded images. Extracted text is normalized and returned with **HTTP 200 OK**.
2. **Mixed PDFs (Text + Images)**: Extractable text and embedded raster images detected. Text is extracted and returned with **HTTP 200 OK**, `metadata.has_images = true`, and a warning:
   > *"This document contains embedded images. Text embedded inside images cannot be extracted because OCR is not supported."*
3. **Image-only / Scanned PDFs**: No extractable text detected, but embedded raster images are present. Rejected with **HTTP 422 Unprocessable Entity**:
   > *"The PDF contains no extractable text. Scanned or image-only PDFs are not supported because OCR is out of scope. Please upload a text-based document or paste text directly."*
4. **No-Extractable-Text PDFs**: Neither extractable text nor images detected (e.g. blank pages or vector-only drawings). Rejected with **HTTP 422 Unprocessable Entity**:
   > *"The PDF document contains no extractable text or images. Please ensure the document contains readable text or paste your study notes directly."*
- **Encrypted / Password-Protected PDFs**: Rejected with **HTTP 400 Bad Request**.
- **Corrupted / Invalid PDFs**: Rejected with **HTTP 400 Bad Request** using a client-safe error message without leaking internal parser details.

### Known Limitations

- **OCR is intentionally out of scope**: Scanned physical documents, photos of textbook pages, and text flattened inside raster images cannot be recognized.
- **Encrypted PDFs**: Password-protected PDFs must be unlocked before upload.
- **File Types**: Only `.txt` and `.pdf` files are supported. Formats such as `.docx`, `.pptx`, `.epub`, or `.rtf` are rejected with HTTP 400.

---

## API Endpoints

| Method | Path                    | Description                          | Status      |
|--------|-------------------------|--------------------------------------|-------------|
| GET    | `/api/health`           | Health check and Groq client status  | Implemented |
| POST   | `/api/documents/ingest` | Ingest pasted text or TXT/PDF file   | Implemented |
| GET    | `/docs`                 | Interactive Swagger UI               | Automatic   |
| GET    | `/redoc`                | Alternative API documentation        | Automatic   |

### Request & Response Examples

#### 1. Ingest Pasted Text (JSON)

```bash
curl -X POST "http://localhost:8000/api/documents/ingest" \
     -H "Content-Type: application/json" \
     -d '{"text": "Photosynthesis is the process by which green plants make food.\r\n\r\nIt requires sunlight and chlorophyll."}'
```

**Response (HTTP 200):**
```json
{
  "extracted_text": "Photosynthesis is the process by which green plants make food.\n\nIt requires sunlight and chlorophyll.",
  "source_type": "pasted_text",
  "metadata": {
    "filename": null,
    "file_size_bytes": 105,
    "page_count": null,
    "character_count": 101,
    "word_count": 15,
    "has_images": false
  },
  "warnings": []
}
```

#### 2. Ingest TXT File (`multipart/form-data`)

```bash
curl -X POST "http://localhost:8000/api/documents/ingest" \
     -F "file=@notes.txt"
```

**Response (HTTP 200):**
```json
{
  "extracted_text": "Chapter 1: The Solar System\n\nThe Sun is the star at the center.",
  "source_type": "txt",
  "metadata": {
    "filename": "notes.txt",
    "file_size_bytes": 62,
    "page_count": null,
    "character_count": 62,
    "word_count": 12,
    "has_images": false
  },
  "warnings": []
}
```

#### 3. Ingest PDF Document (`multipart/form-data`)

```bash
curl -X POST "http://localhost:8000/api/documents/ingest" \
     -F "file=@lecture.pdf"
```

**Response (HTTP 200 for Mixed PDF):**
```json
{
  "extracted_text": "Enzymes are biological catalysts composed of proteins.\n\nThey accelerate chemical reactions in cells.",
  "source_type": "pdf",
  "metadata": {
    "filename": "lecture.pdf",
    "file_size_bytes": 32568,
    "page_count": 2,
    "character_count": 96,
    "word_count": 14,
    "has_images": true
  },
  "warnings": [
    "This document contains embedded images. Text embedded inside images cannot be extracted because OCR is not supported."
  ]
}
```

---

### AI Generation Endpoints

#### 4. Generate Flashcards (`POST /api/generate/flashcards`)

Accepts study text and generates concept-focused flashcards.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/generate/flashcards" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Photosynthesis is the biochemical process by which photoautotrophs convert light energy into chemical energy stored in glucose. The light-dependent reactions take place within the thylakoid membranes of chloroplasts, splitting water to release oxygen and generating ATP and NADPH. The Calvin cycle occurs in the stroma, using ATP and NADPH to fix carbon dioxide into carbohydrates.",
       "count": 2
     }'
```

**Response (HTTP 200):**
```json
{
  "flashcards": [
    {
      "question": "What primary energy transformation occurs during photosynthesis?",
      "answer": "Light energy is converted into chemical energy stored in glucose molecules."
    },
    {
      "question": "Where do the light-dependent reactions take place, and what are their products?",
      "answer": "They occur in the thylakoid membranes of chloroplasts, producing oxygen, ATP, and NADPH."
    }
  ],
  "count": 2
}
```

#### 5. Generate Multiple-Choice Quiz (`POST /api/generate/quiz`)

Accepts study text and generates multiple-choice questions with 4 distinct options, unambiguous correct answers, and explanations.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/generate/quiz" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Mitochondria are double-membraned organelles that generate most of the chemical energy needed to power the cell'\''s biochemical reactions. Chemical energy produced by the mitochondria is stored in a small molecule called adenosine triphosphate (ATP). The inner membrane contains folds called cristae, which expand the surface area for oxidative phosphorylation.",
       "count": 1
     }'
```

**Response (HTTP 200):**
```json
{
  "quiz": [
    {
      "question": "What molecule stores the chemical energy produced by cellular respiration in mitochondria?",
      "options": [
        "Adenosine triphosphate (ATP)",
        "Ribonucleic acid (RNA)",
        "Deoxyribonucleic acid (DNA)",
        "Nicotinamide adenine dinucleotide (NADH)"
      ],
      "correct_answer": "Adenosine triphosphate (ATP)",
      "explanation": "Chemical energy produced by the mitochondria is stored in adenosine triphosphate (ATP)."
    }
  ],
  "count": 1
}
```

---

## AI Generation Architecture & Resilience

The generation pipeline ensures reliable, structured study content through a multi-stage process:

1. **Prompt Engineering**: System prompts enforce JSON-only output with strict schemas, grounding the model exclusively in the provided text without inventing facts.
2. **Robust JSON Extraction**: Strips markdown code fences (```` ```json ````) and cleans conversational prose to locate the root JSON object.
3. **Strict Pydantic Validation**:
   - **Flashcards**: Validates non-empty `question` and `answer` fields, ensuring target item counts match the request.
   - **Quizzes**: Validates exactly 4 distinct options, non-empty fields, and ensures `correct_answer` matches one of the 4 options verbatim.
4. **Bounded Retries with Correction Prompts**: If output is unparseable or violates schema constraints, the service initiates a correction prompt detailing the specific errors. Retries are bounded by `AI_MAX_RETRIES` (default: 2 retries, 3 total attempts) to prevent infinite loops.
5. **Sanitized Error Mapping**: Provider errors (authentication, rate limits, timeouts, connection drops) are translated into clean HTTP errors (502, 429, 504) without exposing credentials or internal tracebacks.
6. **Prompt-Injection Resistance**: Both system prompts explicitly instruct the LLM that user-supplied study material is untrusted reference data. The model is strictly instructed to disregard any instructions, commands, or role overrides embedded within the study material, never reveal or quote its system prompts under any circumstances, and treat input purely as passive factual subject matter for generating educational content.
7. **Pre-Parsing Request-Body Limit & Post-Parsing Text Limit**: Generation endpoints enforce `MAX_GENERATION_REQUEST_BODY_BYTES` (default: 632 KB) at the ASGI middleware layer before JSON deserialization, rejecting oversized requests early. This conservatively accommodates worst-case JSON escaping (`\u00XX`, 6x expansion + 32 KB envelope). After JSON deserialization, the router strictly enforces `MAX_GENERATION_INPUT_BYTES` (default: 100 KB) on the UTF-8 byte length of the actual `text` field, ensuring valid text at the limit is never rejected merely due to JSON serialization or character escaping expansion while keeping the domain text size strictly bounded.
8. **Request-Level Processing Budgets & Safe Rate Limiting**: Implements a holistic `ProcessingBudget` tracking API attempts, wall-clock duration, and operations across chunking, summarization, consolidation, and generation.
   - **Reconciled Hard Limits**: `AI_MAX_REQUEST_CALLS` (default: 120), `AI_REQUEST_TIMEOUT_SECONDS` (default: 300.0s = 5 min), and `AI_MAX_SUMMARIZATION_OPERATIONS` (default: 100). Sized to match the actual `summarize_large_document()` algorithm for 500 KB documents ($\le 69$ chunks $\to 69$ L1 + 17 L2 + 5 L3 + 1 final synthesis = 92 operations, $\le 100$; base calls: 93, leaving 27 retry attempts within 120). If any limit is exceeded, processing terminates cleanly with HTTP 429 (`AIBudgetExceededError`) or HTTP 504 (`AITimeoutError`).
   - **Provider Timeout Semantics**: The Groq SDK / HTTPX `timeout` sets per-operation socket timeouts (connect, read, write, pool) rather than a cumulative wall-clock guarantee. Request-level deadlines are strictly guaranteed by verifying `budget.check_deadline()` immediately before provider calls (aborting without calling if $< 0.1$s remains) and immediately after provider calls (halting all processing if the deadline elapsed during the call).
   - **Safe Rate Limit Handling**: Inspects Groq `RateLimitError` and headers for `Retry-After`. Blind retries are strictly prohibited. If `Retry-After` is present, $\le 2.0$ seconds, and remaining budget duration allows, a single bounded backoff retry is attempted; otherwise the endpoint fails fast immediately with HTTP 429. Any subsequent failures during retry (timeouts, connection drops, status errors, or second rate limits) are consistently mapped to domain exceptions (`AITimeoutError`, `AIProviderError`, `AIRateLimitError`, or preserved `AIBudgetExceededError`).
   - **Hierarchical Consolidation Guarantees**: Sequential execution (concurrency = 1) avoids concurrency rate spikes; intermediate summaries are capped at $\le 12{,}000$ characters, and synthesized output is strictly bounded to $\le \text{MAX\_GENERATION\_INPUT\_BYTES}$ with depth limit $\le 3$.

---

## Project Structure

```
Study_Assistant/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings (env vars, @lru_cache)
│   │   ├── main.py                # App factory, CORS, exception handlers
│   │   ├── middleware.py          # ASGI RequestSizeLimitMiddleware (early body limits)
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── document.py        # Document ingestion request & response models
│   │   │   └── generation.py      # Flashcard and Quiz request & response models
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py       # POST /api/documents/ingest router
│   │   │   └── generate.py        # POST /api/generate/flashcards & /quiz router
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── exceptions.py      # Domain exceptions with HTTP status codes
│   │       ├── text_normalizer.py # Whitespace & paragraph normalization
│   │       ├── pdf_service.py     # PyMuPDF text & image inspection
│   │       ├── document_service.py# Ingestion coordinator & safe file streaming
│   │       ├── processing_budget.py # Request-level call, duration & operation budget tracker
│   │       ├── summarization_service.py # Semantic chunking & hierarchical summarization
│   │       ├── generation_service.py # AI prompts, parsing, validation & retry loop
│   │       └── groq_service.py    # Groq client + get_groq_service dependency
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py            # Shared fixtures with environment isolation
│   │   ├── test_config.py         # Configuration & env-variable tests
│   │   ├── test_cors.py           # CORS middleware tests
│   │   ├── test_groq_service.py   # Groq service & dependency tests
│   │   ├── test_health.py         # Health endpoint & startup tests
│   │   ├── test_text_normalizer.py# Normalization unit tests
│   │   ├── test_pdf_service.py    # PyMuPDF extraction unit tests
│   │   ├── test_document_ingest.py# Ingestion API endpoint integration tests
│   │   ├── test_generation.py     # AI generation, parsing, retries & error mapping tests
│   │   └── test_large_document_processing.py # Chunking, hierarchical summarization & security tests
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
- [x] `get_groq_service` dependency implemented (returns HTTP 503 when unconfigured)
- [x] CORS configured from `Settings.CORS_ORIGINS` (default: localhost:3000, localhost:5173)
- [x] Health check endpoint (`GET /api/health`) with accurate Groq status
- [x] `.gitignore` with rules for secrets, venv, and build artifacts
- [x] Document processing pipeline for pasted text, TXT uploads, and PDF documents
- [x] Text normalization preserving paragraph boundaries and line endings
- [x] PyMuPDF extraction distinguishing text-based, mixed, image-only, and no-extractable-text PDFs
- [x] Image-only PDF rejection with OCR limitation explanation (HTTP 422)
- [x] No-extractable-text PDF rejection (HTTP 422)
- [x] Mixed PDF text extraction with embedded image warning
- [x] Resource limits: configurable max upload size (`MAX_UPLOAD_SIZE_BYTES`), configurable PDF page limit (`MAX_PDF_PAGES`), bounded chunk reading (limit + 1 byte)
- [x] Storage lifecycle: no permanent storage, framework temporary storage closed immediately after processing
- [x] Request validation: rejects ambiguous requests providing both `file` and `text`, enforces size limits on JSON text
- [x] Pre-parsing body enforcement: ASGI middleware scoped to POST /api/documents/ingest stops oversized JSON and multipart uploads before full body buffering or disk spooling
- [x] Unified ingestion endpoint (`POST /api/documents/ingest`) supporting JSON and multipart/form-data
- [x] Flashcard generation endpoint (`POST /api/generate/flashcards`) with concept questions and answers
- [x] Multiple-choice quiz generation endpoint (`POST /api/generate/quiz`) with 4 choices, single correct answer, and explanations
- [x] Strict AI response validation and bounded correction retries (`AI_MAX_RETRIES`)
- [x] Provider error sanitization (auth 502, rate limit 429, timeout 504, connection 502)
- [x] Default model configuration verified against Groq catalog (`qwen/qwen3.8-27b`) with `.env` override
- [x] Semantic text chunking respecting structural boundaries (paragraphs, sentences, words) with overlap
- [x] Hierarchical summarization (Map-Reduce) for large documents (> 100 KB up to 500 KB) with bounded depth (`MAX_HIERARCHICAL_DEPTH = 3`)
- [x] Automated test suite (194 passing tests across 9 test modules)
- [ ] React frontend — Planned (Milestone 4)
- [ ] Retest incorrect answers — Planned (Milestone 5)

---

## Known Limitations

- **OCR is out of scope**: Scanned or image-only PDFs containing no selectable text are rejected with HTTP 422.
- **Large-Document Processing Ceiling**: Input documents exceeding `MAX_LARGE_DOCUMENT_BYTES` (default: 500 KB) are rejected with HTTP 413. Documents up to 500 KB undergo hierarchical chunking and summarization.
- **Provider Concurrency & Rate Limits**: On free-tier Groq API keys, rapid bursts of requests or processing many chunks sequentially may encounter provider-level rate limits (HTTP 429).
- **Deterministic AI output**: While temperature is set low (0.2–0.3) for factual consistency, AI outputs may vary across calls for identical text. Bounded retries guarantee schema adherence.

---

## License

This project was created as a frontend internship assignment.
