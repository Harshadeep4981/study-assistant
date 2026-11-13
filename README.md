# AI Study Assistant

An AI-powered study tool that generates interactive flashcards and quizzes from study materials. Users can paste notes, enter topics, or upload documents (TXT / text-based PDF) and receive structured study content powered by the Groq AI API.

> **Status**: Milestone 1 (Backend Foundation) and Milestone 2 (Document Ingestion & Text Extraction) implemented. AI generation and frontend are under development.

---

## Features

- **Text & topic input** — paste study notes or enter a topic via JSON or form submission.
- **File upload** — upload `.txt` and text-based / mixed `.pdf` files.
- **Normalization pipeline** — standardize line endings (LF), clean horizontal whitespace, and preserve paragraph breaks without excessive vertical gaps.
- **Safe PDF extraction** — PyMuPDF-based page-by-page extraction, distinguishing text-based, image-only, and mixed PDFs.
- **AI flashcards** — *(Planned)* generate interactive flip-cards from study material.
- **AI quizzes** — *(Planned)* generate multiple-choice quizzes with scoring.
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
| AI       | Groq Python SDK (LLaMA 3.1 8B Instant)      |
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
GROQ_MODEL=llama-3.1-8b-instant
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]
MAX_UPLOAD_SIZE_BYTES=10485760
MAX_PDF_PAGES=100
HOST=0.0.0.0
PORT=8000
```

| Variable                | Required | Default                                              | Description                                  |
|-------------------------|----------|------------------------------------------------------|----------------------------------------------|
| `GROQ_API_KEY`          | No*      | `None`                                               | Your Groq API key                            |
| `GROQ_MODEL`            | No       | `llama-3.1-8b-instant`                               | Groq model for AI generation                 |
| `CORS_ORIGINS`          | No       | `["http://localhost:3000", "http://localhost:5173"]`  | Allowed frontend origins (JSON list)         |
| `MAX_UPLOAD_SIZE_BYTES` | No       | `10485760` (10 MB)                                   | Maximum file upload & text size limit in bytes|
| `MAX_PDF_PAGES`         | No       | `100`                                                | Maximum allowed PDF page count               |
| `HOST`                  | No       | `0.0.0.0`                                            | Server bind address                          |
| `PORT`                  | No       | `8000`                                               | Server port                                  |

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

All 92 automated tests run in seconds without external network dependencies.

---

## Document Ingestion & Text Extraction

### Supported Input Formats & Content Types

The endpoint accepts two content types:
1. `application/json`: accepts study notes in a JSON object with a required `text` field (`{"text": "..."}`).
2. `multipart/form-data`: accepts file uploads (`.txt`, `.pdf`) via the `file` field, or form-encoded text via the `text` field.

> **Validation Constraint**: Requests providing both `file` and `text` simultaneously are rejected with **HTTP 400 Bad Request** to prevent ambiguous input handling.

### Upload Limits, Security & Storage Lifecycle

- **Pre-Parsing Request-Body Size Enforcement**: Enforced at the ASGI middleware layer (`RequestSizeLimitMiddleware`) before JSON deserialization or multipart form parsing.
  - If a `Content-Length` header is present and exceeds `MAX_UPLOAD_SIZE_BYTES`, the request is rejected immediately with **HTTP 413 Payload Too Large** before reading any body bytes.
  - If `Content-Length` is missing or misleading (such as in chunked transfer encoding), incoming ASGI body chunks are counted in real-time as they stream in; reading terminates and aborts with **HTTP 413** the moment accumulated bytes exceed the limit. This prevents unbounded memory allocation for JSON and stops oversized multipart uploads before they are parsed or spooled to temporary storage.
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
│   │   │   └── document.py        # Document ingestion request & response models
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   └── documents.py       # POST /api/documents/ingest router
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── exceptions.py      # Domain exceptions with HTTP status codes
│   │       ├── text_normalizer.py # Whitespace & paragraph normalization
│   │       ├── pdf_service.py     # PyMuPDF text & image inspection
│   │       ├── document_service.py# Ingestion coordinator & safe file streaming
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
│   │   └── test_document_ingest.py# Ingestion API endpoint integration tests
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
- [x] Automated test suite (97 passing tests across 7 test modules)
- [ ] Hierarchical summarization — Planned (Milestone 3)
- [ ] Flashcard generation endpoint — Planned (Milestone 4)
- [ ] Quiz generation endpoint — Planned (Milestone 5)
- [ ] React frontend — Planned (Milestone 6)
- [ ] Retest incorrect answers — Planned (Milestone 7)

---

## License

This project was created as a frontend internship assignment.
