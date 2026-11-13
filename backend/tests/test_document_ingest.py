"""
Tests for document ingestion API endpoint (POST /api/documents/ingest).

Covers all required scenarios:
1. Valid text input (JSON)
2. Empty text input
3. Valid TXT upload
4. Invalid text encoding
5. Valid text-based PDF
6. Image-only PDF
7. Mixed PDF
8. Corrupted PDF
9. Encrypted PDF
10. Oversized upload
11. Unsupported file type
12. Normalization behavior
"""

import io
import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def helper_make_text_pdf(text: str = "This is a text-based test PDF.") -> bytes:
    """Create a minimal in-memory text-based PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def helper_make_image_only_pdf() -> bytes:
    """Create an in-memory PDF containing an image but zero extractable text."""
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50), 1)
    pix.clear_with(200)
    page.insert_image(fitz.Rect(50, 50, 100, 100), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


def helper_make_mixed_pdf(text: str = "Text in mixed PDF.") -> bytes:
    """Create an in-memory PDF containing both text and an embedded image."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50), 1)
    pix.clear_with(200)
    page.insert_image(fitz.Rect(50, 100, 100, 150), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


def helper_make_blank_pdf(num_pages: int = 1) -> bytes:
    """Create an in-memory PDF with blank pages (no text, no images)."""
    doc = fitz.open()
    for _ in range(num_pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def helper_make_multipage_pdf(num_pages: int = 2, text_per_page: str = "Page content") -> bytes:
    """Create a multi-page in-memory PDF."""
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((50, 72), f"{text_per_page} {i + 1}")
    data = doc.tobytes()
    doc.close()
    return data


def helper_make_encrypted_pdf(text: str = "Confidential", password: str = "secret123") -> bytes:
    """Create a password-encrypted in-memory PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    data = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw=password,
        owner_pw="owner_pass_123",
    )
    doc.close()
    return data


class TestPastedTextInput:
    """Tests for text input via JSON and multipart form data."""

    def test_valid_text_input_json(self, client_no_key):
        """Pasted text submitted via JSON produces normalized text and metadata."""
        payload = {"text": "Photosynthesis is the process by which green plants make food."}
        response = client_no_key.post("/api/documents/ingest", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["source_type"] == "pasted_text"
        assert "Photosynthesis" in data["extracted_text"]
        assert data["metadata"]["filename"] is None
        assert data["metadata"]["page_count"] is None
        assert data["metadata"]["character_count"] == len(data["extracted_text"])
        assert data["metadata"]["word_count"] > 0
        assert data["metadata"]["has_images"] is False
        assert data["warnings"] == []

    def test_valid_text_input_form(self, client_no_key):
        """Pasted text submitted via multipart/form-data also succeeds."""
        response = client_no_key.post(
            "/api/documents/ingest",
            data={"text": "Newton's first law of motion states that an object remains at rest..."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source_type"] == "pasted_text"
        assert "Newton's first law" in data["extracted_text"]

    def test_empty_text_rejected(self, client_no_key):
        """Empty string text input returns HTTP 400."""
        response = client_no_key.post("/api/documents/ingest", json={"text": ""})
        assert response.status_code == 400
        assert "empty or contain only whitespace" in response.json()["detail"]

    def test_whitespace_only_text_rejected(self, client_no_key):
        """Whitespace-only text input returns HTTP 400."""
        response = client_no_key.post("/api/documents/ingest", json={"text": "   \n\t  \r\n  "})
        assert response.status_code == 400
        assert "empty or contain only whitespace" in response.json()["detail"]

    def test_missing_text_field_rejected(self, client_no_key):
        """JSON body missing the 'text' key returns HTTP 400."""
        response = client_no_key.post("/api/documents/ingest", json={"notes": "My notes"})
        assert response.status_code == 400
        assert "text" in response.json()["detail"]

    def test_invalid_json_rejected(self, client_no_key):
        """Malformed JSON payload returns HTTP 400."""
        response = client_no_key.post(
            "/api/documents/ingest",
            content=b"{invalid json content",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]


class TestTXTUpload:
    """Tests for TXT file uploads."""

    def test_valid_txt_upload(self, client_no_key):
        """Valid UTF-8 TXT file is uploaded and processed successfully."""
        txt_content = b"Chapter 1: The Solar System\n\nThe Sun is the star at the center."
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("notes.txt", txt_content, "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source_type"] == "txt"
        assert "Chapter 1: The Solar System" in data["extracted_text"]
        assert data["metadata"]["filename"] == "notes.txt"
        assert data["metadata"]["file_size_bytes"] == len(txt_content)
        assert data["metadata"]["page_count"] is None
        assert data["metadata"]["has_images"] is False
        assert data["warnings"] == []

    def test_empty_txt_upload_rejected(self, client_no_key):
        """Uploading an empty TXT file returns HTTP 400."""
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_invalid_encoding_rejected(self, client_no_key):
        """Uploading a TXT file with invalid UTF-8 bytes returns HTTP 400."""
        # \xff\xfe with invalid UTF-8 byte sequences
        invalid_bytes = b"Some valid ascii \x80\x81\xfe\xff followed by invalid"
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("corrupted.txt", invalid_bytes, "text/plain")},
        )
        assert response.status_code == 400
        detail = response.json()["detail"].lower()
        assert "encoding" in detail or "decoded" in detail

    def test_binary_txt_with_null_bytes_rejected(self, client_no_key):
        """Uploading a binary file masked as .txt (containing null bytes) returns HTTP 400."""
        binary_txt = b"Text header\x00\x00\x01\x02binary payload"
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("binary.txt", binary_txt, "text/plain")},
        )
        assert response.status_code == 400
        assert "binary" in response.json()["detail"].lower()


class TestPDFUpload:
    """Tests for PDF document uploads using PyMuPDF."""

    def test_valid_text_based_pdf(self, client_no_key):
        """Text-based PDF is processed and returns page count and text."""
        pdf_bytes = helper_make_text_pdf("Quantum mechanics is a fundamental theory in physics.")
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("physics.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source_type"] == "pdf"
        assert "Quantum mechanics" in data["extracted_text"]
        assert data["metadata"]["filename"] == "physics.pdf"
        assert data["metadata"]["page_count"] == 1
        assert data["metadata"]["has_images"] is False
        assert data["warnings"] == []

    def test_image_only_pdf_rejected_with_ocr_message(self, client_no_key):
        """Image-only PDF is rejected with HTTP 422 and clear OCR explanation."""
        pdf_bytes = helper_make_image_only_pdf()
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("scanned.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "no extractable text" in detail
        assert "OCR is out of scope" in detail

    def test_mixed_pdf_extracts_text_with_warning(self, client_no_key):
        """Mixed PDF extracts text, sets has_images=True, and includes OCR warning."""
        pdf_bytes = helper_make_mixed_pdf("Enzymes are biological catalysts composed of proteins.")
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("biology.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source_type"] == "pdf"
        assert "Enzymes are biological catalysts" in data["extracted_text"]
        assert data["metadata"]["has_images"] is True
        assert len(data["warnings"]) == 1
        assert "embedded images" in data["warnings"][0]
        assert "OCR is not supported" in data["warnings"][0]

    def test_corrupted_pdf_rejected(self, client_no_key):
        """Corrupted PDF returns HTTP 400."""
        corrupted = b"%PDF-1.4\ncorrupted garbage ending prematurely"
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("broken.pdf", corrupted, "application/pdf")},
        )
        assert response.status_code == 400
        assert "corrupted" in response.json()["detail"].lower()

    def test_encrypted_pdf_rejected(self, client_no_key):
        """Encrypted PDF returns HTTP 400."""
        encrypted_bytes = helper_make_encrypted_pdf()
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("protected.pdf", encrypted_bytes, "application/pdf")},
        )
        assert response.status_code == 400
        assert "encrypted or password-protected" in response.json()["detail"]

    def test_no_extractable_text_pdf_rejected(self, client_no_key):
        """PDF with neither text nor images is rejected with HTTP 422."""
        blank_pdf = helper_make_blank_pdf(num_pages=2)
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("blank.pdf", blank_pdf, "application/pdf")},
        )
        assert response.status_code == 422
        assert "no extractable text or images" in response.json()["detail"]

    def test_pdf_page_limit_exact_boundary_accepted(self):
        """PDF whose page count matches MAX_PDF_PAGES is accepted."""
        custom_settings = Settings(MAX_PDF_PAGES=3)
        app = create_app(custom_settings)
        with TestClient(app) as client:
            pdf_bytes = helper_make_multipage_pdf(num_pages=3)
            response = client.post(
                "/api/documents/ingest",
                files={"file": ("boundary.pdf", pdf_bytes, "application/pdf")},
            )
            assert response.status_code == 200
            assert response.json()["metadata"]["page_count"] == 3

    def test_pdf_page_limit_exceeded_rejected(self):
        """PDF exceeding MAX_PDF_PAGES is rejected early with HTTP 413."""
        custom_settings = Settings(MAX_PDF_PAGES=3)
        app = create_app(custom_settings)
        with TestClient(app) as client:
            pdf_bytes = helper_make_multipage_pdf(num_pages=4)
            response = client.post(
                "/api/documents/ingest",
                files={"file": ("exceeded.pdf", pdf_bytes, "application/pdf")},
            )
            assert response.status_code == 413
            detail = response.json()["detail"]
            assert "exceeds the maximum allowed limit of 3 pages" in detail


class TestSecurityAndResourceLimits:
    """Tests for file size limits, content validation, and unsupported file types."""

    def test_oversized_upload_rejected(self):
        """File exceeding MAX_UPLOAD_SIZE_BYTES returns HTTP 413."""
        # Configure a small 1 KB limit to test limit enforcement cleanly
        small_limit_settings = Settings(MAX_UPLOAD_SIZE_BYTES=1024)
        app = create_app(small_limit_settings)

        with TestClient(app) as client:
            large_content = b"A" * 2048  # 2 KB, exceeds 1 KB limit
            response = client.post(
                "/api/documents/ingest",
                files={"file": ("large.txt", large_content, "text/plain")},
            )
            assert response.status_code == 413
            assert "exceeds" in response.json()["detail"].lower()

    def test_unsupported_file_extension_rejected(self, client_no_key):
        """Uploading unsupported extension (.docx, .png, .exe) returns HTTP 400."""
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("document.docx", b"PK\x03\x04dummy docx", "application/vnd.openxmlformats")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_fake_pdf_signature_rejected(self, client_no_key):
        """Uploading a non-PDF file named .pdf is rejected by content validation."""
        fake_pdf = b"This is not a real PDF file header."
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("malicious.pdf", fake_pdf, "application/pdf")},
        )
        assert response.status_code == 400
        assert "signature" in response.json()["detail"].lower()

    def test_unsupported_media_type_header(self, client_no_key):
        """Submitting with unsupported Content-Type returns HTTP 415."""
        response = client_no_key.post(
            "/api/documents/ingest",
            content=b"<xml>content</xml>",
            headers={"Content-Type": "application/xml"},
        )
        assert response.status_code == 415
        assert "Unsupported Media Type" in response.json()["detail"]


class TestNormalizationPipelineIntegration:
    """Tests verifying the common normalization pipeline across all input sources."""

    def test_normalization_applied_to_pasted_text(self, client_no_key):
        """CRLF and redundant blank lines in pasted text are normalized."""
        raw = "Para 1.\r\n\r\n\r\n\r\nPara 2.\twith   spaces."
        response = client_no_key.post("/api/documents/ingest", json={"text": raw})
        assert response.status_code == 200
        extracted = response.json()["extracted_text"]
        assert extracted == "Para 1.\n\nPara 2. with spaces."

    def test_normalization_applied_to_txt_file(self, client_no_key):
        """CRLF and redundant blank lines in TXT file are normalized."""
        raw = b"Heading\r\n\r\n\r\n\r\nSubheading\r\n   Indented   "
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("test.txt", raw, "text/plain")},
        )
        assert response.status_code == 200
        extracted = response.json()["extracted_text"]
        assert extracted == "Heading\n\nSubheading\nIndented"


class TestMIMEAndContentValidation:
    """Tests verifying that content is authoritative and unusual MIME types are accepted."""

    def test_valid_pdf_with_unexpected_mime_type(self, client_no_key):
        """Valid PDF with application/octet-stream MIME type is processed successfully."""
        pdf_bytes = helper_make_text_pdf("PDF with unusual MIME type header.")
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("valid.pdf", pdf_bytes, "application/octet-stream")},
        )
        assert response.status_code == 200
        assert response.json()["source_type"] == "pdf"
        assert "PDF with unusual MIME type" in response.json()["extracted_text"]

    def test_valid_txt_with_unexpected_mime_type(self, client_no_key):
        """Valid TXT with application/octet-stream MIME type is processed successfully."""
        txt_bytes = b"TXT notes with binary octet-stream header."
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("valid.txt", txt_bytes, "application/octet-stream")},
        )
        assert response.status_code == 200
        assert response.json()["source_type"] == "txt"
        assert "TXT notes with binary octet-stream" in response.json()["extracted_text"]

    def test_pdf_content_named_txt_rejected(self, client_no_key):
        """PDF content disguised with .txt extension is rejected by content signature validation."""
        pdf_bytes = helper_make_text_pdf("This is real PDF content disguised as TXT.")
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("disguised.txt", pdf_bytes, "text/plain")},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "PDF format signature" in detail
        assert ".txt" in detail

    def test_txt_content_named_pdf_rejected(self, client_no_key):
        """Plain text disguised with .pdf extension is rejected by PDF signature check."""
        fake_pdf = b"Plain text notes without any %PDF- signature."
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("fake.pdf", fake_pdf, "application/pdf")},
        )
        assert response.status_code == 400
        assert "signature" in response.json()["detail"].lower()


class TestErrorHandlingAndSanitization:
    """Tests ensuring error messages are client-safe and do not leak parser internals."""

    def test_malformed_multipart_returns_sanitized_error(self, client_no_key):
        """Corrupted multipart payload returns generic sanitized error message."""
        response = client_no_key.post(
            "/api/documents/ingest",
            content=b"invalid multipart body with corrupt structure",
            headers={"Content-Type": "multipart/form-data; boundary=boundary123"},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail == "Failed to parse form data. Please check that the request format is valid."
        assert "boundary character" not in detail
        assert "python_multipart" not in detail.lower()

    def test_corrupted_pdf_returns_sanitized_error(self, client_no_key):
        """Corrupted PDF returns client-safe error without raw PyMuPDF exception strings."""
        corrupted = b"%PDF-1.4\ncorrupted content that aborts abruptly"
        response = client_no_key.post(
            "/api/documents/ingest",
            files={"file": ("corrupt.pdf", corrupted, "application/pdf")},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "The PDF file is corrupted or could not be parsed."


class TestRequestHandlingConstraints:
    """Tests for ambiguous requests and JSON body size limits."""

    def test_rejects_request_with_both_file_and_text(self, client_no_key):
        """Providing both 'file' and 'text' in multipart form data returns HTTP 400."""
        response = client_no_key.post(
            "/api/documents/ingest",
            data={"text": "Here are my notes."},
            files={"file": ("notes.txt", b"File content here", "text/plain")},
        )
        assert response.status_code == 400
        assert "Cannot provide both 'file' and 'text'" in response.json()["detail"]

    def test_oversized_pasted_json_text_rejected(self):
        """Pasted JSON text exceeding MAX_UPLOAD_SIZE_BYTES returns HTTP 413."""
        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=1024)
        app = create_app(small_settings)
        with TestClient(app) as client:
            large_text = "A" * 2048
            response = client.post(
                "/api/documents/ingest",
                json={"text": large_text},
            )
            assert response.status_code == 413
            assert "exceeds" in response.json()["detail"].lower()


class TestUploadLimitEnforcement:
    """Tests verifying bounded chunk reading behavior and request body limits."""

    def test_oversized_json_with_content_length_rejected(self):
        """Oversized JSON request with Content-Length is rejected with HTTP 413."""
        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=500)
        app = create_app(small_settings)
        with TestClient(app) as client:
            large_body = b'{"text": "' + (b"A" * 1000) + b'"}'
            response = client.post(
                "/api/documents/ingest",
                content=large_body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(large_body))},
            )
            assert response.status_code == 413
            assert "exceeds" in response.json()["detail"].lower()
            assert "500 bytes" in response.json()["detail"]
            assert "0 MB" not in response.json()["detail"]

    def test_oversized_json_without_content_length_rejected(self):
        """Oversized JSON request without Content-Length is rejected with HTTP 413 by bounded stream reading."""
        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=500)
        app = create_app(small_settings)
        with TestClient(app) as client:
            def stream_json():
                yield b'{"text": "' + (b"A" * 600) + b'"}'

            response = client.post(
                "/api/documents/ingest",
                content=stream_json(),
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 413
            assert "exceeds" in response.json()["detail"].lower()
            assert "500 bytes" in response.json()["detail"]
            assert "0 MB" not in response.json()["detail"]

    def test_oversized_multipart_with_content_length_rejected(self):
        """Oversized multipart request with Content-Length is rejected before parsing."""
        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=500)
        app = create_app(small_settings)
        with TestClient(app) as client:
            large_file = b"A" * 1000
            response = client.post(
                "/api/documents/ingest",
                files={"file": ("large.txt", large_file, "text/plain")},
            )
            assert response.status_code == 413
            assert "exceeds" in response.json()["detail"].lower()
            assert "500 bytes" in response.json()["detail"]
            assert "0 MB" not in response.json()["detail"]

    def test_oversized_multipart_without_content_length_rejected(self):
        """Oversized multipart request without Content-Length is rejected with HTTP 413."""
        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=500)
        app = create_app(small_settings)
        with TestClient(app) as client:
            def stream_multipart():
                yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="notes.txt"\r\nContent-Type: text/plain\r\n\r\n'
                yield b"A" * 800
                yield b"\r\n--boundary--\r\n"

            response = client.post(
                "/api/documents/ingest",
                content=stream_multipart(),
                headers={"Content-Type": "multipart/form-data; boundary=boundary"},
            )
            assert response.status_code == 413
            assert "exceeds" in response.json()["detail"].lower()
            assert "500 bytes" in response.json()["detail"]
            assert "0 MB" not in response.json()["detail"]

    def test_misleading_content_length_cannot_bypass_limit(self):
        """A misleading Content-Length smaller than actual payload does not bypass stream counting."""
        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=500)
        app = create_app(small_settings)
        with TestClient(app) as client:
            # Header advertises 50 bytes (within 500), but stream actually delivers 800 bytes
            def stream_body():
                yield b'{"text": "' + (b"A" * 700) + b'"}'

            response = client.post(
                "/api/documents/ingest",
                content=stream_body(),
                headers={"Content-Type": "application/json", "Content-Length": "50"},
            )
            assert response.status_code == 413
            assert "exceeds" in response.json()["detail"].lower()
            assert "500 bytes" in response.json()["detail"]
            assert "0 MB" not in response.json()["detail"]

    @pytest.mark.anyio
    async def test_multiple_chunks_counted_cumulatively(self):
        """Multiple stream chunks each below limit are counted cumulatively until limit is exceeded."""
        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=500)
        app = create_app(small_settings)

        chunk_counter = 0

        async def mock_receive():
            nonlocal chunk_counter
            chunk_counter += 1
            if chunk_counter == 1:
                # 200 bytes <= 500
                return {"type": "http.request", "body": b'{"text": "' + (b"A" * 190), "more_body": True}
            elif chunk_counter == 2:
                # 200 + 200 = 400 bytes <= 500
                return {"type": "http.request", "body": b"A" * 200, "more_body": True}
            elif chunk_counter == 3:
                # 400 + 150 = 550 bytes > 500 -> exceeded on chunk 3
                return {"type": "http.request", "body": b"A" * 150, "more_body": True}
            return {"type": "http.request", "body": b'"}', "more_body": False}

        responses = []

        async def mock_send(message):
            responses.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/documents/ingest",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
        }

        await app(scope, mock_receive, mock_send)

        assert chunk_counter == 3
        status_event = next(r for r in responses if r["type"] == "http.response.start")
        assert status_event["status"] == 413

    @pytest.mark.anyio
    async def test_oversized_json_stream_aborts_without_consuming_remaining_chunks(self):
        """ASGI body streaming aborts immediately when max_bytes is exceeded without consuming subsequent chunks."""
        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=500)
        app = create_app(small_settings)

        chunks_requested = []

        async def mock_receive():
            chunks_requested.append(len(chunks_requested) + 1)
            if len(chunks_requested) == 1:
                return {"type": "http.request", "body": b'{"text": "' + (b"A" * 300), "more_body": True}
            elif len(chunks_requested) == 2:
                # 300 + 300 = 600 > 500, limit exceeded here!
                return {"type": "http.request", "body": b"A" * 300, "more_body": True}
            elif len(chunks_requested) == 3:
                # Should NEVER be called/consumed
                return {"type": "http.request", "body": b'"}', "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        responses = []

        async def mock_send(message):
            responses.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/documents/ingest",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
        }

        await app(scope, mock_receive, mock_send)

        # Verified: exactly 2 chunks requested; the 3rd chunk was NEVER read
        assert len(chunks_requested) == 2
        status_event = next(r for r in responses if r["type"] == "http.response.start")
        assert status_event["status"] == 413

    @pytest.mark.anyio
    async def test_oversized_multipart_stream_aborts_without_consuming_remaining_chunks(self):
        """Multipart ASGI streaming aborts immediately without spooling or reading remaining chunks."""
        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=500)
        app = create_app(small_settings)

        chunks_requested = []

        async def mock_receive():
            chunks_requested.append(len(chunks_requested) + 1)
            if len(chunks_requested) == 1:
                return {
                    "type": "http.request",
                    "body": b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="notes.txt"\r\nContent-Type: text/plain\r\n\r\n' + (b"A" * 200),
                    "more_body": True,
                }
            elif len(chunks_requested) == 2:
                # Exceeds 500 bytes limit here
                return {"type": "http.request", "body": b"A" * 400, "more_body": True}
            elif len(chunks_requested) == 3:
                # Should NEVER be called/consumed
                return {"type": "http.request", "body": b"\r\n--boundary--\r\n", "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        responses = []

        async def mock_send(message):
            responses.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/documents/ingest",
            "headers": [(b"content-type", b"multipart/form-data; boundary=boundary")],
            "query_string": b"",
        }

        await app(scope, mock_receive, mock_send)

        # Verified: exactly 2 chunks requested; subsequent chunks were NEVER consumed
        assert len(chunks_requested) == 2
        status_event = next(r for r in responses if r["type"] == "http.response.start")
        assert status_event["status"] == 413

    def test_request_exactly_at_limit_accepted(self):
        """Request whose body length is exactly at MAX_UPLOAD_SIZE_BYTES is accepted."""
        # Template: '{"text": "<padding>"}'
        template_prefix = b'{"text": "'
        template_suffix = b'"}'
        fixed_overhead = len(template_prefix) + len(template_suffix)
        target_limit = 200
        padding_len = target_limit - fixed_overhead

        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=target_limit)
        app = create_app(small_settings)
        with TestClient(app) as client:
            exact_body = template_prefix + (b"A" * padding_len) + template_suffix
            assert len(exact_body) == target_limit

            response = client.post(
                "/api/documents/ingest",
                content=exact_body,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 200
            assert response.json()["source_type"] == "pasted_text"

    def test_other_routes_unaffected_by_ingestion_middleware(self):
        """Routes other than POST /api/documents/ingest are not intercepted by the size limit middleware."""
        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=500)
        app = create_app(small_settings)
        with TestClient(app) as client:
            # 1. GET /api/health with large Content-Length header is unaffected and returns 200
            resp_health = client.get(
                "/api/health",
                headers={"Content-Length": "5000"},
            )
            assert resp_health.status_code == 200
            assert resp_health.json()["status"] == "healthy"

            # 2. GET /api/documents/ingest is not POST; returns 405 Method Not Allowed, not 413
            resp_method = client.get(
                "/api/documents/ingest",
                headers={"Content-Length": "5000"},
            )
            assert resp_method.status_code == 405

            # 3. Non-existent routes return 404, not 413
            resp_404 = client.post(
                "/api/nonexistent",
                content=b"A" * 2000,
                headers={"Content-Length": "2000"},
            )
            assert resp_404.status_code == 404

    def test_error_message_accurately_represents_small_configured_limits(self):
        """Size-limit error messages accurately represent small limits and never display '0 MB'."""
        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=500)
        app = create_app(small_settings)
        with TestClient(app) as client:
            response = client.post(
                "/api/documents/ingest",
                content=b'{"text": "' + (b"A" * 1000) + b'"}',
                headers={"Content-Type": "application/json", "Content-Length": "1012"},
            )
            assert response.status_code == 413
            detail = response.json()["detail"]
            assert "500 bytes" in detail
            assert "0 MB" not in detail
            assert detail == "Request body exceeds the maximum limit of 500 bytes."

    def test_format_byte_size_representations(self):
        """format_byte_size produces correct human-readable strings across units."""
        from app.middleware import format_byte_size

        assert format_byte_size(500) == "500 bytes"
        assert format_byte_size(1) == "1 byte"
        assert format_byte_size(1024) == "1 KB"
        assert format_byte_size(2048) == "2 KB"
        assert format_byte_size(1536) == "1.5 KB"
        assert format_byte_size(1024 * 1024) == "1 MB"
        assert format_byte_size(10 * 1024 * 1024) == "10 MB"

    def test_normal_json_and_multipart_continue_to_work(self):
        """Normal JSON and multipart requests work smoothly with request size limits active."""
        settings = Settings(MAX_UPLOAD_SIZE_BYTES=10 * 1024 * 1024)
        app = create_app(settings)
        with TestClient(app) as client:
            # Normal JSON
            r_json = client.post("/api/documents/ingest", json={"text": "Hello world from JSON notes."})
            assert r_json.status_code == 200
            assert "Hello world" in r_json.json()["extracted_text"]

            # Normal multipart
            r_multi = client.post(
                "/api/documents/ingest",
                files={"file": ("lecture.txt", b"Hello from uploaded lecture notes.", "text/plain")},
            )
            assert r_multi.status_code == 200
            assert "lecture notes" in r_multi.json()["extracted_text"]

    @pytest.mark.anyio
    async def test_read_safely_stops_at_limit_plus_one_byte(self):
        """read_upload_file_safely reads at most limit + 1 byte before raising OversizedFileError."""
        from unittest.mock import AsyncMock
        from starlette.datastructures import Headers, UploadFile as StarletteUploadFile
        from app.services.document_service import DocumentService
        from app.services.exceptions import OversizedFileError

        small_settings = Settings(MAX_UPLOAD_SIZE_BYTES=100)
        service = DocumentService(small_settings)

        # Create a mock upload file that records chunks read
        chunks_returned = [b"A" * 50, b"B" * 51, b"C" * 100]
        chunks_requested: list[int] = []

        async def mock_read(size: int = -1):
            chunks_requested.append(size)
            if chunks_returned:
                return chunks_returned.pop(0)
            return b""

        mock_file = StarletteUploadFile(
            file=None,
            filename="test.txt",
            headers=Headers({"content-type": "text/plain"}),
        )
        mock_file.read = mock_read
        mock_file.close = AsyncMock()

        with pytest.raises(OversizedFileError):
            await service.read_upload_file_safely(mock_file)

        # After reading 50 bytes, remaining was 50, so next read requested was min(64KB, 50 + 1) = 51 bytes!
        # The third chunk (b"C" * 100) was never requested!
        assert len(chunks_requested) == 2
        assert chunks_requested[0] == 101  # min(65536, 100 + 1)
        assert chunks_requested[1] == 51   # min(65536, 50 + 1)

