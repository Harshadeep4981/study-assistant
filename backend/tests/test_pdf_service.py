"""Tests for the PDF extraction service."""

import fitz
import pytest

from app.services.exceptions import (
    CorruptedPDFError,
    EmptyDocumentError,
    EncryptedPDFError,
    ImageOnlyPDFError,
    InvalidFileTypeError,
    NoExtractableTextError,
    PDFPageLimitExceededError,
)
from app.services.pdf_service import extract_text_from_pdf


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


class TestPDFService:
    """Unit tests for extract_text_from_pdf."""

    def test_extracts_text_based_pdf(self):
        """Text-based PDF is processed cleanly without image warnings."""
        pdf_bytes = helper_make_text_pdf("Cellular respiration is a metabolic pathway.")
        result = extract_text_from_pdf(pdf_bytes)

        assert "Cellular respiration" in result.text
        assert result.page_count == 1
        assert result.has_images is False
        assert result.is_mixed is False
        assert len(result.warnings) == 0

    def test_rejects_image_only_pdf_with_ocr_explanation(self):
        """Image-only PDF is rejected with an explanation that OCR is unsupported."""
        pdf_bytes = helper_make_image_only_pdf()
        with pytest.raises(ImageOnlyPDFError) as exc_info:
            extract_text_from_pdf(pdf_bytes)

        error_message = str(exc_info.value)
        assert "no extractable text" in error_message
        assert "OCR is out of scope" in error_message

    def test_extracts_mixed_pdf_with_warning(self):
        """Mixed PDF extracts available text and returns image/OCR warning."""
        pdf_bytes = helper_make_mixed_pdf("Mitosis is the process of cell division.")
        result = extract_text_from_pdf(pdf_bytes)

        assert "Mitosis is the process" in result.text
        assert result.page_count == 1
        assert result.has_images is True
        assert result.is_mixed is True
        assert len(result.warnings) == 1
        assert "embedded images" in result.warnings[0]
        assert "OCR is not supported" in result.warnings[0]

    def test_rejects_encrypted_pdf(self):
        """Password-protected / encrypted PDF is rejected."""
        pdf_bytes = helper_make_encrypted_pdf()
        with pytest.raises(EncryptedPDFError) as exc_info:
            extract_text_from_pdf(pdf_bytes)

        assert "encrypted or password-protected" in str(exc_info.value)

    def test_rejects_corrupted_pdf(self):
        """Corrupted PDF stream raises CorruptedPDFError."""
        corrupted_bytes = b"%PDF-1.4\nCorrupted content ending prematurely"
        with pytest.raises(CorruptedPDFError):
            extract_text_from_pdf(corrupted_bytes)

    def test_rejects_missing_pdf_signature(self):
        """Data not starting with %PDF- signature raises InvalidFileTypeError."""
        non_pdf_bytes = b"Hello, this is just plain text, not a PDF."
        with pytest.raises(InvalidFileTypeError) as exc_info:
            extract_text_from_pdf(non_pdf_bytes)

        assert "signature" in str(exc_info.value)

    def test_rejects_empty_bytes(self):
        """Empty byte string raises InvalidFileTypeError."""
        with pytest.raises(InvalidFileTypeError):
            extract_text_from_pdf(b"")

    def test_rejects_no_extractable_text_pdf(self):
        """Blank PDF with neither text nor images raises NoExtractableTextError."""
        pdf_bytes = helper_make_blank_pdf(num_pages=2)
        with pytest.raises(NoExtractableTextError) as exc_info:
            extract_text_from_pdf(pdf_bytes)

        error_message = str(exc_info.value)
        assert "no extractable text or images" in error_message

    def test_pdf_page_limit_exact_boundary_passes(self):
        """PDF with page count equal to max_pages extracts successfully."""
        pdf_bytes = helper_make_multipage_pdf(num_pages=5, text_per_page="Lecture section")
        result = extract_text_from_pdf(pdf_bytes, max_pages=5)

        assert result.page_count == 5
        assert "Lecture section 5" in result.text

    def test_pdf_page_limit_exceeded_rejected_early(self):
        """PDF with page count exceeding max_pages raises PDFPageLimitExceededError."""
        pdf_bytes = helper_make_multipage_pdf(num_pages=6, text_per_page="Lecture section")
        with pytest.raises(PDFPageLimitExceededError) as exc_info:
            extract_text_from_pdf(pdf_bytes, max_pages=5)

        error_message = str(exc_info.value)
        assert "exceeds the maximum allowed limit of 5 pages" in error_message

