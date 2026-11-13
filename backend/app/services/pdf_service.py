"""
PDF extraction service using PyMuPDF.

Handles page-by-page text extraction, distinguishes text-based, image-only,
and mixed PDFs, and enforces security constraints (encryption, corruption).
"""

from dataclasses import dataclass, field
import fitz

from app.services.exceptions import (
    CorruptedPDFError,
    EmptyDocumentError,
    EncryptedPDFError,
    ImageOnlyPDFError,
    InvalidFileTypeError,
    NoExtractableTextError,
    PDFPageLimitExceededError,
)


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction and inspection."""

    text: str
    page_count: int
    has_images: bool
    is_mixed: bool
    warnings: list[str] = field(default_factory=list)


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 100) -> PDFExtractionResult:
    """
    Extract text and analyze structure of a PDF document.

    Inspects each page for extractable text and embedded images. Distinguishes:
    - Text-based PDFs: extractable text, no detected images.
    - Mixed PDFs: extractable text and detected images (warning attached).
    - Image-only PDFs: no extractable text, but images detected (rejected with OCR guidance).
    - No-extractable-text PDFs: neither extractable text nor images (rejected).

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        max_pages: Maximum allowed page count.

    Returns:
        PDFExtractionResult containing extracted text, page count, and warnings.

    Raises:
        InvalidFileTypeError: If bytes do not start with the PDF signature.
        CorruptedPDFError: If the document cannot be opened or parsed.
        EncryptedPDFError: If the document is password-protected or encrypted.
        EmptyDocumentError: If the document contains 0 pages.
        PDFPageLimitExceededError: If the document exceeds max_pages.
        ImageOnlyPDFError: If the document contains images but no extractable text.
        NoExtractableTextError: If the document contains neither text nor images.
    """
    # 1. Content signature validation
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
        raise InvalidFileTypeError(
            "File content does not match PDF format signature (must start with %PDF-)."
        )

    # 2. Open document with PyMuPDF
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise CorruptedPDFError("The PDF file is corrupted or could not be parsed.") from exc

    try:
        # 3. Check for encryption
        if doc.is_encrypted or doc.needs_pass:
            raise EncryptedPDFError(
                "The PDF document is encrypted or password-protected. "
                "Please provide an unencrypted PDF."
            )

        # 4. Check page count and page limit before extracting
        page_count = doc.page_count
        if page_count == 0:
            raise EmptyDocumentError("The PDF document is empty (contains 0 pages).")

        if page_count > max_pages:
            raise PDFPageLimitExceededError(
                f"The PDF contains {page_count} pages, which exceeds the maximum allowed limit of {max_pages} pages."
            )

        # 5. Extract text and detect images page by page
        extracted_pages: list[str] = []
        has_images = False

        for page_idx in range(page_count):
            page = doc[page_idx]
            page_text = page.get_text("text").strip()
            if page_text:
                extracted_pages.append(page_text)

            # Check for images on this page
            if not has_images:
                images = page.get_images(full=True)
                if images:
                    has_images = True

        combined_text = "\n\n".join(extracted_pages).strip()

        # 6. Distinguish text-based vs mixed vs image-only vs no-extractable-text
        if not combined_text:
            if has_images:
                raise ImageOnlyPDFError(
                    "The PDF contains no extractable text. Scanned or image-only PDFs are not "
                    "supported because OCR is out of scope. Please upload a text-based document "
                    "or paste text directly."
                )
            else:
                raise NoExtractableTextError(
                    "The PDF document contains no extractable text or images. "
                    "Please ensure the document contains readable text or paste your study notes directly."
                )

        warnings: list[str] = []
        is_mixed = False
        if has_images:
            is_mixed = True
            warnings.append(
                "This document contains embedded images. Text embedded inside images cannot "
                "be extracted because OCR is not supported."
            )

        return PDFExtractionResult(
            text=combined_text,
            page_count=page_count,
            has_images=has_images,
            is_mixed=is_mixed,
            warnings=warnings,
        )

    finally:
        doc.close()

