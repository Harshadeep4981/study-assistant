"""
Document ingestion service.

Coordinates validation, safe reading, format inspection, and normalization
across pasted text, TXT file uploads, and PDF document uploads.
"""

from pathlib import Path
from fastapi import Request, UploadFile

from app.config import Settings, get_settings
from app.middleware import format_byte_size
from app.models.document import DocumentIngestResponse, DocumentMetadata
from app.services.exceptions import (
    EmptyDocumentError,
    EmptyTextError,
    InvalidEncodingError,
    InvalidFileTypeError,
    OversizedFileError,
    UnsupportedFileTypeError,
)
from app.services.pdf_service import extract_text_from_pdf
from app.services.text_normalizer import normalize_text


class DocumentService:
    """Service that orchestrates document ingestion and text extraction."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def read_upload_file_safely(self, file: UploadFile) -> bytes:
        """
        Read file bytes up to the configured limit without unbounded memory consumption.

        Reads in chunks of at most the remaining byte allowance plus one byte, detecting
        overflow immediately without reading unnecessary extra chunks.

        Args:
            file: FastAPI UploadFile instance.

        Returns:
            The raw bytes of the uploaded file.

        Raises:
            OversizedFileError: If file size exceeds the configured maximum.
        """
        max_bytes = self.settings.MAX_UPLOAD_SIZE_BYTES
        limit_str = format_byte_size(max_bytes)
        default_chunk_size = 64 * 1024  # 64 KB chunks
        chunks: list[bytes] = []
        total_read = 0

        try:
            while True:
                remaining = max_bytes - total_read
                # Read at most remaining allowance + 1 byte to detect overflow immediately
                read_size = min(default_chunk_size, remaining + 1)
                chunk = await file.read(read_size)
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > max_bytes:
                    raise OversizedFileError(
                        f"File size exceeds the maximum limit of {limit_str}."
                    )
                chunks.append(chunk)

            return b"".join(chunks)
        finally:
            await file.close()

    def ingest_pasted_text(self, raw_text: str) -> DocumentIngestResponse:
        """
        Ingest and normalize raw pasted study notes.

        Args:
            raw_text: Raw input string.

        Returns:
            DocumentIngestResponse with normalized text and metadata.

        Raises:
            EmptyTextError: If text is empty or whitespace-only.
            OversizedFileError: If raw text exceeds the configured size limit.
        """
        if not raw_text or not raw_text.strip():
            raise EmptyTextError("Text input cannot be empty or contain only whitespace.")

        raw_bytes_len = len(raw_text.encode("utf-8"))
        if raw_bytes_len > self.settings.MAX_UPLOAD_SIZE_BYTES:
            limit_str = format_byte_size(self.settings.MAX_UPLOAD_SIZE_BYTES)
            raise OversizedFileError(
                f"Pasted text size ({raw_bytes_len} bytes) exceeds the maximum limit of {limit_str}."
            )

        normalized = normalize_text(raw_text)
        raw_bytes_len = len(raw_text.encode("utf-8"))

        metadata = DocumentMetadata(
            filename=None,
            file_size_bytes=raw_bytes_len,
            page_count=None,
            character_count=len(normalized),
            word_count=len(normalized.split()),
            has_images=False,
        )

        return DocumentIngestResponse(
            extracted_text=normalized,
            source_type="pasted_text",
            metadata=metadata,
            warnings=[],
        )

    def ingest_txt_bytes(self, filename: str, content: bytes) -> DocumentIngestResponse:
        """
        Ingest, validate, and normalize a TXT file's contents.

        Args:
            filename: Original name of the file.
            content: Raw byte content of the file.

        Returns:
            DocumentIngestResponse with normalized text and metadata.

        Raises:
            EmptyDocumentError: If the file is 0 bytes or decodes to empty text.
            InvalidEncodingError: If content cannot be decoded as UTF-8 or contains binary null bytes.
        """
        if not content:
            raise EmptyDocumentError("The uploaded TXT file is empty (0 bytes).")

        # Size check
        if len(content) > self.settings.MAX_UPLOAD_SIZE_BYTES:
            limit_str = format_byte_size(self.settings.MAX_UPLOAD_SIZE_BYTES)
            raise OversizedFileError(
                f"File size exceeds the maximum limit of {limit_str}."
            )

        # Mismatched signature check: reject PDF binary disguised as .txt
        if content.startswith(b"%PDF-"):
            raise InvalidFileTypeError(
                "File content matches PDF format signature but was uploaded as a .txt file. "
                "Please upload with a .pdf extension."
            )

        # Encoding validation
        try:
            decoded_text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidEncodingError(
                "The uploaded TXT file contains an invalid text encoding and could not be decoded. "
                "Please ensure it is encoded in valid UTF-8."
            ) from exc

        # Binary check (null bytes indicate non-text files)
        if "\x00" in decoded_text:
            raise InvalidEncodingError(
                "The uploaded file contains binary data (null bytes). "
                "Only valid plain-text files are supported."
            )

        normalized = normalize_text(decoded_text)

        metadata = DocumentMetadata(
            filename=filename,
            file_size_bytes=len(content),
            page_count=None,
            character_count=len(normalized),
            word_count=len(normalized.split()),
            has_images=False,
        )

        return DocumentIngestResponse(
            extracted_text=normalized,
            source_type="txt",
            metadata=metadata,
            warnings=[],
        )

    def ingest_pdf_bytes(self, filename: str, content: bytes) -> DocumentIngestResponse:
        """
        Ingest, validate, and extract text from a PDF file's bytes.

        Args:
            filename: Original name of the file.
            content: Raw byte content of the PDF.

        Returns:
            DocumentIngestResponse with normalized text, metadata, and warnings.

        Raises:
            EmptyDocumentError: If file is 0 bytes.
            OversizedFileError: If file exceeds size limit.
            ImageOnlyPDFError: If PDF contains no extractable text.
            CorruptedPDFError: If PDF structure is invalid.
            EncryptedPDFError: If PDF is password-protected.
        """
        if not content:
            raise EmptyDocumentError("The uploaded PDF file is empty (0 bytes).")

        # Size check
        if len(content) > self.settings.MAX_UPLOAD_SIZE_BYTES:
            limit_str = format_byte_size(self.settings.MAX_UPLOAD_SIZE_BYTES)
            raise OversizedFileError(
                f"File size exceeds the maximum limit of {limit_str}."
            )

        # PyMuPDF extraction and structural inspection
        extraction = extract_text_from_pdf(content, max_pages=self.settings.MAX_PDF_PAGES)
        normalized = normalize_text(extraction.text)

        metadata = DocumentMetadata(
            filename=filename,
            file_size_bytes=len(content),
            page_count=extraction.page_count,
            character_count=len(normalized),
            word_count=len(normalized.split()),
            has_images=extraction.has_images,
        )

        return DocumentIngestResponse(
            extracted_text=normalized,
            source_type="pdf",
            metadata=metadata,
            warnings=extraction.warnings,
        )

    async def ingest_upload_file(self, file: UploadFile) -> DocumentIngestResponse:
        """
        Ingest an uploaded file (TXT or PDF) safely.

        Args:
            file: FastAPI UploadFile instance.

        Returns:
            DocumentIngestResponse with normalized text and metadata.

        Raises:
            UnsupportedFileTypeError: If file extension is neither .txt nor .pdf.
            OversizedFileError: If file exceeds size limit.
            DocumentProcessingError: If extraction fails.
        """
        filename = file.filename or "unknown"
        ext = Path(filename).suffix.lower()

        if ext not in [".txt", ".pdf"]:
            # Make sure resources are closed
            await file.close()
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{ext}'. Only .txt and .pdf files are supported."
            )

        content = await self.read_upload_file_safely(file)

        if ext == ".txt":
            return self.ingest_txt_bytes(filename=filename, content=content)
        else:
            return self.ingest_pdf_bytes(filename=filename, content=content)


def get_document_service(request: Request) -> DocumentService:
    """
    FastAPI dependency that provides a DocumentService instance.

    Uses settings from app state if available, or loads default cached settings.
    """
    settings = getattr(request.app.state, "settings", None) or get_settings()
    return DocumentService(settings)

