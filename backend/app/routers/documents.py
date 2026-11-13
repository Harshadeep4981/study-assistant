"""
Document ingestion router.

Provides unified endpoint for ingesting study material via pasted text (JSON)
or file uploads (TXT/PDF via multipart form-data).
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.middleware import format_byte_size
from app.models.document import DocumentIngestResponse
from app.services.document_service import DocumentService, get_document_service
from app.services.exceptions import DocumentProcessingError

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest study document or text",
    description=(
        "Unified ingestion endpoint. Accepts study notes as JSON (`{'text': '...'}`) "
        "or file uploads (`.txt`, `.pdf`) via `multipart/form-data` with field name `file`. "
        "Extracts, normalizes, and returns plain text with metadata."
    ),
    responses={
        200: {"description": "Document successfully extracted and normalized."},
        400: {"description": "Empty text, invalid encoding, corrupted or encrypted PDF."},
        413: {"description": "File exceeds maximum upload size limit."},
        415: {"description": "Unsupported media type or file format."},
        422: {"description": "Image-only PDF where OCR is unsupported."},
    },
)
async def ingest_document(
    request: Request,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentIngestResponse:
    """
    Ingest study notes or documents.

    Inspects Content-Type:
    - `application/json`: reads `text` field.
    - `multipart/form-data`: reads `file` (UploadFile) or `text` (form field).
    """
    content_type = request.headers.get("content-type", "").lower()

    # 1. JSON handling
    if "application/json" in content_type:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                cl_int = int(content_length)
                if cl_int > document_service.settings.MAX_UPLOAD_SIZE_BYTES:
                    limit_str = format_byte_size(document_service.settings.MAX_UPLOAD_SIZE_BYTES)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Request body exceeds the maximum limit of {limit_str}.",
                    )
            except ValueError:
                pass

        try:
            body = await request.json()
        except (DocumentProcessingError, HTTPException):
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON in request body.",
            ) from exc

        if not isinstance(body, dict) or "text" not in body:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON body must include a 'text' field.",
            )

        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text input cannot be empty or contain only whitespace.",
            )

        return document_service.ingest_pasted_text(text)

    # 2. Multipart form-data or urlencoded handling
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
        except (DocumentProcessingError, HTTPException):
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to parse form data. Please check that the request format is valid.",
            ) from exc

        file_item = form.get("file")
        text_item = form.get("text")

        has_file = (
            isinstance(file_item, StarletteUploadFile)
            or (hasattr(file_item, "filename") and hasattr(file_item, "read"))
        ) and bool(getattr(file_item, "filename", None))

        has_text = isinstance(text_item, str) and bool(text_item.strip())

        # Disallow ambiguous requests providing both file and text
        if has_file and has_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot provide both 'file' and 'text'. Supply either a file upload or text input, not both.",
            )

        # Case A: File uploaded
        if has_file:
            return await document_service.ingest_upload_file(file_item)

        # Case B: Text provided via form data
        elif has_text:
            return document_service.ingest_pasted_text(text_item)

        # Case C: Neither provided (or empty file/text)
        if isinstance(file_item, StarletteUploadFile) and not file_item.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a filename.",
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document or text provided. Supply 'file' as form-data or 'text' as JSON.",
        )

    # 3. Unsupported Content-Type
    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=(
            f"Unsupported Media Type '{content_type}'. "
            "Expected 'application/json' or 'multipart/form-data'."
        ),
    )

