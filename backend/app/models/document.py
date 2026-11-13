"""
Document ingestion Pydantic models.

Defines schemas for text input requests, document metadata, and ingestion responses.
"""

from pydantic import BaseModel, Field


class TextIngestRequest(BaseModel):
    """Request model for pasted study text."""

    text: str = Field(
        ...,
        description="Raw study notes or text content to ingest.",
        examples=["Photosynthesis is the process by which green plants..."],
    )


class DocumentMetadata(BaseModel):
    """Metadata describing the ingested document."""

    filename: str | None = Field(
        default=None,
        description="Original name of the uploaded file, or null if text was pasted.",
    )
    file_size_bytes: int | None = Field(
        default=None,
        description="Size of the raw input in bytes.",
    )
    page_count: int | None = Field(
        default=None,
        description="Number of pages in the PDF document, or null for plain text.",
    )
    character_count: int = Field(
        ...,
        description="Total character count of the normalized text.",
    )
    word_count: int = Field(
        ...,
        description="Total word count of the normalized text.",
    )
    has_images: bool = Field(
        default=False,
        description="Whether embedded images were detected in the document.",
    )


class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion."""

    extracted_text: str = Field(
        ...,
        description="The extracted and normalized plain text.",
    )
    source_type: str = Field(
        ...,
        description="Source of the input: 'pasted_text', 'txt', or 'pdf'.",
    )
    metadata: DocumentMetadata = Field(
        ...,
        description="Metadata associated with the ingested content.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings regarding extraction limitations (e.g. embedded images).",
    )

