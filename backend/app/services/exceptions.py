"""
Document processing domain exceptions.

Each exception maps to an appropriate HTTP status code and message.
"""


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""

    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class EmptyTextError(DocumentProcessingError):
    """Raised when text input is empty or whitespace-only."""

    def __init__(self, message: str = "Text input cannot be empty or contain only whitespace."):
        super().__init__(message=message, status_code=400)


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when an uploaded file is not an allowed type."""

    def __init__(self, message: str = "Unsupported file type. Only .txt and .pdf files are supported."):
        super().__init__(message=message, status_code=400)


class InvalidFileTypeError(DocumentProcessingError):
    """Raised when file content does not match its expected format or signature."""

    def __init__(self, message: str = "File content does not match the expected format."):
        super().__init__(message=message, status_code=400)


class InvalidEncodingError(DocumentProcessingError):
    """Raised when a text file cannot be decoded."""

    def __init__(
        self,
        message: str = (
            "The uploaded TXT file contains an invalid text encoding and could not be decoded. "
            "Please ensure it is encoded in valid UTF-8."
        ),
    ):
        super().__init__(message=message, status_code=400)


class EmptyDocumentError(DocumentProcessingError):
    """Raised when an uploaded document contains no content."""

    def __init__(self, message: str = "The uploaded document contains no content."):
        super().__init__(message=message, status_code=400)


class CorruptedPDFError(DocumentProcessingError):
    """Raised when a PDF file is corrupted or cannot be parsed."""

    def __init__(self, message: str = "The PDF file is corrupted or could not be parsed."):
        super().__init__(message=message, status_code=400)


class EncryptedPDFError(DocumentProcessingError):
    """Raised when a PDF file is encrypted or password-protected."""

    def __init__(
        self,
        message: str = (
            "The PDF document is encrypted or password-protected. "
            "Please provide an unencrypted PDF."
        ),
    ):
        super().__init__(message=message, status_code=400)


class ImageOnlyPDFError(DocumentProcessingError):
    """Raised when a PDF contains no extractable text but contains images (e.g. scanned/image-only)."""

    def __init__(
        self,
        message: str = (
            "The PDF contains no extractable text. Scanned or image-only PDFs are not "
            "supported because OCR is out of scope. Please upload a text-based document "
            "or paste text directly."
        ),
    ):
        super().__init__(message=message, status_code=422)


class NoExtractableTextError(DocumentProcessingError):
    """Raised when a PDF contains neither extractable text nor detectable images (e.g. blank pages or vector-only drawings)."""

    def __init__(
        self,
        message: str = (
            "The PDF document contains no extractable text or images. "
            "Please ensure the document contains readable text or paste your study notes directly."
        ),
    ):
        super().__init__(message=message, status_code=422)


class PDFPageLimitExceededError(DocumentProcessingError):
    """Raised when a PDF exceeds the maximum allowed page count."""

    def __init__(self, message: str = "PDF page count exceeds the allowed limit."):
        super().__init__(message=message, status_code=413)


class OversizedFileError(DocumentProcessingError):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(self, message: str = "File size exceeds the allowed limit."):
        super().__init__(message=message, status_code=413)


# ----------------------------------------------------------------------
# AI Generation domain exceptions
# ----------------------------------------------------------------------


class AIGenerationError(Exception):
    """Base exception for AI generation errors."""

    def __init__(self, message: str, status_code: int = 502):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AIAuthenticationError(AIGenerationError):
    """Raised when Groq API key is invalid or unauthorized."""

    def __init__(
        self,
        message: str = (
            "AI service authentication failed. "
            "Please check that a valid GROQ_API_KEY is configured."
        ),
    ):
        super().__init__(message=message, status_code=502)


class AIRateLimitError(AIGenerationError):
    """Raised when Groq API rate limit is exceeded."""

    def __init__(
        self,
        message: str = "AI service rate limit exceeded. Please try again later.",
        retry_after: int | None = None,
    ):
        super().__init__(message=message, status_code=429)
        self.retry_after = retry_after


class AIBudgetExceededError(AIGenerationError):
    """Raised when request-level processing budget (API calls or operations) is exhausted."""

    def __init__(
        self,
        message: str = (
            "Request processing budget exceeded. The document requires more AI operations "
            "or API calls than the allowed allowance for a single request."
        ),
        status_code: int = 429,
    ):
        super().__init__(message=message, status_code=status_code)


class AITimeoutError(AIGenerationError):
    """Raised when Groq API call times out."""

    def __init__(
        self,
        message: str = "AI service request timed out while generating content. Please try again.",
    ):
        super().__init__(message=message, status_code=504)


class AIProviderError(AIGenerationError):
    """Raised when Groq API returns a generic connection or status error."""

    def __init__(
        self,
        message: str = "An error occurred while communicating with the AI service.",
    ):
        super().__init__(message=message, status_code=502)


class AIGenerationValidationError(AIGenerationError):
    """Raised when AI output fails JSON parsing or schema validation after all retries."""

    def __init__(
        self,
        message: str = (
            "Failed to generate valid study material from the AI provider. "
            "The model output could not be parsed or validated after retries."
        ),
    ):
        super().__init__(message=message, status_code=502)


class AISummarizationError(AIGenerationError):
    """Raised when an error occurs during large-document chunking or summarization."""

    def __init__(
        self,
        message: str = (
            "An error occurred while summarizing large study material. "
            "Failed to produce a valid consolidated summary after retries."
        ),
    ):
        super().__init__(message=message, status_code=502)

