"""
Text normalization service.

Handles whitespace normalization, line ending standardization, and
paragraph boundary preservation for study material ingestion.
"""

import re

from app.services.exceptions import EmptyTextError


def normalize_text(text: str) -> str:
    """
    Normalize text content for consistent downstream processing.

    Transformations:
    1. Validates that the input is a non-empty string.
    2. Replaces unicode non-breaking spaces and zero-width spaces with standard spaces.
    3. Normalizes all line endings (CRLF, CR) to LF (\\n).
    4. Trims trailing whitespace from each line and collapses redundant horizontal spaces.
    5. Collapses three or more consecutive newlines down to two (\\n\\n),
       preserving meaningful paragraph boundaries without excessive vertical whitespace.
    6. Strips leading and trailing whitespace from the overall result.
    7. Validates that the final normalized text is not empty.

    Args:
        text: Raw input string.

    Returns:
        Cleaned and normalized text.

    Raises:
        EmptyTextError: If text is empty, whitespace-only, or not a string.
    """
    if not isinstance(text, str):
        raise EmptyTextError("Text input must be a string.")

    if not text.strip():
        raise EmptyTextError("Text input cannot be empty or contain only whitespace.")

    # 1. Normalize unicode space quirks
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")

    # 2. Normalize line endings (CRLF / CR -> LF)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Process line-by-line: collapse internal horizontal whitespace, strip trailing whitespace
    lines: list[str] = []
    for line in text.split("\n"):
        # Collapse multiple spaces or tabs into a single space
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)

    text = "\n".join(lines)

    # 4. Collapse 3 or more newlines to 2 (preserves double-newline paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Final outer strip
    normalized = text.strip()

    if not normalized:
        raise EmptyTextError("Text input cannot be empty or contain only whitespace.")

    return normalized

