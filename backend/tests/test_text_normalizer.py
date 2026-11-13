"""Tests for text normalization service."""

import pytest

from app.services.exceptions import EmptyTextError
from app.services.text_normalizer import normalize_text


class TestTextNormalization:
    """Tests for normalising whitespace, line endings, and paragraphs."""

    def test_normalizes_line_endings_crlf(self):
        """CRLF line endings are converted to standard LF."""
        raw = "Line 1\r\nLine 2\r\nLine 3"
        result = normalize_text(raw)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_normalizes_line_endings_cr(self):
        """Old Mac CR line endings are converted to standard LF."""
        raw = "Line 1\rLine 2\rLine 3"
        result = normalize_text(raw)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_collapses_multiple_horizontal_spaces(self):
        """Excessive spaces and tabs within a line are collapsed to single spaces."""
        raw = "Word1     Word2\t\t\tWord3    Word4"
        result = normalize_text(raw)
        assert result == "Word1 Word2 Word3 Word4"

    def test_trims_trailing_whitespace_per_line(self):
        """Trailing spaces and tabs on individual lines are stripped."""
        raw = "Line 1    \nLine 2\t\t\nLine 3   "
        result = normalize_text(raw)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_preserves_paragraph_boundaries(self):
        """Double newlines separating paragraphs are preserved."""
        raw = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        result = normalize_text(raw)
        assert result == "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."

    def test_collapses_excessive_vertical_whitespace(self):
        """Three or more consecutive newlines are collapsed to two (paragraph boundary)."""
        raw = "Paragraph 1.\n\n\n\n\n\nParagraph 2."
        result = normalize_text(raw)
        assert result == "Paragraph 1.\n\nParagraph 2."

    def test_normalizes_unicode_spaces(self):
        """Non-breaking spaces (\\xa0) and zero-width spaces are normalized."""
        raw = "Word1\u00a0Word2\u200bWord3"
        result = normalize_text(raw)
        assert result == "Word1 Word2Word3"

    def test_trims_overall_leading_and_trailing_whitespace(self):
        """Outer whitespace surrounding the document is stripped."""
        raw = "\n\n\t  Hello World!  \n\t\n"
        result = normalize_text(raw)
        assert result == "Hello World!"

    def test_rejects_empty_string(self):
        """Empty string raises EmptyTextError."""
        with pytest.raises(EmptyTextError) as exc_info:
            normalize_text("")
        assert "empty or contain only whitespace" in str(exc_info.value)

    def test_rejects_whitespace_only_string(self):
        """String containing only spaces, tabs, and newlines raises EmptyTextError."""
        with pytest.raises(EmptyTextError) as exc_info:
            normalize_text("   \t  \r\n \n   ")
        assert "empty or contain only whitespace" in str(exc_info.value)

    def test_rejects_non_string_input(self):
        """Non-string inputs raise EmptyTextError."""
        with pytest.raises(EmptyTextError):
            normalize_text(None)  # type: ignore
        with pytest.raises(EmptyTextError):
            normalize_text(12345)  # type: ignore

