"""Markdown conversion utilities using Microsoft's markitdown.

Converts documents (PDF, DOCX, PPTX, XLSX) to Markdown for LLM consumption.
All processing is done locally - no external API calls.
"""

from pathlib import Path

from markitdown import MarkItDown

from semstash.exceptions import UnsupportedContentTypeError

# Content types that can be converted to Markdown
MARKDOWN_CONVERTIBLE_TYPES = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx
    }
)

# Human-readable names for error messages
MARKDOWN_FORMAT_NAMES = {
    "application/pdf": "PDF",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word (DOCX)",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel (XLSX)",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PPTX",
}


def is_markdown_convertible(content_type: str) -> bool:
    """Check if a content type can be converted to Markdown.

    Args:
        content_type: MIME type to check.

    Returns:
        True if the content type can be converted to Markdown.
    """
    return content_type in MARKDOWN_CONVERTIBLE_TYPES


def get_supported_formats() -> list[str]:
    """Get list of human-readable format names that can be converted to Markdown.

    Returns:
        List of format names (e.g., ["PDF", "Word (DOCX)", ...]).
    """
    return list(MARKDOWN_FORMAT_NAMES.values())


def to_markdown(file_path: Path, content_type: str | None = None) -> str:
    """Convert a document to Markdown.

    Uses Microsoft's markitdown library for local conversion.
    No external API calls are made.

    Args:
        file_path: Path to the document file.
        content_type: Optional MIME type. If provided, validates convertibility.

    Returns:
        Markdown text content.

    Raises:
        UnsupportedContentTypeError: If content type is not convertible.
        FileNotFoundError: If file doesn't exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if content_type and not is_markdown_convertible(content_type):
        supported = ", ".join(get_supported_formats())
        raise UnsupportedContentTypeError(
            f"Cannot convert {content_type} to Markdown. Supported formats: {supported}"
        )

    md = MarkItDown()
    result = md.convert(str(file_path))
    return result.text_content
