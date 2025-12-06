"""Shared utility functions for semstash interfaces."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semstash.client import SemStash


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string.

    Args:
        size_bytes: Size in bytes to format.

    Returns:
        Human-readable string like "1.5 MB" or "256 B".
    """
    size: float = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


# --- Path Utilities ---


def normalize_path(path: str) -> str:
    """Normalize path to have leading / and handle edge cases.

    Ensures consistent path format for user-facing paths.

    Args:
        path: Input path string.

    Returns:
        Normalized path with leading /.

    Examples:
        >>> normalize_path('docs/file.txt')
        '/docs/file.txt'
        >>> normalize_path('/docs/file.txt')
        '/docs/file.txt'
        >>> normalize_path('/')
        '/'
        >>> normalize_path('')
        '/'
    """
    if not path or path == "/":
        return "/"

    # Ensure leading /
    if not path.startswith("/"):
        path = "/" + path

    return path


def path_to_key(path: str) -> str:
    """Convert user-facing path to S3 key (strip leading /).

    Args:
        path: User-facing path with leading /.

    Returns:
        S3 key without leading /.

    Examples:
        >>> path_to_key('/docs/file.txt')
        'docs/file.txt'
        >>> path_to_key('/file.txt')
        'file.txt'
        >>> path_to_key('/')
        ''
    """
    path = normalize_path(path)
    if path == "/":
        return ""
    return path.lstrip("/")


def key_to_path(key: str) -> str:
    """Convert S3 key to user-facing path (add leading /).

    Args:
        key: S3 key without leading /.

    Returns:
        User-facing path with leading /.

    Examples:
        >>> key_to_path('docs/file.txt')
        '/docs/file.txt'
        >>> key_to_path('file.txt')
        '/file.txt'
        >>> key_to_path('')
        '/'
    """
    if not key:
        return "/"
    return "/" + key


def resolve_upload_target(source: Path, target: str) -> tuple[str, str]:
    """Resolve upload target to (key, path).

    Determines the S3 key and user-facing path based on the target string.
    If target ends with '/', the source filename is appended (folder semantics).
    Otherwise, target is used as the exact path (rename semantics).

    Args:
        source: Source file path (used to extract filename).
        target: Target path string.

    Returns:
        Tuple of (S3 key, user-facing path).

    Raises:
        ValueError: If target is empty or invalid.

    Examples:
        >>> resolve_upload_target(Path('file.txt'), '/')
        ('file.txt', '/file.txt')
        >>> resolve_upload_target(Path('file.txt'), '/docs/')
        ('docs/file.txt', '/docs/file.txt')
        >>> resolve_upload_target(Path('file.txt'), '/docs/readme.txt')
        ('docs/readme.txt', '/docs/readme.txt')
    """
    if not target:
        raise ValueError("Target path cannot be empty")

    target = normalize_path(target)
    filename = source.name

    # Folder semantics: target ends with /
    if target.endswith("/"):
        if target == "/":
            # Root folder
            path = f"/{filename}"
            key = filename
        else:
            # Subfolder
            path = f"{target}{filename}"
            key = path.lstrip("/")
    else:
        # Exact path (rename semantics)
        path = target
        key = target.lstrip("/")

    return (key, path)


# --- Navigation Utilities ---


def generate_breadcrumbs(path: str) -> list[tuple[str, str]]:
    """Generate breadcrumb navigation for a path.

    Args:
        path: Current path like '/docs/subdir/' or '/docs/file.txt'.

    Returns:
        List of (name, path) tuples for breadcrumb links.

    Examples:
        >>> generate_breadcrumbs('/')
        [('/', '/')]
        >>> generate_breadcrumbs('/docs/')
        [('/', '/'), ('docs', '/docs/')]
        >>> generate_breadcrumbs('/docs/subdir/')
        [('/', '/'), ('docs', '/docs/'), ('subdir', '/docs/subdir/')]
        >>> generate_breadcrumbs('/docs/file.txt')
        [('/', '/'), ('docs', '/docs/'), ('file.txt', '/docs/file.txt')]
    """
    path = normalize_path(path)
    if path == "/":
        return [("/", "/")]

    parts = path.strip("/").split("/")
    breadcrumbs: list[tuple[str, str]] = [("/", "/")]

    current = ""
    for i, part in enumerate(parts):
        current += f"/{part}"
        # Add trailing / for directories (all except possibly the last part)
        is_last = i == len(parts) - 1
        is_folder = path.endswith("/") or not is_last
        display_path = current + "/" if is_folder and not current.endswith("/") else current
        breadcrumbs.append((part, display_path))

    return breadcrumbs


def get_parent_path(path: str) -> str | None:
    """Get parent folder path.

    Args:
        path: Current path like '/docs/subdir/' or '/docs/file.txt'.

    Returns:
        Parent path or None if at root.

    Examples:
        >>> get_parent_path('/docs/subdir/')
        '/docs/'
        >>> get_parent_path('/docs/')
        '/'
        >>> get_parent_path('/')
        >>> get_parent_path('/file.txt')
        '/'
    """
    path = normalize_path(path)
    if path == "/":
        return None

    # Remove trailing slash for consistent handling
    path = path.rstrip("/")
    last_slash = path.rfind("/")

    if last_slash <= 0:
        return "/"

    return path[:last_slash] + "/"


def get_containing_folder(file_path: str) -> str:
    """Get the folder containing a file.

    Args:
        file_path: File path like '/docs/readme.txt'.

    Returns:
        Containing folder path.

    Examples:
        >>> get_containing_folder('/docs/readme.txt')
        '/docs/'
        >>> get_containing_folder('/readme.txt')
        '/'
        >>> get_containing_folder('/')
        '/'
    """
    file_path = normalize_path(file_path)
    if file_path == "/":
        return "/"

    # If it's already a folder path, return it
    if file_path.endswith("/"):
        return file_path

    last_slash = file_path.rfind("/")
    if last_slash <= 0:
        return "/"

    return file_path[: last_slash + 1]


def get_filename(file_path: str) -> str:
    """Extract filename from a path.

    Args:
        file_path: File path like '/docs/readme.txt'.

    Returns:
        Filename without path.

    Examples:
        >>> get_filename('/docs/readme.txt')
        'readme.txt'
        >>> get_filename('/readme.txt')
        'readme.txt'
        >>> get_filename('/')
        ''
    """
    file_path = normalize_path(file_path)
    if file_path == "/" or file_path.endswith("/"):
        return ""

    return file_path.split("/")[-1]


# --- Fragment Key Utilities ---


def make_chunk_key(base_key: str, chunk_id: str) -> str:
    """Create a chunk key by appending fragment identifier.

    Fragment notation uses '#' to separate base key from chunk identifier,
    similar to URL fragments.

    Args:
        base_key: Base S3 key (e.g., 'docs/report.pdf').
        chunk_id: Chunk identifier (e.g., 'page=1', 'slide=3').

    Returns:
        Combined key with fragment (e.g., 'docs/report.pdf#page=1').

    Examples:
        >>> make_chunk_key('docs/report.pdf', 'page=1')
        'docs/report.pdf#page=1'
        >>> make_chunk_key('audio/podcast.mp3', 'segment=5')
        'audio/podcast.mp3#segment=5'
        >>> make_chunk_key('file.txt', '')
        'file.txt'
    """
    if not chunk_id:
        return base_key
    return f"{base_key}#{chunk_id}"


def parse_chunk_key(chunk_key: str) -> tuple[str, str | None]:
    """Parse a chunk key into base key and fragment.

    Args:
        chunk_key: Key with optional fragment (e.g., 'docs/report.pdf#page=1').

    Returns:
        Tuple of (base_key, chunk_id or None).

    Examples:
        >>> parse_chunk_key('docs/report.pdf#page=1')
        ('docs/report.pdf', 'page=1')
        >>> parse_chunk_key('docs/report.pdf')
        ('docs/report.pdf', None)
        >>> parse_chunk_key('audio/file.mp3#segment=10')
        ('audio/file.mp3', 'segment=10')
    """
    if "#" not in chunk_key:
        return (chunk_key, None)

    parts = chunk_key.split("#", 1)
    return (parts[0], parts[1] if len(parts) > 1 else None)


def parse_chunk_id(chunk_id: str) -> tuple[str, int]:
    """Parse a chunk identifier into type and index.

    Args:
        chunk_id: Chunk identifier (e.g., 'page=5', 'segment=10').

    Returns:
        Tuple of (chunk_type, chunk_index).

    Raises:
        ValueError: If chunk_id format is invalid.

    Examples:
        >>> parse_chunk_id('page=5')
        ('page', 5)
        >>> parse_chunk_id('segment=10')
        ('segment', 10)
    """
    if "=" not in chunk_id:
        raise ValueError(f"Invalid chunk_id format: {chunk_id}")

    parts = chunk_id.split("=", 1)
    try:
        return (parts[0], int(parts[1]))
    except ValueError:
        raise ValueError(f"Invalid chunk index in: {chunk_id}") from None


def get_source_key(chunk_key: str) -> str:
    """Get the source file key from a chunk key.

    Args:
        chunk_key: Key with optional fragment.

    Returns:
        Base key without fragment.

    Examples:
        >>> get_source_key('docs/report.pdf#page=1')
        'docs/report.pdf'
        >>> get_source_key('docs/report.pdf')
        'docs/report.pdf'
    """
    base_key, _ = parse_chunk_key(chunk_key)
    return base_key


def is_chunk_key(key: str) -> bool:
    """Check if a key is a chunk key (has fragment).

    Args:
        key: S3 key to check.

    Returns:
        True if key contains a fragment identifier.

    Examples:
        >>> is_chunk_key('docs/report.pdf#page=1')
        True
        >>> is_chunk_key('docs/report.pdf')
        False
    """
    return "#" in key


# --- Caching Utilities ---


@cache
def get_cached_stash() -> SemStash:
    """Get or create a cached SemStash instance from environment variables.

    Uses functools.cache for thread-safe lazy initialization.
    The stash is created once and reused for all subsequent calls.

    Returns:
        Configured SemStash instance.

    Raises:
        ConfigurationError: If required environment variables are missing.
    """
    # Lazy import to avoid circular dependency
    from semstash.client import create_stash_from_env

    return create_stash_from_env()


def clear_stash_cache() -> None:
    """Clear the cached stash instance.

    Useful for testing or when reconfiguration is needed.
    """
    get_cached_stash.cache_clear()
