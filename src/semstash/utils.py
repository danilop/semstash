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
        if is_folder and not current.endswith("/"):
            display_path = current + "/"
        else:
            display_path = current
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
