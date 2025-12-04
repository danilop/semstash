"""Shared utility functions for semstash interfaces."""

from functools import cache

from semstash.client import SemStash, create_stash_from_env


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
    return create_stash_from_env()


def clear_stash_cache() -> None:
    """Clear the cached stash instance.

    Useful for testing or when reconfiguration is needed.
    """
    get_cached_stash.cache_clear()
