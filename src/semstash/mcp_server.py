"""MCP Server for semstash - AI agent interface.

Provides MCP tools for semantic storage operations using FastMCP.
Can be used with any MCP-compatible client. All content is addressed
by paths (like a filesystem), with '/' as root.

Usage:
    # Run the server
    python -m semstash.mcp_server

    # Or via CLI
    semstash mcp
"""

import base64
import json
import tempfile
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from semstash.config import DEFAULT_BROWSE_LIMIT, DEFAULT_SEARCH_TOP_K
from semstash.markdown import is_markdown_convertible, to_markdown
from semstash.utils import get_cached_stash

# Create FastMCP server
mcp = FastMCP("semstash")


def to_json(data: dict[str, Any]) -> str:
    """Format result as JSON string."""
    return json.dumps(data, indent=2, default=str)


# --- Tools ---


@mcp.tool()
def init() -> str:
    """Initialize semantic storage. Creates S3 bucket and vector index if they don't exist."""
    result = get_cached_stash().init()
    return to_json(
        {
            "bucket": result.bucket,
            "vector_bucket": result.vector_bucket,
            "region": result.region,
            "dimension": result.dimension,
            "message": "Storage initialized successfully",
        }
    )


@mcp.tool()
def upload(file_path: str, target: str, force: bool = False) -> str:
    """Upload a file to semantic storage. Stores content and generates embeddings for search.

    Args:
        file_path: Local file path to upload.
        target: Target path in storage. Use '/' for root (preserves filename),
            '/folder/' to upload to folder (preserves filename),
            '/folder/name.ext' to rename on upload.
        force: Overwrite if content already exists at target path.
    """
    source = Path(file_path)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    result = get_cached_stash().upload(file_path=source, target=target, force=force)
    return to_json(
        {
            "path": result.path,
            "key": result.key,
            "content_type": result.content_type,
            "file_size": result.file_size,
        }
    )


@mcp.tool()
def query(
    query_text: str,
    top_k: int = DEFAULT_SEARCH_TOP_K,
    tags: list[str] | None = None,
    path: str | None = None,
) -> str:
    """Query for content using natural language. Returns semantically similar content.

    Args:
        query_text: Natural language query.
        top_k: Maximum number of results to return.
        tags: Filter results by tags (any match).
        path: Filter results by path prefix (e.g., '/docs/').
    """
    results = get_cached_stash().query(query_text=query_text, top_k=top_k, tags=tags, path=path)
    return to_json(
        {
            "query": query_text,
            "count": len(results),
            "results": [
                {
                    "path": r.path,
                    "key": r.key,
                    "score": r.score,
                    "content_type": r.content_type,
                    "url": r.url,
                }
                for r in results
            ],
        }
    )


@mcp.tool()
def get(path: str, return_content: bool = False) -> str:
    """Get content metadata and optionally the content itself.

    Args:
        path: Path of the content (e.g., '/docs/readme.txt').
        return_content: If True, returns the actual content:
            - Text files: returns text directly
            - Documents (PDF, DOCX, XLSX, PPTX): converted to Markdown
            - Images: returns base64-encoded data
            - Audio/Video: returns URL only (too large)

    Returns:
        JSON with metadata, URL, and optionally content.
    """
    stash = get_cached_stash()
    result = stash.get(path)

    response: dict[str, Any] = {
        "path": result.path,
        "key": result.key,
        "content_type": result.content_type,
        "file_size": result.file_size,
        "url": result.url,
    }

    if return_content:
        # Download to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = stash.download(path, Path(tmpdir))

            # Handle based on content type
            if result.content_type.startswith("text/"):
                # Text files: return as-is
                response["content"] = dest.read_text(encoding="utf-8")
            elif is_markdown_convertible(result.content_type):
                # Documents: convert to Markdown
                response["content"] = to_markdown(dest, result.content_type)
                response["content_format"] = "markdown"
            elif result.content_type.startswith("image/"):
                # Images: return base64
                response["content"] = base64.b64encode(dest.read_bytes()).decode("ascii")
                response["content_format"] = "base64"
            elif result.content_type.startswith(("audio/", "video/")):
                # Audio/Video: too large, just return URL
                response["note"] = "Audio/video content available via URL only"
            else:
                # Other: try as text or skip
                try:
                    response["content"] = dest.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    response["note"] = "Binary content available via URL only"

    return to_json(response)


@mcp.tool()
def delete(path: str) -> str:
    """Delete content from storage.

    Args:
        path: Path of the content to delete (e.g., '/docs/readme.txt').
    """
    result = get_cached_stash().delete(path)
    return to_json({"path": path, "key": result.key, "deleted": result.deleted})


@mcp.tool()
def browse(path: str, limit: int = DEFAULT_BROWSE_LIMIT) -> str:
    """Browse stored content at a path.

    Args:
        path: Path to browse. Use '/' for root, '/folder/' for subfolders.
        limit: Maximum number of items to return.
    """
    result = get_cached_stash().browse(path=path, limit=limit)
    return to_json(
        {
            "path": path,
            "total": result.total,
            "items": [
                {
                    "path": item.path,
                    "key": item.key,
                    "content_type": item.content_type,
                    "file_size": item.file_size,
                }
                for item in result.items
            ],
        }
    )


@mcp.tool()
def stats() -> str:
    """Get storage statistics and AWS resource information."""
    s = get_cached_stash().get_stats()
    return to_json(
        {
            "content_count": s.content_count,
            "vector_count": s.vector_count,
            "storage_bytes": s.storage_bytes,
            "dimension": s.dimension,
            "bucket": s.bucket,
            "vector_bucket": s.vector_bucket,
            "index_name": s.index_name,
            "region": s.region,
        }
    )


def main() -> None:
    """Run the MCP server."""
    import os
    import sys

    # Accept bucket as command-line argument
    if len(sys.argv) > 1:
        os.environ["SEMSTASH_BUCKET"] = sys.argv[1]

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
