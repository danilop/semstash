"""FastAPI web application for semstash.

Provides a REST API and web UI for semantic storage operations.

Usage:
    # Run the server
    uvicorn semstash.web:app --reload

    # Or with Python
    python -m semstash.web
"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from semstash.client import SemStash, create_stash_from_env
from semstash.config import (
    DEFAULT_BROWSE_LIMIT,
    DEFAULT_DIMENSION,
    DEFAULT_REGION,
    DEFAULT_SEARCH_TOP_K,
)
from semstash.exceptions import (
    AlreadyExistsError,
    BucketNotFoundError,
    ContentExistsError,
    ContentNotFoundError,
    NotInitializedError,
    SemStashError,
    StorageError,
    UnsupportedContentTypeError,
)
from semstash.utils import format_size

# Template directory (relative to this file)
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Set up Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Add format_size to Jinja2 globals
templates.env.globals["format_size"] = format_size

# Create FastAPI app
app = FastAPI(
    title="SemStash API",
    description="Unlimited semantic storage for humans and AI agents",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global stash instance (cached for performance)
_stash: SemStash | None = None


def get_stash() -> SemStash:
    """Get or create the SemStash instance from environment variables."""
    global _stash
    if _stash is None:
        _stash = create_stash_from_env()
    return _stash


# --- Response Models ---


class SuccessResponse(BaseModel):
    """Base success response."""

    success: bool = True


class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = False
    error: dict[str, str]


class InitResponse(SuccessResponse):
    """Init endpoint response."""

    bucket: str
    vector_bucket: str
    region: str
    dimension: int


class UploadResponse(SuccessResponse):
    """Upload endpoint response."""

    key: str
    content_type: str
    file_size: int
    dimension: int


class QueryResultItem(BaseModel):
    """Single query result."""

    key: str
    score: float
    content_type: str | None
    file_size: int | None
    url: str | None


class QueryResponse(SuccessResponse):
    """Query endpoint response."""

    query: str
    count: int
    results: list[QueryResultItem]


class GetResponse(SuccessResponse):
    """Get endpoint response."""

    key: str
    content_type: str
    file_size: int
    created_at: str
    url: str


class DeleteResponse(SuccessResponse):
    """Delete endpoint response."""

    key: str
    deleted: bool


class BrowseItem(BaseModel):
    """Browse result item."""

    key: str
    content_type: str
    file_size: int
    created_at: str


class BrowseResponse(SuccessResponse):
    """Browse endpoint response."""

    total: int
    next_token: str | None = None
    items: list[BrowseItem]


class StatsResponse(SuccessResponse):
    """Stats endpoint response."""

    content_count: int
    vector_count: int
    storage_bytes: int
    storage_gb: float
    dimension: int


class CheckResponse(SuccessResponse):
    """Check endpoint response."""

    content_count: int
    vector_count: int
    orphaned_vectors: list[str]
    missing_vectors: list[str]
    is_consistent: bool
    message: str


class SyncResponse(SuccessResponse):
    """Sync endpoint response."""

    deleted_vectors: list[str]
    created_vectors: list[str]
    failed_keys: list[str]
    deleted_count: int
    created_count: int
    message: str


class DestroyResponse(SuccessResponse):
    """Destroy endpoint response."""

    bucket: str
    vector_bucket: str
    content_deleted: int
    vectors_deleted: int
    bucket_deleted: bool
    vector_bucket_deleted: bool
    destroyed: bool
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    bucket: str | None
    region: str


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    bucket = os.environ.get("SEMSTASH_BUCKET")
    region = os.environ.get("SEMSTASH_REGION", DEFAULT_REGION)

    return HealthResponse(
        status="healthy",
        bucket=bucket,
        region=region,
    )


@app.post("/init", response_model=InitResponse)
async def init(
    bucket: str = Form(...),
    region: str = Form(DEFAULT_REGION),
    dimension: int = Form(DEFAULT_DIMENSION),
) -> InitResponse:
    """Create new semantic storage (fails if already exists)."""
    global _stash

    try:
        _stash = SemStash(
            bucket=bucket,
            region=region,
            dimension=dimension,
            auto_init=False,
        )
        result = _stash.init()

        return InitResponse(
            bucket=result.bucket,
            vector_bucket=result.vector_bucket,
            region=result.region,
            dimension=result.dimension,
        )
    except AlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    except SemStashError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


@app.post("/open", response_model=InitResponse)
async def open_stash(
    bucket: str = Form(...),
    region: str = Form(DEFAULT_REGION),
    dimension: int = Form(DEFAULT_DIMENSION),
) -> InitResponse:
    """Open existing semantic storage (fails if not found)."""
    global _stash

    try:
        _stash = SemStash(
            bucket=bucket,
            region=region,
            dimension=dimension,
            auto_init=False,
        )
        result = _stash.open()

        return InitResponse(
            bucket=result.bucket,
            vector_bucket=result.vector_bucket,
            region=result.region,
            dimension=result.dimension,
        )
    except BucketNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except SemStashError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: Annotated[UploadFile, File(...)],
    key: Annotated[str | None, Form()] = None,
    tags: Annotated[str | None, Form()] = None,
    force: Annotated[bool, Form()] = False,
) -> UploadResponse:
    """Upload a file to semantic storage.

    Args:
        file: The file to upload.
        key: Optional key for the file. Defaults to the filename.
        tags: Optional comma-separated tags.
        force: If True, overwrite existing content with the same key.
    """
    try:
        stash = get_stash()

        # Save uploaded file temporarily
        suffix = Path(file.filename or "file").suffix
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            # Parse tags if provided
            tag_list = tags.split(",") if tags else None

            result = stash.upload(
                file_path=tmp_path,
                key=key or file.filename,
                tags=tag_list,
                force=force,
            )

            return UploadResponse(
                key=result.key,
                content_type=result.content_type,
                file_size=result.file_size,
                dimension=result.dimension,
            )
        finally:
            tmp_path.unlink()  # Clean up temp file

    except NotInitializedError:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. POST to /init first.",
        ) from None
    except ContentExistsError as e:
        raise HTTPException(
            status_code=409,
            detail=f"{e} Use force=true to overwrite.",
        ) from None
    except UnsupportedContentTypeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except SemStashError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.get("/query", response_model=QueryResponse)
async def query(
    q: str = Query(..., description="Query text"),
    top_k: int = Query(DEFAULT_SEARCH_TOP_K, ge=1, le=100, description="Max results"),
    content_type: str | None = Query(None, description="Filter by content type"),
    tag: list[str] | None = Query(None, description="Filter by tags (any match)"),
) -> QueryResponse:
    """Query for content using natural language."""
    try:
        stash = get_stash()
        results = stash.query(
            query_text=q,
            top_k=top_k,
            content_type=content_type,
            tags=tag,
        )

        return QueryResponse(
            query=q,
            count=len(results),
            results=[
                QueryResultItem(
                    key=r.key,
                    score=r.score,
                    content_type=r.content_type,
                    file_size=r.file_size,
                    url=r.url,
                )
                for r in results
            ],
        )
    except NotInitializedError:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. POST to /init first.",
        ) from None
    except SemStashError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.get("/content/{key:path}", response_model=GetResponse)
async def get_content(key: str) -> GetResponse:
    """Get content metadata and download URL."""
    try:
        stash = get_stash()
        result = stash.get(key)

        return GetResponse(
            key=result.key,
            content_type=result.content_type,
            file_size=result.file_size,
            created_at=result.created_at.isoformat(),
            url=result.url,
        )
    except ContentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Content not found: {key}") from None
    except NotInitializedError:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. POST to /init first.",
        ) from None
    except SemStashError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.delete("/content/{key:path}", response_model=DeleteResponse)
async def delete(key: str) -> DeleteResponse:
    """Delete content from storage."""
    try:
        stash = get_stash()
        result = stash.delete(key)

        return DeleteResponse(
            key=result.key,
            deleted=result.deleted,
        )
    except NotInitializedError:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. POST to /init first.",
        ) from None
    except SemStashError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.get("/browse", response_model=BrowseResponse)
async def browse(
    prefix: str = Query("", description="Key prefix filter"),
    content_type: str | None = Query(None, description="Content type filter"),
    limit: int = Query(DEFAULT_BROWSE_LIMIT, ge=1, le=100, description="Max results"),
    next_token: str | None = Query(None, description="Pagination token"),
) -> BrowseResponse:
    """Browse stored content."""
    try:
        stash = get_stash()
        result = stash.browse(
            prefix=prefix,
            content_type=content_type,
            limit=limit,
            continuation_token=next_token,
        )

        return BrowseResponse(
            total=result.total,
            next_token=result.next_token,
            items=[
                BrowseItem(
                    key=item.key,
                    content_type=item.content_type,
                    file_size=item.file_size,
                    created_at=item.created_at.isoformat(),
                )
                for item in result.items
            ],
        )
    except NotInitializedError:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. POST to /init first.",
        ) from None
    except SemStashError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Get storage statistics."""
    try:
        stash = get_stash()
        result = stash.get_stats()

        return StatsResponse(
            content_count=result.content_count,
            vector_count=result.vector_count,
            storage_bytes=result.storage_bytes,
            storage_gb=result.storage_gb,
            dimension=result.dimension,
        )
    except NotInitializedError:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. POST to /init first.",
        ) from None
    except SemStashError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.get("/check", response_model=CheckResponse)
async def check() -> CheckResponse:
    """Check storage consistency between S3 content and vector embeddings."""
    try:
        stash = get_stash()
        result = stash.check()

        return CheckResponse(
            content_count=result.content_count,
            vector_count=result.vector_count,
            orphaned_vectors=result.orphaned_vectors,
            missing_vectors=result.missing_vectors,
            is_consistent=result.is_consistent,
            message=result.message,
        )
    except NotInitializedError:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. POST to /init first.",
        ) from None
    except SemStashError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.post("/sync", response_model=SyncResponse)
async def sync(
    delete_orphaned: bool = Query(True, description="Delete orphaned vectors"),
    create_missing: bool = Query(True, description="Create missing embeddings"),
) -> SyncResponse:
    """Synchronize storage consistency."""
    try:
        stash = get_stash()
        result = stash.sync(
            delete_orphaned=delete_orphaned,
            create_missing=create_missing,
        )

        return SyncResponse(
            deleted_vectors=result.deleted_vectors,
            created_vectors=result.created_vectors,
            failed_keys=result.failed_keys,
            deleted_count=result.deleted_count,
            created_count=result.created_count,
            message=result.message,
        )
    except NotInitializedError:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. POST to /init first.",
        ) from None
    except SemStashError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.delete("/destroy", response_model=DestroyResponse)
async def destroy(
    force: bool = Query(False, description="Force deletion of non-empty storage"),
) -> DestroyResponse:
    """Permanently destroy the semantic stash. WARNING: This is irreversible!"""
    try:
        stash = get_stash()
        result = stash.destroy(force=force)

        return DestroyResponse(
            bucket=result.bucket,
            vector_bucket=result.vector_bucket,
            content_deleted=result.content_deleted,
            vectors_deleted=result.vectors_deleted,
            bucket_deleted=result.bucket_deleted,
            vector_bucket_deleted=result.vector_bucket_deleted,
            destroyed=result.destroyed,
            message=result.message,
        )
    except StorageError as e:
        raise HTTPException(
            status_code=400,
            detail=f"{e} Use force=true to delete non-empty storage.",
        ) from None
    except NotInitializedError:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. POST to /init first.",
        ) from None
    except SemStashError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


# --- Web UI Routes ---


@app.get("/ui/", response_class=HTMLResponse)
async def ui_dashboard(request: Request) -> HTMLResponse:
    """Dashboard page with stats and quick actions."""
    bucket = os.environ.get("SEMSTASH_BUCKET")
    region = os.environ.get("SEMSTASH_REGION", DEFAULT_REGION)

    try:
        stash = get_stash()
        stats_result = stash.get_stats()
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "active_page": "dashboard",
                "stats": stats_result,
                "bucket": bucket,
                "region": region,
            },
        )
    except NotInitializedError:
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "active_page": "dashboard",
                "error": "Storage not initialized. Set SEMSTASH_BUCKET environment variable.",
                "bucket": bucket,
                "region": region,
            },
        )
    except SemStashError as e:
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "active_page": "dashboard",
                "error": str(e),
                "bucket": bucket,
                "region": region,
            },
        )


@app.get("/ui/upload", response_class=HTMLResponse)
async def ui_upload(request: Request) -> HTMLResponse:
    """Upload page with drag-and-drop."""
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "active_page": "upload",
        },
    )


@app.get("/ui/browse", response_class=HTMLResponse)
async def ui_browse(
    request: Request,
    prefix: str = Query("", description="Key prefix filter"),
    content_type: str | None = Query(None, description="Content type filter"),
    next_token: str | None = Query(None, description="Pagination token"),
) -> HTMLResponse:
    """Browse stored content."""
    try:
        stash = get_stash()
        result = stash.browse(
            prefix=prefix,
            content_type=content_type,
            limit=20,
            continuation_token=next_token,
        )

        return templates.TemplateResponse(
            request,
            "browse.html",
            {
                "active_page": "browse",
                "items": result.items,
                "total": result.total,
                "next_token": result.next_token,
                "prefix": prefix,
                "content_type": content_type,
            },
        )
    except NotInitializedError:
        return templates.TemplateResponse(
            request,
            "browse.html",
            {
                "active_page": "browse",
                "error": "Storage not initialized.",
                "items": [],
                "prefix": prefix,
                "content_type": content_type,
            },
        )
    except SemStashError as e:
        return templates.TemplateResponse(
            request,
            "browse.html",
            {
                "active_page": "browse",
                "error": str(e),
                "items": [],
                "prefix": prefix,
                "content_type": content_type,
            },
        )


@app.get("/ui/search", response_class=HTMLResponse)
async def ui_search(
    request: Request,
    q: str = Query("", description="Search query"),
    tag: str | None = Query(None, description="Filter by tag"),
    content_type: str | None = Query(None, description="Filter by content type"),
) -> HTMLResponse:
    """Search page with semantic query."""
    results = []
    error = None

    if q:
        try:
            stash = get_stash()
            tags_list = [tag] if tag else None
            results = stash.query(
                query_text=q,
                top_k=20,
                content_type=content_type,
                tags=tags_list,
            )
        except NotInitializedError:
            error = "Storage not initialized."
        except SemStashError as e:
            error = str(e)

    return templates.TemplateResponse(
        request,
        "search.html",
        {
            "active_page": "search",
            "query": q,
            "results": results,
            "tag": tag,
            "content_type": content_type,
            "error": error,
        },
    )


@app.get("/ui/content/{key:path}", response_class=HTMLResponse)
async def ui_content(request: Request, key: str) -> HTMLResponse:
    """Content detail page."""
    try:
        stash = get_stash()
        content = stash.get(key)

        return templates.TemplateResponse(
            request,
            "content.html",
            {
                "active_page": "browse",
                "content": content,
            },
        )
    except ContentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Content not found: {key}") from None
    except NotInitializedError:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. Set SEMSTASH_BUCKET first.",
        ) from None
    except SemStashError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


def main() -> None:
    """Run the web server."""
    import uvicorn

    host = os.environ.get("SEMSTASH_HOST", "0.0.0.0")  # nosec B104 - intentional for server
    port = int(os.environ.get("SEMSTASH_PORT", "8000"))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
