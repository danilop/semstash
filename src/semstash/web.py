"""FastAPI web application for semstash.

Provides a REST API and web UI for semantic storage operations.

Usage:
    # Run the server
    uvicorn semstash.web:app --reload

    # Or with Python
    python -m semstash.web
"""

import asyncio
import contextlib
import os
import time
import uuid
from pathlib import Path
from tempfile import gettempdir
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from semstash.client import SemStash
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
from semstash.models import StorageItem
from semstash.utils import (
    format_size,
    generate_breadcrumbs,
    get_cached_stash,
    get_containing_folder,
    get_filename,
    get_parent_path,
    key_to_path,
    normalize_path,
)

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

# Global stash instance (can be set by /init or /open endpoints)
_stash: SemStash | None = None


def get_stash() -> SemStash:
    """Get the SemStash instance.

    Uses either the explicitly initialized stash (via /init or /open) or
    falls back to creating one from environment variables.
    """
    global _stash
    if _stash is None:
        _stash = get_cached_stash()
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

    path: str
    key: str
    content_type: str
    file_size: int
    dimension: int


class QueryResultItem(BaseModel):
    """Single query result."""

    path: str
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

    path: str
    key: str
    content_type: str
    file_size: int
    created_at: str
    url: str


class DeleteResponse(SuccessResponse):
    """Delete endpoint response."""

    path: str
    key: str
    deleted: bool


class BrowseItem(BaseModel):
    """Browse result item."""

    path: str
    key: str
    content_type: str
    file_size: int
    created_at: str


class BrowseResponse(SuccessResponse):
    """Browse endpoint response."""

    path: str
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
    bucket: str
    vector_bucket: str
    index_name: str
    region: str


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


# --- Agent Response Models ---


class AgentSessionResponse(SuccessResponse):
    """Agent session response."""

    session_id: str
    created_at: str
    model_id: str


class AgentChatRequest(BaseModel):
    """Agent chat request."""

    message: str = Field(..., min_length=1, description="Message to send to the agent")


class AgentToolCall(BaseModel):
    """Tool call made by the agent."""

    name: str
    arguments: dict[str, Any] | None = None


class AgentChatResponse(SuccessResponse):
    """Agent chat response (non-streaming)."""

    response: str
    tool_calls: list[AgentToolCall] = []


class AgentResetResponse(SuccessResponse):
    """Agent reset response."""

    message: str


# --- Agent Session Management ---

# Default model for agent
DEFAULT_AGENT_MODEL = "us.amazon.nova-lite-v1:0"

# Session TTL in seconds (30 minutes)
AGENT_SESSION_TTL = 30 * 60


class AgentSession:
    """Container for an agent session with TTL tracking."""

    def __init__(self, session_id: str, model_id: str) -> None:
        self.session_id = session_id
        self.model_id = model_id
        self.created_at = time.time()
        self.last_activity = time.time()
        self.agent: Any = None  # Lazy initialization
        self._lock = asyncio.Lock()

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return time.time() - self.last_activity > AGENT_SESSION_TTL


# Global agent sessions store
_agent_sessions: dict[str, AgentSession] = {}


def _cleanup_expired_sessions() -> None:
    """Remove expired sessions."""
    expired = [sid for sid, session in _agent_sessions.items() if session.is_expired()]
    for sid in expired:
        session = _agent_sessions.pop(sid, None)
        if session and session.agent:
            # Agent cleanup happens via context manager in the agent module
            pass


def _get_agent_session(session_id: str) -> AgentSession:
    """Get an agent session by ID, cleaning up expired sessions first."""
    _cleanup_expired_sessions()

    session = _agent_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    session.touch()
    return session


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
    target: Annotated[
        str, Form(..., description="Target: '/' root, '/folder/' keeps name, else renames")
    ],
    tags: Annotated[str | None, Form()] = None,
    force: Annotated[bool, Form()] = False,
) -> UploadResponse:
    """Upload a file to semantic storage.

    Args:
        file: The file to upload.
        target: Target path. '/' for root, '/folder/' preserves filename, '/path/name.ext' renames.
        tags: Optional comma-separated tags.
        force: If True, overwrite existing content.
    """
    try:
        stash = get_stash()

        # Save uploaded file temporarily, preserving original filename
        original_filename = file.filename or "file"
        tmp_dir = Path(gettempdir())
        tmp_path = tmp_dir / original_filename
        content = await file.read()
        tmp_path.write_bytes(content)

        try:
            # Parse tags if provided
            tag_list = tags.split(",") if tags else None

            result = stash.upload(
                file_path=tmp_path,
                target=target,
                tags=tag_list,
                force=force,
            )

            return UploadResponse(
                path=result.path,
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
    path: str | None = Query(None, description="Filter by path prefix (e.g., /docs/)"),
) -> QueryResponse:
    """Query for content using natural language."""
    try:
        stash = get_stash()
        results = stash.query(
            query_text=q,
            top_k=top_k,
            content_type=content_type,
            tags=tag,
            path=path,
        )

        return QueryResponse(
            query=q,
            count=len(results),
            results=[
                QueryResultItem(
                    path=r.path,
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


@app.get("/content/{path:path}", response_model=GetResponse)
async def get_content(path: str) -> GetResponse:
    """Get content metadata and download URL.

    Path should include leading slash (e.g., /images/photo.jpg).
    """
    path = normalize_path(path)

    try:
        stash = get_stash()
        result = stash.get(path)

        return GetResponse(
            path=result.path,
            key=result.key,
            content_type=result.content_type,
            file_size=result.file_size,
            created_at=result.created_at.isoformat(),
            url=result.url,
        )
    except ContentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Content not found: {path}") from None
    except NotInitializedError:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. POST to /init first.",
        ) from None
    except SemStashError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.delete("/content/{path:path}", response_model=DeleteResponse)
async def delete_content(path: str) -> DeleteResponse:
    """Delete content from storage.

    Path should include leading slash (e.g., /images/photo.jpg).
    """
    path = normalize_path(path)

    try:
        stash = get_stash()
        result = stash.delete(path)

        return DeleteResponse(
            path=path,
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


@app.get("/browse/{path:path}", response_model=BrowseResponse)
async def browse(
    path: str,
    content_type: str | None = Query(None, description="Content type filter"),
    limit: int = Query(DEFAULT_BROWSE_LIMIT, ge=1, le=100, description="Max results"),
    next_token: str | None = Query(None, description="Pagination token"),
) -> BrowseResponse:
    """Browse stored content at a path.

    Path should be '/' for root or '/folder/' for a subfolder.
    """
    path = normalize_path(path)

    try:
        stash = get_stash()
        result = stash.browse(
            path=path,
            content_type=content_type,
            limit=limit,
            continuation_token=next_token,
        )

        return BrowseResponse(
            path=path,
            total=result.total,
            next_token=result.next_token,
            items=[
                BrowseItem(
                    path=item.path,
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
    """Get storage statistics and AWS resource information."""
    try:
        stash = get_stash()
        result = stash.get_stats()

        return StatsResponse(
            content_count=result.content_count,
            vector_count=result.vector_count,
            storage_bytes=result.storage_bytes,
            storage_gb=result.storage_gb,
            dimension=result.dimension,
            bucket=result.bucket,
            vector_bucket=result.vector_bucket,
            index_name=result.index_name,
            region=result.region,
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


# --- Agent Endpoints ---


@app.post("/agent/session", response_model=AgentSessionResponse)
async def create_agent_session(
    model_id: str = Form(DEFAULT_AGENT_MODEL),
) -> AgentSessionResponse:
    """Create a new agent session.

    The agent provides conversational access to the stash for searching,
    browsing, and getting summaries of stored content.
    """
    from datetime import datetime

    # Clean up expired sessions first
    _cleanup_expired_sessions()

    # Create new session
    session_id = str(uuid.uuid4())
    session = AgentSession(session_id=session_id, model_id=model_id)
    _agent_sessions[session_id] = session

    return AgentSessionResponse(
        session_id=session_id,
        created_at=datetime.fromtimestamp(session.created_at).isoformat(),
        model_id=model_id,
    )


@app.post("/agent/chat/{session_id}", response_model=AgentChatResponse)
async def agent_chat(
    session_id: str,
    request: AgentChatRequest,
) -> AgentChatResponse:
    """Send a message to the agent (non-streaming).

    Use /agent/stream/{session_id} for streaming responses.
    """
    from semstash.agent import SemStashAgent

    session = _get_agent_session(session_id)

    # Get bucket from environment
    bucket = os.environ.get("SEMSTASH_BUCKET")
    if not bucket:
        raise HTTPException(
            status_code=400,
            detail="SEMSTASH_BUCKET not set. Initialize storage first.",
        )

    try:
        # Initialize agent if needed
        async with session._lock:
            if session.agent is None:
                session.agent = SemStashAgent(
                    bucket=bucket,
                    model_id=session.model_id,
                    streaming=False,
                )
                session.agent.__enter__()

        # Send message and get response
        response = session.agent.chat(request.message)

        return AgentChatResponse(
            response=response,
            tool_calls=[],  # Tool calls tracked internally by agent
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}") from None


@app.post("/agent/stream/{session_id}")
async def agent_stream(
    session_id: str,
    request: AgentChatRequest,
) -> StreamingResponse:
    """Stream a response from the agent using Server-Sent Events (SSE).

    Returns a stream of events:
    - data: {"type": "text", "content": "..."} - Text chunks
    - data: {"type": "tool", "name": "...", "status": "start|end"} - Tool calls
    - data: {"type": "done"} - Stream complete
    - data: {"type": "error", "message": "..."} - Error occurred
    """
    import json

    from semstash.agent import SemStashAgent

    session = _get_agent_session(session_id)

    # Get bucket from environment
    bucket = os.environ.get("SEMSTASH_BUCKET")
    if not bucket:
        raise HTTPException(
            status_code=400,
            detail="SEMSTASH_BUCKET not set. Initialize storage first.",
        )

    async def event_generator() -> Any:
        """Generate SSE events from agent stream."""
        try:
            # Initialize agent if needed
            async with session._lock:
                if session.agent is None:
                    session.agent = SemStashAgent(
                        bucket=bucket,
                        model_id=session.model_id,
                        streaming=True,
                    )
                    session.agent.__enter__()

            # Stream response
            for event in session.agent.chat_stream(request.message):
                # Handle different event types from Strands
                if hasattr(event, "data"):
                    chunk = event.data
                    if isinstance(chunk, str) and chunk:
                        yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
                elif hasattr(event, "content"):
                    # Handle content blocks
                    for block in event.content:
                        if isinstance(block, dict) and "text" in block:
                            chunk = block["text"]
                            if chunk:
                                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/agent/reset/{session_id}", response_model=AgentResetResponse)
async def reset_agent_session(session_id: str) -> AgentResetResponse:
    """Reset the conversation history for an agent session."""
    session = _get_agent_session(session_id)

    if session.agent:
        session.agent.reset_conversation()

    return AgentResetResponse(message="Conversation reset successfully")


@app.delete("/agent/session/{session_id}", response_model=SuccessResponse)
async def delete_agent_session(session_id: str) -> SuccessResponse:
    """Delete an agent session."""
    session = _agent_sessions.pop(session_id, None)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    # Clean up agent if initialized
    if session.agent:
        with contextlib.suppress(Exception):
            session.agent.__exit__(None, None, None)

    return SuccessResponse()


# --- Web UI Helpers ---


class UIBrowseItem:
    """Item in browse view - either a folder or file."""

    def __init__(
        self,
        name: str,
        full_path: str,
        is_folder: bool,
        content_type: str | None = None,
        file_size: int | None = None,
        created_at: str | None = None,
    ):
        self.name = name
        self.full_path = full_path
        self.is_folder = is_folder
        self.content_type = content_type
        self.file_size = file_size
        self.created_at = created_at


def extract_folders_and_files(path: str, items: list[StorageItem]) -> list[UIBrowseItem]:
    """Extract virtual folders and files at the given path level.

    Args:
        path: Current browsing path (e.g., '/', '/docs/')
        items: List of storage items from browse result

    Returns:
        List of UIBrowseItem with folders first, then files, sorted alphabetically.
    """
    # Handle root and ensure path ends with /
    if path == "/":
        prefix = ""
    else:
        prefix = path.lstrip("/")
        if not prefix.endswith("/"):
            prefix = prefix + "/"

    prefix_len = len(prefix)
    seen_folders: set[str] = set()
    result: list[UIBrowseItem] = []

    for item in items:
        # Get the part after our prefix
        if prefix and not item.key.startswith(prefix):
            continue

        relative = item.key[prefix_len:]  # 'subdir/file.txt' or 'file.txt'

        if "/" in relative:
            # This is in a subfolder - extract folder name
            folder_name = relative.split("/")[0]
            if folder_name not in seen_folders:
                seen_folders.add(folder_name)
                if path != "/":
                    folder_path = f"{path.rstrip('/')}/{folder_name}/"
                else:
                    folder_path = f"/{folder_name}/"
                result.append(
                    UIBrowseItem(
                        name=folder_name,
                        full_path=folder_path,
                        is_folder=True,
                    )
                )
        else:
            # Direct child file
            file_path = key_to_path(item.key)
            result.append(
                UIBrowseItem(
                    name=relative,
                    full_path=file_path,
                    is_folder=False,
                    content_type=item.content_type,
                    file_size=item.file_size,
                    created_at=item.created_at.strftime("%Y-%m-%d") if item.created_at else None,
                )
            )

    # Sort: folders first, then files, alphabetically
    result.sort(key=lambda x: (not x.is_folder, x.name.lower()))
    return result


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
async def ui_upload(
    request: Request,
    target: str = Query("/", description="Default target path"),
) -> HTMLResponse:
    """Upload page with drag-and-drop and target path."""
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "active_page": "upload",
            "target": target,
        },
    )


@app.get("/ui/browse", response_class=HTMLResponse)
async def ui_browse(
    request: Request,
    path: str = Query("/", description="Path to browse"),
    content_type: str | None = Query(None, description="Content type filter"),
    next_token: str | None = Query(None, description="Pagination token"),
) -> HTMLResponse:
    """Browse stored content at a path with folder navigation."""
    path = normalize_path(path)

    # Ensure path ends with / for folder browsing (unless it's just /)
    if path != "/" and not path.endswith("/"):
        path = path + "/"

    # Generate navigation context
    breadcrumbs = generate_breadcrumbs(path)
    parent_path = get_parent_path(path)

    try:
        stash = get_stash()
        result = stash.browse(
            path=path,
            content_type=content_type,
            limit=100,  # Get more to ensure we see all folders
            continuation_token=next_token,
        )

        # Extract folders and files at this level
        items = extract_folders_and_files(path, result.items)

        return templates.TemplateResponse(
            request,
            "browse.html",
            {
                "active_page": "browse",
                "items": items,
                "total": result.total,
                "next_token": result.next_token,
                "path": path,
                "breadcrumbs": breadcrumbs,
                "parent_path": parent_path,
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
                "path": path,
                "breadcrumbs": breadcrumbs,
                "parent_path": parent_path,
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
                "path": path,
                "breadcrumbs": breadcrumbs,
                "parent_path": parent_path,
                "content_type": content_type,
            },
        )


class UISearchResult:
    """Search result with navigation context."""

    def __init__(
        self,
        path: str,
        key: str,
        score: float,
        content_type: str | None,
        file_size: int | None,
        url: str | None,
    ):
        self.path = path
        self.key = key
        self.score = score
        self.content_type = content_type
        self.file_size = file_size
        self.url = url
        self.filename = get_filename(path)
        self.containing_folder = get_containing_folder(path)


@app.get("/ui/search", response_class=HTMLResponse)
async def ui_search(
    request: Request,
    q: str = Query("", description="Search query"),
    tag: str | None = Query(None, description="Filter by tag"),
    content_type: str | None = Query(None, description="Filter by content type"),
    path: str | None = Query(None, description="Filter by path prefix"),
) -> HTMLResponse:
    """Search page with semantic query and folder navigation."""
    results: list[UISearchResult] = []
    error = None

    if q:
        try:
            stash = get_stash()
            tags_list = [tag] if tag else None
            search_results = stash.query(
                query_text=q,
                top_k=20,
                content_type=content_type,
                tags=tags_list,
                path=path,
            )
            # Wrap results with navigation context
            results = [
                UISearchResult(
                    path=r.path,
                    key=r.key,
                    score=r.score,
                    content_type=r.content_type,
                    file_size=r.file_size,
                    url=r.url,
                )
                for r in search_results
            ]
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
            "path": path,
            "error": error,
        },
    )


@app.get("/ui/chat", response_class=HTMLResponse)
async def ui_chat(request: Request) -> HTMLResponse:
    """AI agent chat interface."""
    bucket = os.environ.get("SEMSTASH_BUCKET")

    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "active_page": "chat",
            "bucket": bucket,
            "model_id": DEFAULT_AGENT_MODEL,
        },
    )


@app.get("/ui/content/{path:path}", response_class=HTMLResponse)
async def ui_content(request: Request, path: str) -> HTMLResponse:
    """Content detail page with navigation context."""
    path = normalize_path(path)

    # Generate navigation context
    containing_folder = get_containing_folder(path)
    breadcrumbs = generate_breadcrumbs(path)
    filename = get_filename(path)

    try:
        stash = get_stash()
        content = stash.get(path)

        return templates.TemplateResponse(
            request,
            "content.html",
            {
                "active_page": "browse",
                "content": content,
                "containing_folder": containing_folder,
                "breadcrumbs": breadcrumbs,
                "filename": filename,
            },
        )
    except ContentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Content not found: {path}") from None
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
