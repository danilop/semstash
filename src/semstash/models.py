"""Pydantic data models for semstash.

All data structures used throughout the library are defined here.
Models use Pydantic for validation, serialization, and documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Supported content types for embedding."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"
    OFFICE = "office"
    SPREADSHEET = "spreadsheet"


class StashConfig(BaseModel):
    """Configuration stored in the stash's S3 bucket.

    This config is created during init and stored at .semstash/config.json
    in the S3 bucket. It contains all settings needed to use the stash.
    """

    version: str = Field(default="1.0", description="Config format version")
    dimension: int = Field(description="Embedding dimension (256, 384, 1024, 3072)")
    prefix_depth: int = Field(
        default=0,
        ge=0,
        le=16,
        description="Number of 4-char hash prefix segments for S3 keys (0-16)",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    region: str = Field(description="AWS region for this stash")


class StorageItem(BaseModel):
    """Metadata for stored content.

    Represents a single item in semantic storage.
    """

    key: str = Field(description="Unique key for this content")
    content_type: str = Field(description="MIME type: image/jpeg, text/plain, etc.")
    file_size: int = Field(description="Size in bytes")
    created_at: datetime
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Single search result with similarity score.

    Results are sorted by score (highest first).
    Score is 0-1, where 1 is exact match.

    Note: content_type, file_size, created_at may be None for
    vector-only queries. Full metadata is filled by the client.
    """

    key: str
    score: float = Field(ge=0, le=1, description="Similarity score, 1 = exact match")
    distance: float = Field(default=0.0, description="Raw distance from query vector")
    content_type: str | None = Field(default=None)
    file_size: int | None = Field(default=None)
    created_at: datetime | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    url: str | None = Field(default=None, description="Pre-signed download URL")


class UploadResult(BaseModel):
    """Result of an upload operation.

    Includes the storage key and metadata.
    """

    key: str = Field(description="Storage key for this content")
    content_type: str
    file_size: int
    dimension: int = Field(description="Embedding dimension used")
    created_at: datetime = Field(default_factory=datetime.now)


class GetResult(BaseModel):
    """Result of a get operation.

    Includes metadata and a pre-signed URL for download.
    """

    key: str
    content_type: str
    file_size: int
    created_at: datetime
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    url: str = Field(description="Pre-signed download URL")


class DeleteResult(BaseModel):
    """Result of a delete operation."""

    key: str
    deleted: bool = True


class BrowseResult(BaseModel):
    """Result of a browse operation with pagination.

    Use next_token to fetch the next page of results.
    """

    items: list[StorageItem]
    total: int = Field(description="Total items in this page")
    next_token: str | None = Field(default=None, description="Token for next page")


class InitResult(BaseModel):
    """Result of initializing a new stash.

    Returned by SemStash.init() or 'semstash init'.
    """

    bucket: str = Field(description="S3 bucket name for content")
    vector_bucket: str = Field(description="S3 Vectors bucket name")
    region: str
    dimension: int
    prefix_depth: int = Field(default=0, description="Hash prefix depth for S3 keys")
    message: str = Field(default="", description="Initialization message")


class UsageStats(BaseModel):
    """Current usage statistics for the stash."""

    content_count: int = Field(description="Number of content items stored")
    vector_count: int = Field(description="Total vectors in S3 Vectors")
    storage_bytes: int = Field(description="Total bytes stored in S3")
    dimension: int = Field(description="Embedding dimension")

    @property
    def storage_gb(self) -> float:
        """Storage in gigabytes."""
        return self.storage_bytes / (1024**3)


class CheckResult(BaseModel):
    """Result of storage consistency check.

    Compares S3 content objects with vector embeddings to identify:
    - Orphaned vectors: embeddings without corresponding S3 objects
    - Missing vectors: S3 objects without embeddings

    Example:
        result = stash.check()
        if result.is_consistent:
            print("Storage is consistent!")
        else:
            print(f"Found {len(result.orphaned_vectors)} orphaned vectors")
            print(f"Found {len(result.missing_vectors)} missing embeddings")
    """

    content_count: int = Field(description="Number of S3 content objects")
    vector_count: int = Field(description="Number of vector embeddings")
    orphaned_vectors: list[str] = Field(
        default_factory=list,
        description="Vector keys without corresponding S3 objects",
    )
    missing_vectors: list[str] = Field(
        default_factory=list,
        description="S3 object keys without embeddings",
    )
    is_consistent: bool = Field(description="True if no issues found")
    message: str = Field(default="", description="Status message")


class SyncResult(BaseModel):
    """Result of storage synchronization.

    Reports what was fixed during sync:
    - deleted_vectors: orphaned vectors that were removed
    - created_vectors: missing embeddings that were generated

    Example:
        result = stash.sync()
        print(f"Deleted {result.deleted_count} orphaned vectors")
        print(f"Created {result.created_count} missing embeddings")
    """

    deleted_vectors: list[str] = Field(
        default_factory=list,
        description="Vector keys that were deleted",
    )
    created_vectors: list[str] = Field(
        default_factory=list,
        description="S3 keys that got new embeddings",
    )
    failed_keys: list[str] = Field(
        default_factory=list,
        description="Keys that failed to process",
    )
    deleted_count: int = Field(default=0, description="Number of vectors deleted")
    created_count: int = Field(default=0, description="Number of vectors created")
    message: str = Field(default="", description="Status message")


class DestroyResult(BaseModel):
    """Result of destroying a semantic stash.

    WARNING: This operation permanently deletes:
    - The S3 bucket with all content
    - The S3 Vectors bucket with all embeddings

    This cannot be undone!

    Example:
        # Must use force=True if buckets contain data
        result = stash.destroy(force=True)
        if result.destroyed:
            print("Stash completely destroyed")
    """

    bucket: str = Field(description="S3 bucket name")
    vector_bucket: str = Field(description="S3 Vectors bucket name")
    content_deleted: int = Field(default=0, description="Number of S3 objects deleted")
    vectors_deleted: int = Field(default=0, description="Number of vectors deleted")
    bucket_deleted: bool = Field(default=False, description="Whether S3 bucket was deleted")
    vector_bucket_deleted: bool = Field(
        default=False, description="Whether vector bucket was deleted"
    )
    destroyed: bool = Field(description="True if stash was completely destroyed")
    message: str = Field(default="", description="Status message")
