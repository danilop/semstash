"""semstash - Unlimited semantic storage for humans and AI agents.

Store any content (text, images, audio, video) and search semantically.
Uses Amazon S3, S3 Vectors, and Nova Multimodal Embeddings.

Example:
    from semstash import SemStash

    # Create client with bucket name
    stash = SemStash("my-bucket")

    # Upload content (target path is required)
    result = stash.upload("photo.jpg", target="/")  # Upload to root
    print(f"Stored at: {result.path}")  # /photo.jpg

    result = stash.upload("notes.txt", target="/docs/")  # Upload to folder
    print(f"Stored at: {result.path}")  # /docs/notes.txt

    # Query semantically
    for item in stash.query("sunset on beach", top_k=5):
        print(f"{item.score:.2f} - {item.path}")

    # Query with path filter
    for item in stash.query("meeting notes", path="/docs/"):
        print(f"{item.path}: {item.score:.2f}")

    # Get content metadata by path
    content = stash.get("/photo.jpg")
    print(f"URL: {content.url}")

    # Download content by path
    stash.download("/photo.jpg", "./local-photo.jpg")
"""

__version__ = "0.1.0"

# Core client
from semstash.client import SemStash, create_stash_from_env

# Configuration
from semstash.config import SUPPORTED_DIMENSIONS, Config, load_config

# Embeddings
from semstash.embeddings import (
    EmbeddingGenerator,
    get_supported_content_types,
    is_supported_content_type,
)

# Exceptions
from semstash.exceptions import (
    AlreadyExistsError,
    BucketNotFoundError,
    ConfigurationError,
    ContentExistsError,
    ContentNotFoundError,
    DimensionError,
    EmbeddingError,
    NotInitializedError,
    RegionNotSupportedError,
    SemStashError,
    StorageError,
    UnsupportedContentTypeError,
)

# Models
from semstash.models import (
    BrowseResult,
    CheckResult,
    ContentType,
    DeleteResult,
    DestroyResult,
    GetResult,
    InitResult,
    SearchResult,
    StashConfig,
    StorageItem,
    SyncResult,
    UploadResult,
    UsageStats,
)

# Storage
from semstash.storage import ContentStorage, VectorStorage

__all__ = [
    # Version
    "__version__",
    # Main client
    "SemStash",
    "create_stash_from_env",
    # Exceptions
    "SemStashError",
    "ConfigurationError",
    "NotInitializedError",
    "BucketNotFoundError",
    "ContentNotFoundError",
    "ContentExistsError",
    "EmbeddingError",
    "UnsupportedContentTypeError",
    "RegionNotSupportedError",
    "DimensionError",
    "StorageError",
    "AlreadyExistsError",
    # Models
    "ContentType",
    "StashConfig",
    "StorageItem",
    "SearchResult",
    "UploadResult",
    "GetResult",
    "DeleteResult",
    "BrowseResult",
    "InitResult",
    "CheckResult",
    "SyncResult",
    "DestroyResult",
    "UsageStats",
    # Configuration
    "Config",
    "load_config",
    "SUPPORTED_DIMENSIONS",
    # Embeddings
    "EmbeddingGenerator",
    "get_supported_content_types",
    "is_supported_content_type",
    # Storage
    "ContentStorage",
    "VectorStorage",
]
