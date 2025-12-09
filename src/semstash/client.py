"""SemStash client - main entry point for semantic storage.

Provides a simple interface for storing content with semantic embeddings
and searching using natural language queries.

All content is addressed by paths (like a filesystem), with '/' as root.
Paths map 1:1 to S3 prefixes.

Example:
    from semstash import SemStash

    # Create client with bucket name
    stash = SemStash("my-bucket")

    # Upload content - target path is mandatory
    # '/' = root, '/docs/' = docs folder (preserves filename)
    result = stash.upload("photo.jpg", "/images/", tags=["vacation", "beach"])
    print(f"Stored at: {result.path}")

    # Query semantically (optionally filter by path)
    for item in stash.query("sunset on beach", top_k=5, path="/images/"):
        print(f"{item.score:.2f} - {item.path}")

    # Get content metadata by path
    content = stash.get("/images/photo.jpg")
    print(f"URL: {content.url}")

    # Download content by path
    stash.download("/images/photo.jpg", "./local-photo.jpg")

    # Browse a path
    items = stash.browse("/images/")

    # Delete content by path
    stash.delete("/images/photo.jpg")
"""

import tempfile
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from semstash.config import load_config
from semstash.embeddings import EmbeddingGenerator, is_supported_content_type
from semstash.exceptions import (
    AlreadyExistsError,
    BucketNotFoundError,
    NotInitializedError,
    StorageError,
    UnsupportedContentTypeError,
)
from semstash.models import (
    BrowseResult,
    CheckResult,
    DeleteResult,
    DestroyResult,
    GetResult,
    InitResult,
    SearchResult,
    StashConfig,
    SyncResult,
    UploadResult,
    UsageStats,
)
from semstash.storage import ContentStorage, VectorStorage
from semstash.utils import (
    key_to_path,
    make_chunk_key,
    normalize_path,
    parse_chunk_key,
    path_to_key,
    resolve_upload_target,
)


class SemStash:
    """Semantic storage client for multimodal content.

    Stores content in S3 and generates embeddings using Amazon Nova
    for semantic search across text, images, audio, and video.

    Example:
        # Simple usage - target path is required
        stash = SemStash("my-bucket")
        stash.upload("document.pdf", target="/")  # Upload to root
        results = stash.query("quarterly revenue", top_k=5)

        # With configuration
        stash = SemStash(
            bucket="my-bucket",
            region="us-east-1",
            dimension=1024,
        )

        # Context manager
        with SemStash("my-bucket") as stash:
            stash.upload("file.txt", target="/docs/")  # Upload to folder
    """

    def __init__(
        self,
        bucket: str | None = None,
        region: str | None = None,
        dimension: int | None = None,
        config_path: Path | None = None,
        auto_init: bool = True,
        _content_storage: ContentStorage | None = None,
        _vector_storage: VectorStorage | None = None,
        _embedding_generator: EmbeddingGenerator | None = None,
    ) -> None:
        """Initialize SemStash client.

        Args:
            bucket: S3 bucket name for content storage.
            region: AWS region (default: us-east-1).
            dimension: Embedding dimension (256, 384, 1024, 3072).
            config_path: Path to config file.
            auto_init: Automatically initialize storage on first use.
            _content_storage: Injected storage for testing.
            _vector_storage: Injected vector storage for testing.
            _embedding_generator: Injected generator for testing.

        Raises:
            ConfigurationError: If configuration is invalid.
            DimensionError: If dimension is not supported.
        """
        # Load configuration with overrides
        self.config = load_config(
            config_path=config_path,
            bucket=bucket,
            region=region,
            dimension=dimension,
        )

        self.auto_init = auto_init
        self._initialized = False

        # Storage components (can be injected for testing)
        self._content_storage = _content_storage
        self._vector_storage = _vector_storage
        self._embedding_generator = _embedding_generator

    def __enter__(self) -> "SemStash":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass

    @property
    def bucket(self) -> str | None:
        """Content bucket name."""
        return self.config.bucket

    @property
    def vector_bucket(self) -> str | None:
        """Vector bucket name."""
        return self.config.vector_bucket

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.config.dimension

    @property
    def region(self) -> str:
        """AWS region."""
        return self.config.region

    def _require_bucket(self) -> None:
        """Ensure bucket is configured.

        Raises:
            NotInitializedError: If bucket not configured.
        """
        if self.config.bucket is None:
            raise NotInitializedError(
                "Bucket not configured. "
                "Use SemStash('bucket-name') or set SEMSTASH_BUCKET environment variable."
            )

    def _ensure_storage_components(self) -> None:
        """Create storage components if not already injected."""
        assert self.config.bucket is not None
        assert self.config.vector_bucket is not None

        if self._content_storage is None:
            self._content_storage = ContentStorage(
                bucket=self.config.bucket,
                region=self.config.region,
            )

        if self._vector_storage is None:
            self._vector_storage = VectorStorage(
                bucket=self.config.vector_bucket,
                region=self.config.region,
                dimension=self.config.dimension,
            )

        if self._embedding_generator is None:
            self._embedding_generator = EmbeddingGenerator(
                region=self.config.region,
                dimension=self.config.dimension,
            )

    def _sync_dimension_from_storage(self) -> int:
        """Load actual dimension from storage and sync components if needed.

        Reads the stash config from S3 to determine the actual embedding dimension.
        If it differs from the current config, recreates VectorStorage and
        EmbeddingGenerator with the correct dimension.

        Returns:
            The actual dimension from storage (may differ from initial config).
        """
        assert self._content_storage is not None
        assert self._vector_storage is not None
        assert self.config.vector_bucket is not None

        # Try loading dimension from stash config, then fallback to index stats
        stash_config = self._content_storage.load_config()
        if stash_config:
            actual_dimension = stash_config.dimension
        elif self._vector_storage.index_exists():
            actual_dimension = self._vector_storage.get_stats()["dimension"]
        else:
            return self.config.dimension

        # Recreate components if dimension differs
        if actual_dimension != self.config.dimension:
            self.config.dimension = actual_dimension
            self._vector_storage = VectorStorage(
                bucket=self.config.vector_bucket,
                region=self.config.region,
                dimension=actual_dimension,
            )
            self._embedding_generator = EmbeddingGenerator(
                region=self.config.region,
                dimension=actual_dimension,
            )

        return actual_dimension

    def _ensure_initialized(self) -> None:
        """Ensure storage is initialized (lazy initialization).

        For existing stashes, loads config and updates dimension to match
        what's stored, ensuring query embeddings have correct dimensions.
        """
        if self._initialized:
            return

        self._require_bucket()
        self._ensure_storage_components()
        assert self._content_storage is not None
        assert self._vector_storage is not None

        if self.auto_init:
            self._content_storage.ensure_bucket_exists()
            self._vector_storage.ensure_index_exists()
            self._sync_dimension_from_storage()
            # Re-ensure index exists with correct dimension
            self._vector_storage.ensure_index_exists()

        self._initialized = True

    def init(self) -> InitResult:
        """Create new storage (fails if already exists).

        Creates S3 bucket and vector index. Fails if they already exist.
        Use open() to use an existing stash.

        Returns:
            InitResult with initialization details.

        Raises:
            NotInitializedError: If bucket not configured.
            AlreadyExistsError: If bucket or index already exists.
            StorageError: If initialization fails.
        """
        self._require_bucket()
        self._ensure_storage_components()
        assert self._content_storage is not None
        assert self._vector_storage is not None
        assert self.config.bucket is not None
        assert self.config.vector_bucket is not None

        # Check if bucket already exists
        if self._content_storage.bucket_exists():
            raise AlreadyExistsError(
                f"Bucket '{self.config.bucket}' already exists. "
                "Use open() to use an existing stash."
            )

        # Check if index already exists
        if self._vector_storage.index_exists():
            raise AlreadyExistsError(
                f"Vector index already exists in '{self.config.vector_bucket}'. "
                "Use open() to use an existing stash."
            )

        # Create resources
        self._content_storage.ensure_bucket_exists()
        self._vector_storage.ensure_index_exists()

        # Save stash config to S3
        stash_config = StashConfig(
            dimension=self.config.dimension,
            prefix_depth=0,  # Will be configurable in future
            region=self.config.region,
        )
        self._content_storage.save_config(stash_config)

        self._initialized = True

        return InitResult(
            bucket=self.config.bucket,
            vector_bucket=self.config.vector_bucket,
            region=self.config.region,
            dimension=self.config.dimension,
            prefix_depth=stash_config.prefix_depth,
            message=f"Created new storage in {self.config.region}",
        )

    def open(self) -> InitResult:
        """Open existing storage (fails if not found).

        Opens an existing S3 bucket and vector index. Fails if they don't exist.
        Use init() to create a new stash.

        Returns:
            InitResult with storage details.

        Raises:
            NotInitializedError: If bucket not configured.
            BucketNotFoundError: If bucket or index doesn't exist.
        """
        self._require_bucket()
        self._ensure_storage_components()
        assert self._content_storage is not None
        assert self._vector_storage is not None
        assert self.config.bucket is not None
        assert self.config.vector_bucket is not None

        # Check if bucket exists
        if not self._content_storage.bucket_exists():
            raise BucketNotFoundError(
                f"Bucket '{self.config.bucket}' not found. Use init() to create a new stash."
            )

        # Check if index exists
        if not self._vector_storage.index_exists():
            raise BucketNotFoundError(
                f"Vector index not found in '{self.config.vector_bucket}'. "
                "Use init() to create a new stash."
            )

        # Load stash config for prefix_depth (dimension handled by sync helper)
        stash_config = self._content_storage.load_config()
        prefix_depth = stash_config.prefix_depth if stash_config else 0

        # Sync dimension from storage and recreate components if needed
        self._sync_dimension_from_storage()
        self._vector_storage.ensure_index_exists()
        self._initialized = True

        return InitResult(
            bucket=self.config.bucket,
            vector_bucket=self.config.vector_bucket,
            region=self.config.region,
            dimension=self.config.dimension,
            prefix_depth=prefix_depth,
            message=f"Opened existing storage in {self.config.region}",
        )

    def upload(
        self,
        file_path: str | Path,
        target: str,
        tags: list[str] | None = None,
        metadata: dict[str, str] | None = None,
        content_type: str | None = None,
        force: bool = False,
    ) -> UploadResult:
        """Upload a file to semantic storage.

        Stores the file in S3 and generates embeddings for semantic search.
        Multi-page documents (PDF, PPTX) generate one embedding per page/slide.

        Args:
            file_path: Path to the file to upload.
            target: Target path in the stash. If ends with '/', filename is
                preserved (folder semantics). Otherwise, used as exact path
                (rename semantics).
                Examples:
                    '/' -> stores at root with original filename
                    '/docs/' -> stores in /docs/ with original filename
                    '/docs/readme.txt' -> stores as /docs/readme.txt
            tags: Optional tags for filtering.
            metadata: Optional custom metadata.
            content_type: MIME type (auto-detected if None).
            force: If True, overwrite existing content. If False (default),
                raise ContentExistsError if path already exists.

        Returns:
            UploadResult with storage details including path.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ContentExistsError: If path exists and force=False.
            UnsupportedContentTypeError: If content type cannot be embedded.
            StorageError: If upload fails.
            EmbeddingError: If embedding generation fails.
            ValueError: If target is empty.
        """
        self._ensure_initialized()

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Resolve target to key and path
        key, path = resolve_upload_target(file_path, target)

        # Generate embeddings first to know chunk count (before S3 upload)
        # This is needed to store chunk metadata in S3 for proper deletion later
        actual_content_type = content_type
        if actual_content_type is None:
            import mimetypes

            actual_content_type, _ = mimetypes.guess_type(str(file_path))
            actual_content_type = actual_content_type or "application/octet-stream"

        # Check if content type is supported for embedding
        if not is_supported_content_type(actual_content_type):
            raise UnsupportedContentTypeError(f"Cannot embed content type: {actual_content_type}")

        # Generate embeddings (may produce multiple for multi-page documents)
        # Pass bucket for async segmentation of large text/audio/video files
        file_embeddings = self._embedding_generator.embed_file_chunked(  # type: ignore
            file_path=file_path,
            content_type=actual_content_type,
            s3_bucket=self.config.bucket,
        )

        # Build S3 metadata including chunk info for deletion support
        s3_metadata = dict(metadata) if metadata else {}
        if not file_embeddings.is_single_chunk:
            # Store chunk info so delete can find all chunk vectors
            s3_metadata["x-semstash-chunks"] = str(file_embeddings.total_chunks)
            s3_metadata["x-semstash-chunk-type"] = file_embeddings.chunks[0].chunk_type.value

        # Upload to S3
        storage_item = self._content_storage.upload(  # type: ignore
            file_path=file_path,
            key=key,
            content_type=actual_content_type,
            metadata=s3_metadata if s3_metadata else None,
            force=force,
        )

        # Build base metadata for all vectors
        base_metadata: dict[str, Any] = {
            "content_type": storage_item.content_type,
            "file_size": storage_item.file_size,
            "created_at": storage_item.created_at.isoformat(),
            "path": path,  # Store path for path-based filtering
            "source_key": storage_item.key,  # Original file key (for chunk grouping)
        }
        if tags:
            base_metadata["tags"] = tags
        if metadata:
            base_metadata.update(metadata)

        # Store embeddings
        if file_embeddings.is_single_chunk:
            # Single embedding - use original key (no fragment)
            self._vector_storage.put_vector(  # type: ignore
                key=storage_item.key,
                vector=file_embeddings.chunks[0].embedding,
                metadata=base_metadata,
            )
        else:
            # Multiple embeddings - use fragment keys and batch insert
            vectors_to_store: list[tuple[str, list[float], dict[str, Any] | None]] = []

            for chunk in file_embeddings.chunks:
                # Create chunk-specific key with fragment
                chunk_key = make_chunk_key(storage_item.key, chunk.chunk_id)

                # Add chunk-specific metadata
                chunk_metadata = {
                    **base_metadata,
                    "chunk_type": chunk.chunk_type.value,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                }

                vectors_to_store.append((chunk_key, chunk.embedding, chunk_metadata))

            # Batch insert all chunk vectors
            self._vector_storage.put_vectors_batch(vectors_to_store)  # type: ignore

        return UploadResult(
            key=storage_item.key,
            path=path,
            content_type=storage_item.content_type,
            file_size=storage_item.file_size,
            dimension=self.config.dimension,
            created_at=storage_item.created_at,
        )

    def upload_directory(
        self,
        source_dir: str | Path,
        target_path: str,
        pattern: str = "**/*",
        tags: list[str] | None = None,
        force: bool = False,
    ) -> list[UploadResult]:
        """Upload all files from a directory to semantic storage.

        Recursively uploads files matching the pattern, preserving directory
        structure relative to source_dir.

        Args:
            source_dir: Local directory to upload from.
            target_path: Target path in the stash. Must end with '/'.
                Example: '/docs/' uploads to /docs/ preserving subdirectory structure.
            pattern: Glob pattern to match files (default: '**/*' for all files).
            tags: Optional tags to apply to all uploaded files.
            force: If True, overwrite existing content.

        Returns:
            List of UploadResult for each successfully uploaded file.

        Raises:
            ValueError: If target_path doesn't end with '/'.
            NotADirectoryError: If source_dir doesn't exist or isn't a directory.
            StorageError: If upload fails.

        Example:
            # Upload all markdown files from docs/ to /docs/
            results = stash.upload_directory('./docs/', '/docs/', pattern='**/*.md')
            print(f"Uploaded {len(results)} files")
        """
        self._ensure_initialized()

        source_dir = Path(source_dir)
        if not source_dir.exists() or not source_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {source_dir}")

        # Ensure target_path ends with /
        if not target_path.endswith("/"):
            raise ValueError("target_path must end with '/' for directory uploads")

        # Normalize target path
        target_path = normalize_path(target_path)

        # Find all matching files
        results: list[UploadResult] = []
        for file_path in source_dir.glob(pattern):
            if file_path.is_file():
                # Calculate relative path from source_dir
                relative_path = file_path.relative_to(source_dir)

                # Build target path: target_path + relative_path
                # e.g., target_path='/docs/', relative_path='subdir/file.txt'
                # -> full_target='/docs/subdir/file.txt'
                full_target = f"{target_path}{relative_path.as_posix()}"

                try:
                    result = self.upload(
                        file_path=file_path,
                        target=full_target,
                        tags=tags,
                        force=force,
                    )
                    results.append(result)
                except Exception:
                    # Skip files that can't be uploaded (unsupported types, etc.)
                    # TODO: Consider returning a summary of failures
                    pass

        return results

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        content_type: str | None = None,
        tags: list[str] | None = None,
        path: str | None = None,
        include_url: bool = True,
        url_expiry: int | None = None,
    ) -> list[SearchResult]:
        """Query for similar content using natural language.

        Args:
            query_text: Natural language query text.
            top_k: Maximum number of results (default: config.search_top_k).
            content_type: Filter by content type.
            tags: Filter by tags (any match).
            path: Filter by path prefix (e.g., '/docs/' to search only in docs).
            include_url: Include presigned download URLs.
            url_expiry: Presigned URL expiry in seconds (default: config.presigned_url_expiry).

        Returns:
            List of SearchResult sorted by similarity (highest first).

        Raises:
            NotInitializedError: If storage not initialized.
            EmbeddingError: If query embedding fails.
        """
        self._ensure_initialized()

        # Use config defaults if not specified
        if top_k is None:
            top_k = self.config.search_top_k
        if url_expiry is None:
            url_expiry = self.config.presigned_url_expiry

        # Normalize path filter for client-side filtering
        path_prefix = normalize_path(path) if path else None

        # Generate query embedding
        query_embedding = self._embedding_generator.embed_text(query_text)  # type: ignore

        # Build native S3 Vectors filter expression (path filtering done client-side)
        filter_expr = self._build_filter_expression(content_type=content_type, tags=tags)

        # Request more results if path filtering to ensure we get enough matches
        query_top_k = top_k * 3 if path_prefix else top_k

        # Query vectors with server-side filtering
        results = self._vector_storage.query(  # type: ignore
            vector=query_embedding,
            top_k=query_top_k,
            filter_expression=filter_expr,
        )

        # Enrich results with URLs and metadata
        enriched_results = []
        for result in results:
            # Handle chunk keys - get source key for URL generation
            source_key, chunk_id = parse_chunk_key(result.key)

            # Fill in path from vector metadata (or compute from source key)
            result.path = result.metadata.get("path", key_to_path(source_key))

            # Client-side path filtering (S3 Vectors doesn't support prefix matching)
            if path_prefix and not result.path.startswith(path_prefix):
                continue

            # Add URL if requested (use source key, not chunk key)
            if include_url:
                result.url = self._content_storage.get_presigned_url(source_key, expiry=url_expiry)  # type: ignore

            # Fill in content metadata from vector metadata
            result.content_type = result.metadata.get("content_type")
            if "file_size" in result.metadata:
                result.file_size = int(result.metadata["file_size"])
            if "created_at" in result.metadata:
                result.created_at = datetime.fromisoformat(result.metadata["created_at"])
            if "tags" in result.metadata:
                # Tags stored as list in S3 Vectors
                result.tags = result.metadata["tags"]

            # Add chunk metadata if present (for multi-page/multi-slide results)
            # This allows callers to see "Page 5 of 47" or "Slide 3 of 20"
            if chunk_id:
                result.metadata["_chunk_id"] = chunk_id
            if "chunk_type" in result.metadata:
                result.metadata["_chunk_type"] = result.metadata["chunk_type"]
            if "chunk_index" in result.metadata:
                result.metadata["_chunk_index"] = result.metadata["chunk_index"]
            if "total_chunks" in result.metadata:
                result.metadata["_total_chunks"] = result.metadata["total_chunks"]
            if "source_key" in result.metadata:
                result.metadata["_source_key"] = result.metadata["source_key"]

            enriched_results.append(result)

            # Stop once we have enough results
            if len(enriched_results) >= top_k:
                break

        return enriched_results

    def _build_filter_expression(
        self,
        content_type: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Build S3 Vectors native filter expression.

        Args:
            content_type: Filter by exact content type match.
            tags: Filter by tags (any match using $in operator).

        Returns:
            S3 Vectors filter expression dict, or None if no filters.

        Note:
            Path filtering is done client-side after query since S3 Vectors
            doesn't support prefix matching operators like $startsWith.
        """
        filters: list[dict[str, Any]] = []

        if content_type:
            # Exact match on content_type
            filters.append({"content_type": {"$eq": content_type}})

        if tags:
            # Match any of the specified tags using $in
            # S3 Vectors $in on array metadata returns true if any element matches
            filters.append({"tags": {"$in": tags}})

        if not filters:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            # Combine with $and
            return {"$and": filters}

    def get(self, path: str, url_expiry: int | None = None) -> GetResult:
        """Get content metadata and download URL.

        Args:
            path: Path of the content (e.g., '/docs/readme.txt').
            url_expiry: Presigned URL expiry in seconds (default: config.presigned_url_expiry).

        Returns:
            GetResult with metadata and presigned URL.

        Raises:
            ContentNotFoundError: If content doesn't exist.
        """
        self._ensure_initialized()

        # Use config default if not specified
        if url_expiry is None:
            url_expiry = self.config.presigned_url_expiry

        # Convert path to key
        key = path_to_key(path)
        normalized_path = normalize_path(path)

        # Get S3 metadata
        storage_item = self._content_storage.get_metadata(key)  # type: ignore

        # Get presigned URL
        url = self._content_storage.get_presigned_url(key, expiry=url_expiry)  # type: ignore

        return GetResult(
            key=storage_item.key,
            path=normalized_path,
            content_type=storage_item.content_type,
            file_size=storage_item.file_size,
            created_at=storage_item.created_at,
            tags=storage_item.tags,
            metadata=storage_item.metadata,
            url=url,
        )

    def download(self, path: str, destination: str | Path) -> Path:
        """Download content to a local file.

        Args:
            path: Path of the content (e.g., '/docs/readme.txt').
            destination: Local path to save the file. If a directory is given,
                the file is saved with its original filename.

        Returns:
            Path to the downloaded file.

        Raises:
            ContentNotFoundError: If content doesn't exist.
            StorageError: If download fails.

        Example:
            # Download to specific file
            stash.download("/photo.jpg", "./local-photo.jpg")

            # Download to directory (uses original filename)
            stash.download("/photos/beach.jpg", "./downloads/")
        """
        self._ensure_initialized()

        # Convert path to key
        key = path_to_key(path)
        destination = Path(destination)

        # If destination is a directory, use the filename from path
        if destination.is_dir():
            # Extract just the filename from the path
            filename = Path(key).name
            destination = destination / filename

        # Ensure parent directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Download from S3
        self._content_storage.download(key, destination)  # type: ignore

        return destination

    def delete(self, path: str) -> DeleteResult:
        """Delete content from storage.

        Removes both the S3 object and all associated vector embeddings.
        For chunked files (multi-page PDFs, multi-slide PPTX), deletes all
        chunk vectors.

        Args:
            path: Path of the content (e.g., '/docs/readme.txt').

        Returns:
            DeleteResult with deletion details.

        Raises:
            StorageError: If deletion fails.
        """
        self._ensure_initialized()

        # Convert path to key
        key = path_to_key(path)

        # Get S3 metadata to check for chunks
        try:
            storage_item = self._content_storage.get_metadata(key)  # type: ignore
            s3_metadata = storage_item.metadata or {}
        except Exception:
            s3_metadata = {}

        # Delete from S3
        self._content_storage.delete(key)  # type: ignore

        # Delete vectors - handle both single and chunked files
        chunks_str = s3_metadata.get("x-semstash-chunks")
        chunk_type = s3_metadata.get("x-semstash-chunk-type")

        if chunks_str and chunk_type:
            # Multi-chunk file - delete all chunk vectors
            try:
                total_chunks = int(chunks_str)
                chunk_keys = []
                for i in range(1, total_chunks + 1):
                    chunk_id = f"{chunk_type}={i}"
                    chunk_keys.append(make_chunk_key(key, chunk_id))

                if chunk_keys:
                    self._vector_storage.delete_vectors_batch(chunk_keys)  # type: ignore
            except (ValueError, TypeError):
                # Fallback: try to delete single key
                self._vector_storage.delete_vector(key)  # type: ignore
        else:
            # Single-chunk file - delete single vector
            self._vector_storage.delete_vector(key)  # type: ignore

        return DeleteResult(
            key=key,
            deleted=True,
        )

    def browse(
        self,
        path: str,
        content_type: str | None = None,
        limit: int | None = None,
        continuation_token: str | None = None,
    ) -> BrowseResult:
        """Browse stored content at a path.

        Args:
            path: Path to browse (e.g., '/' for root, '/docs/' for docs folder).
                Must be '/' or end with '/'.
            content_type: Filter by content type.
            limit: Maximum results per page (default: config.browse_limit).
            continuation_token: Token for pagination.

        Returns:
            BrowseResult with items and pagination info.
        """
        self._ensure_initialized()

        # Use config default if not specified
        if limit is None:
            limit = self.config.browse_limit

        # Convert path to S3 prefix
        prefix = path_to_key(path)

        items, next_token = self._content_storage.list_objects(  # type: ignore
            prefix=prefix,
            max_keys=limit,
            continuation_token=continuation_token,
        )

        # Filter by content type if specified
        if content_type:
            items = [i for i in items if i.content_type == content_type]

        return BrowseResult(
            items=items,
            total=len(items),
            next_token=next_token,
        )

    def get_stats(self) -> UsageStats:
        """Get storage statistics and AWS resource information.

        Returns:
            UsageStats with storage counts and AWS resource names.
        """
        self._ensure_initialized()

        items, _ = self._content_storage.list_objects(max_keys=1000)  # type: ignore
        total_size = sum(i.file_size for i in items)
        content_count = len(items)

        # These are guaranteed to be set after _ensure_initialized()
        assert self.config.bucket is not None
        assert self.config.vector_bucket is not None
        assert self.config.index_name is not None
        assert self.config.region is not None

        return UsageStats(
            content_count=content_count,
            vector_count=content_count,
            storage_bytes=total_size,
            dimension=self.config.dimension,
            bucket=self.config.bucket,
            vector_bucket=self.config.vector_bucket,
            index_name=self.config.index_name,
            region=self.config.region,
        )

    def check(self) -> CheckResult:
        """Check storage consistency.

        Compares S3 content objects with vector embeddings to identify:
        - Orphaned vectors: embeddings without corresponding S3 objects
        - Missing vectors: S3 objects without embeddings

        This is automatically called by open() to verify storage integrity.
        Can also be used standalone for diagnostics.

        Returns:
            CheckResult with consistency status and any issues found.

        Example:
            result = stash.check()
            if not result.is_consistent:
                print(f"Orphaned vectors: {len(result.orphaned_vectors)}")
                print(f"Missing vectors: {len(result.missing_vectors)}")
                # Use sync() to fix issues
                stash.sync()
        """
        self._ensure_initialized()

        # Get all content keys from S3
        content_keys = self._content_storage.list_all_keys()  # type: ignore

        # Get all vector keys from S3 Vectors
        vector_keys = self._vector_storage.list_all_keys()  # type: ignore

        # Extract source keys from vector keys (strip fragments for chunk vectors)
        # A content file "docs/file.pdf" may have vectors:
        # - "docs/file.pdf" (single-chunk)
        # - "docs/file.pdf#page=1", "docs/file.pdf#page=2", ... (multi-chunk)
        vector_source_keys: set[str] = set()
        for vkey in vector_keys:
            source_key, _ = parse_chunk_key(vkey)
            vector_source_keys.add(source_key)

        # Find orphaned vectors (vectors whose source file doesn't exist)
        # We report the actual vector keys (with fragments) for deletion
        orphaned: set[str] = set()
        for vkey in vector_keys:
            source_key, _ = parse_chunk_key(vkey)
            if source_key not in content_keys:
                orphaned.add(vkey)

        # Find missing vectors (content files without any vectors)
        missing = content_keys - vector_source_keys

        is_consistent = not orphaned and not missing

        if is_consistent:
            message = f"Storage is consistent: {len(content_keys)} items"
        else:
            issues = []
            if orphaned:
                # Count unique source files with orphaned vectors
                orphaned_sources = {parse_chunk_key(k)[0] for k in orphaned}
                issues.append(f"{len(orphaned)} orphaned vectors ({len(orphaned_sources)} files)")
            if missing:
                issues.append(f"{len(missing)} missing embeddings")
            message = f"Storage inconsistent: {', '.join(issues)}"

        return CheckResult(
            content_count=len(content_keys),
            vector_count=len(vector_keys),
            orphaned_vectors=sorted(orphaned),
            missing_vectors=sorted(missing),
            is_consistent=is_consistent,
            message=message,
        )

    def sync(
        self,
        delete_orphaned: bool = True,
        create_missing: bool = True,
    ) -> SyncResult:
        """Synchronize S3 content with vector embeddings.

        Fixes storage inconsistencies by:
        - Deleting orphaned vectors (embeddings without S3 objects)
        - Creating missing vectors (generating embeddings for S3 objects)

        Args:
            delete_orphaned: Delete vectors without corresponding S3 objects.
            create_missing: Generate embeddings for S3 objects without vectors.

        Returns:
            SyncResult with details of what was fixed.

        Example:
            # Check first
            check_result = stash.check()
            if not check_result.is_consistent:
                # Sync to fix issues
                sync_result = stash.sync()
                print(f"Deleted {sync_result.deleted_count} orphaned vectors")
                print(f"Created {sync_result.created_count} embeddings")
        """
        self._ensure_initialized()

        # First, check for inconsistencies
        check_result = self.check()

        deleted_vectors: list[str] = []
        created_vectors: list[str] = []
        failed_keys: list[str] = []

        # Delete orphaned vectors
        if delete_orphaned and check_result.orphaned_vectors:
            for key in check_result.orphaned_vectors:
                try:
                    self._vector_storage.delete_vector(key)  # type: ignore
                    deleted_vectors.append(key)
                except Exception:
                    failed_keys.append(key)

        # Create missing embeddings
        if create_missing and check_result.missing_vectors:
            for key in check_result.missing_vectors:
                try:
                    # Get content metadata first
                    storage_metadata = self._content_storage.get_metadata(key)  # type: ignore

                    # Check if content type is supported
                    if not is_supported_content_type(storage_metadata.content_type):
                        failed_keys.append(key)
                        continue

                    # Download to temp file
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        self._content_storage.download(key, Path(tmp.name))  # type: ignore

                        # Generate embeddings (may produce multiple for multi-page docs)
                        # Pass bucket for async segmentation of large text/audio/video
                        file_embeddings = self._embedding_generator.embed_file_chunked(  # type: ignore
                            Path(tmp.name),
                            content_type=storage_metadata.content_type,
                            s3_bucket=self.config.bucket,
                        )

                        # Build base metadata
                        base_metadata: dict[str, Any] = {
                            "content_type": storage_metadata.content_type,
                            "file_size": storage_metadata.file_size,
                            "created_at": storage_metadata.created_at.isoformat(),
                            "path": key_to_path(key),
                            "source_key": key,
                        }

                        # Store embeddings
                        if file_embeddings.is_single_chunk:
                            # Single embedding
                            self._vector_storage.put_vector(  # type: ignore
                                key=key,
                                vector=file_embeddings.chunks[0].embedding,
                                metadata=base_metadata,
                            )
                        else:
                            # Multiple embeddings - use fragment keys
                            vectors_to_store: list[
                                tuple[str, list[float], dict[str, Any] | None]
                            ] = []
                            for chunk in file_embeddings.chunks:
                                chunk_key = make_chunk_key(key, chunk.chunk_id)
                                chunk_metadata = {
                                    **base_metadata,
                                    "chunk_type": chunk.chunk_type.value,
                                    "chunk_index": chunk.chunk_index,
                                    "total_chunks": chunk.total_chunks,
                                }
                                vec_data = (chunk_key, chunk.embedding, chunk_metadata)
                                vectors_to_store.append(vec_data)

                            self._vector_storage.put_vectors_batch(vectors_to_store)  # type: ignore

                        # Clean up temp file
                        Path(tmp.name).unlink()

                    created_vectors.append(key)

                except Exception:
                    failed_keys.append(key)

        # Build result message
        parts = []
        if deleted_vectors:
            parts.append(f"deleted {len(deleted_vectors)} orphaned vectors")
        if created_vectors:
            parts.append(f"created {len(created_vectors)} embeddings")
        if failed_keys:
            parts.append(f"{len(failed_keys)} failures")

        message = "Sync complete: " + (", ".join(parts) if parts else "no changes needed")

        return SyncResult(
            deleted_vectors=deleted_vectors,
            created_vectors=created_vectors,
            failed_keys=failed_keys,
            deleted_count=len(deleted_vectors),
            created_count=len(created_vectors),
            message=message,
        )

    def destroy(self, force: bool = False) -> DestroyResult:
        """Permanently destroy the semantic stash.

        WARNING: This operation is irreversible!

        Deletes:
        - All content objects in the S3 bucket
        - All vector embeddings in S3 Vectors
        - The S3 bucket itself
        - The S3 Vectors bucket and index

        Args:
            force: Required when buckets contain data.
                   Set to True to confirm deletion of non-empty storage.

        Returns:
            DestroyResult with details of what was deleted.

        Raises:
            NotInitializedError: If bucket not configured.
            BucketNotFoundError: If storage doesn't exist.
            StorageError: If force=False and buckets are not empty.

        Example:
            # Check what will be destroyed
            stats = stash.get_stats()
            print(f"Will delete {stats.content_count} items")

            # Destroy (must use force=True if not empty)
            result = stash.destroy(force=True)
            if result.destroyed:
                print("Stash completely destroyed")
        """
        self._require_bucket()
        self._ensure_storage_components()
        assert self._content_storage is not None
        assert self._vector_storage is not None
        assert self.config.bucket is not None
        assert self.config.vector_bucket is not None

        # Check if resources exist
        bucket_exists = self._content_storage.bucket_exists()
        index_exists = self._vector_storage.index_exists()

        if not bucket_exists and not index_exists:
            raise BucketNotFoundError(
                f"Storage not found: neither bucket '{self.config.bucket}' "
                f"nor vector bucket '{self.config.vector_bucket}' exists."
            )

        # Initialize result tracking
        content_deleted = 0
        vectors_deleted = 0
        bucket_deleted = False
        vector_bucket_deleted = False

        # Check if buckets have content
        content_keys: set[str] = set()
        vector_keys: set[str] = set()

        if bucket_exists:
            content_keys = self._content_storage.list_all_keys()

        if index_exists:
            self._vector_storage._initialized = True  # Enable listing
            vector_keys = self._vector_storage.list_all_keys()

        has_content = bool(content_keys or vector_keys)

        if has_content and not force:
            raise StorageError(
                f"Storage is not empty ({len(content_keys)} objects, {len(vector_keys)} vectors). "
                "Use destroy(force=True) to confirm deletion."
            )

        # Delete all content from S3 (including internal .semstash/ files)
        if bucket_exists:
            content_deleted = self._content_storage.delete_all_objects()

        # Delete all vectors
        if index_exists and vector_keys:
            vectors_deleted = self._vector_storage.delete_all_vectors()

        # Delete vector index first (before vector bucket)
        if index_exists:
            with suppress(Exception):
                self._vector_storage.delete_index()

        # Delete vector bucket
        with suppress(Exception):
            self._vector_storage.delete_vector_bucket()
            vector_bucket_deleted = True

        # Delete S3 bucket
        if bucket_exists:
            with suppress(Exception):
                self._content_storage.delete_bucket()
                bucket_deleted = True

        # Mark as no longer initialized
        self._initialized = False

        # Build message
        parts = []
        if content_deleted:
            parts.append(f"{content_deleted} objects")
        if vectors_deleted:
            parts.append(f"{vectors_deleted} vectors")

        destroyed = bucket_deleted or vector_bucket_deleted
        if destroyed:
            message = "Stash destroyed"
            if parts:
                message += f": deleted {', '.join(parts)}"
        else:
            message = "Stash destruction incomplete"

        return DestroyResult(
            bucket=self.config.bucket,
            vector_bucket=self.config.vector_bucket,
            content_deleted=content_deleted,
            vectors_deleted=vectors_deleted,
            bucket_deleted=bucket_deleted,
            vector_bucket_deleted=vector_bucket_deleted,
            destroyed=destroyed,
            message=message,
        )


def create_stash_from_env(auto_init: bool = True) -> SemStash:
    """Create a SemStash instance from environment variables.

    This is a convenience factory function for servers (MCP, Web) that need
    to create a SemStash instance configured via environment variables.

    Environment variables:
        SEMSTASH_BUCKET: S3 bucket name (required for operations)
        SEMSTASH_REGION: AWS region (default: us-east-1)
        SEMSTASH_DIMENSION: Embedding dimension (default: 3072)

    Args:
        auto_init: If True, automatically open or create storage.

    Returns:
        Configured SemStash instance.

    Example:
        # In MCP or Web server
        from semstash.client import create_stash_from_env

        stash = create_stash_from_env()
    """
    import os

    from semstash.config import DEFAULT_DIMENSION, DEFAULT_REGION

    bucket = os.environ.get("SEMSTASH_BUCKET")
    region = os.environ.get("SEMSTASH_REGION", DEFAULT_REGION)
    dimension = int(os.environ.get("SEMSTASH_DIMENSION", str(DEFAULT_DIMENSION)))

    return SemStash(
        bucket=bucket,
        region=region,
        dimension=dimension,
        auto_init=auto_init,
    )
