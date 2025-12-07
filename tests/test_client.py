"""Tests for SemStash client."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from semstash import SemStash
from semstash.exceptions import (
    AlreadyExistsError,
    BucketNotFoundError,
    ContentExistsError,
    ContentNotFoundError,
    NotInitializedError,
    UnsupportedContentTypeError,
)
from semstash.models import (
    BrowseResult,
    DeleteResult,
    GetResult,
    InitResult,
    UploadResult,
    UsageStats,
)
from semstash.storage import ContentStorage, VectorStorage


@pytest.fixture
def mock_content_storage(mock_s3: Any, s3_bucket: str) -> ContentStorage:
    """Create mocked content storage with bucket already created."""
    storage = ContentStorage(s3_bucket, client=mock_s3)
    storage.ensure_bucket_exists()
    return storage


@pytest.fixture
def mock_content_storage_empty(mock_s3: Any) -> ContentStorage:
    """Create mocked content storage WITHOUT creating the bucket."""
    # Don't use s3_bucket fixture - it creates the bucket!
    return ContentStorage("test-bucket", client=mock_s3)


@pytest.fixture
def mock_vector_storage(mock_s3vectors: MagicMock) -> VectorStorage:
    """Create mocked vector storage with index already created."""
    storage = VectorStorage("test-bucket-vectors", client=mock_s3vectors)
    storage.ensure_index_exists()  # Actually create the index in the mock
    return storage


@pytest.fixture
def mock_vector_storage_empty(mock_s3vectors: MagicMock) -> VectorStorage:
    """Create mocked vector storage WITHOUT creating the index."""
    return VectorStorage("test-bucket-vectors", client=mock_s3vectors)


@pytest.fixture
def mock_embedding_generator(mock_bedrock: MagicMock) -> Any:
    """Create mocked embedding generator."""
    from semstash.embeddings import EmbeddingGenerator

    return EmbeddingGenerator(client=mock_bedrock)


@pytest.fixture
def stash(
    mock_content_storage: ContentStorage,
    mock_vector_storage: VectorStorage,
    mock_embedding_generator: Any,
) -> SemStash:
    """Create SemStash client with mocked dependencies."""
    client = SemStash(
        bucket="test-bucket",
        _content_storage=mock_content_storage,
        _vector_storage=mock_vector_storage,
        _embedding_generator=mock_embedding_generator,
        auto_init=False,
    )
    client._initialized = True
    return client


class TestSemStashInit:
    """Tests for SemStash initialization."""

    def test_default_values(self) -> None:
        """Client initializes with defaults."""
        client = SemStash(bucket="test-bucket", auto_init=False)

        assert client.bucket == "test-bucket"
        assert client.region == "us-east-1"
        assert client.dimension == 3072

    def test_custom_values(self) -> None:
        """Client accepts custom values."""
        client = SemStash(
            bucket="my-bucket",
            region="us-east-2",
            dimension=1024,
            auto_init=False,
        )

        assert client.bucket == "my-bucket"
        assert client.region == "us-east-2"
        assert client.dimension == 1024

    def test_vector_bucket_derived(self) -> None:
        """Vector bucket is derived from content bucket."""
        client = SemStash(bucket="my-bucket", auto_init=False)

        assert client.vector_bucket == "my-bucket-vectors"

    def test_context_manager(self) -> None:
        """Client works as context manager."""
        with SemStash(bucket="test-bucket", auto_init=False) as client:
            assert client.bucket == "test-bucket"


class TestSemStashInitMethod:
    """Tests for the init method."""

    def test_init_creates_storage(
        self,
        mock_content_storage_empty: ContentStorage,
        mock_vector_storage_empty: VectorStorage,
        mock_embedding_generator: Any,
    ) -> None:
        """Init method creates storage when bucket doesn't exist."""
        client = SemStash(
            bucket="test-bucket",
            _content_storage=mock_content_storage_empty,
            _vector_storage=mock_vector_storage_empty,
            _embedding_generator=mock_embedding_generator,
            auto_init=False,
        )

        result = client.init()

        assert isinstance(result, InitResult)
        assert result.bucket == "test-bucket"
        assert result.dimension == 3072

    def test_init_fails_if_bucket_exists(
        self,
        mock_content_storage: ContentStorage,
        mock_vector_storage_empty: VectorStorage,
        mock_embedding_generator: Any,
    ) -> None:
        """Init method fails if bucket already exists."""
        client = SemStash(
            bucket="test-bucket",
            _content_storage=mock_content_storage,  # Bucket already exists
            _vector_storage=mock_vector_storage_empty,
            _embedding_generator=mock_embedding_generator,
            auto_init=False,
        )

        with pytest.raises(AlreadyExistsError) as exc_info:
            client.init()

        assert "already exists" in str(exc_info.value)
        assert "open()" in str(exc_info.value)

    def test_init_fails_if_index_exists(
        self,
        mock_content_storage_empty: ContentStorage,
        mock_vector_storage: VectorStorage,
        mock_embedding_generator: Any,
    ) -> None:
        """Init method fails if vector index already exists."""
        client = SemStash(
            bucket="test-bucket",
            _content_storage=mock_content_storage_empty,  # Bucket doesn't exist
            _vector_storage=mock_vector_storage,  # Index already exists
            _embedding_generator=mock_embedding_generator,
            auto_init=False,
        )

        with pytest.raises(AlreadyExistsError) as exc_info:
            client.init()

        assert "already exists" in str(exc_info.value)

    def test_init_without_bucket_raises(self) -> None:
        """Init without bucket raises error."""
        client = SemStash(auto_init=False)

        with pytest.raises(NotInitializedError) as exc_info:
            client.init()

        assert "Bucket not configured" in str(exc_info.value)


class TestSemStashOpenMethod:
    """Tests for the open method."""

    def test_open_existing_storage(
        self,
        mock_content_storage: ContentStorage,
        mock_vector_storage: VectorStorage,
        mock_embedding_generator: Any,
    ) -> None:
        """Open method opens existing storage."""
        client = SemStash(
            bucket="test-bucket",
            _content_storage=mock_content_storage,  # Bucket exists
            _vector_storage=mock_vector_storage,  # Index exists
            _embedding_generator=mock_embedding_generator,
            auto_init=False,
        )

        result = client.open()

        assert isinstance(result, InitResult)
        assert result.bucket == "test-bucket"
        assert result.dimension == 3072

    def test_open_fails_if_bucket_not_found(
        self,
        mock_content_storage_empty: ContentStorage,
        mock_vector_storage: VectorStorage,
        mock_embedding_generator: Any,
    ) -> None:
        """Open method fails if bucket doesn't exist."""
        client = SemStash(
            bucket="test-bucket",
            _content_storage=mock_content_storage_empty,  # Bucket doesn't exist
            _vector_storage=mock_vector_storage,
            _embedding_generator=mock_embedding_generator,
            auto_init=False,
        )

        with pytest.raises(BucketNotFoundError) as exc_info:
            client.open()

        assert "not found" in str(exc_info.value)
        assert "init()" in str(exc_info.value)

    def test_open_fails_if_index_not_found(
        self,
        mock_content_storage: ContentStorage,
        mock_vector_storage_empty: VectorStorage,
        mock_embedding_generator: Any,
    ) -> None:
        """Open method fails if vector index doesn't exist."""
        client = SemStash(
            bucket="test-bucket",
            _content_storage=mock_content_storage,  # Bucket exists
            _vector_storage=mock_vector_storage_empty,  # Index doesn't exist
            _embedding_generator=mock_embedding_generator,
            auto_init=False,
        )

        with pytest.raises(BucketNotFoundError) as exc_info:
            client.open()

        assert "not found" in str(exc_info.value)

    def test_open_without_bucket_raises(self) -> None:
        """Open without bucket raises error."""
        client = SemStash(auto_init=False)

        with pytest.raises(NotInitializedError) as exc_info:
            client.open()

        assert "Bucket not configured" in str(exc_info.value)


class TestSemStashUpload:
    """Tests for upload functionality."""

    def test_upload_text_file_to_root(self, stash: SemStash, sample_text_file: Path) -> None:
        """Upload text file to root."""
        result = stash.upload(sample_text_file, target="/")

        assert isinstance(result, UploadResult)
        assert result.key == sample_text_file.name
        assert result.path == f"/{sample_text_file.name}"
        assert result.content_type == "text/plain"
        assert result.file_size > 0
        assert result.dimension == 3072

    def test_upload_to_folder(self, stash: SemStash, sample_text_file: Path) -> None:
        """Upload to folder preserves filename."""
        result = stash.upload(sample_text_file, target="/docs/")

        assert result.key == f"docs/{sample_text_file.name}"
        assert result.path == f"/docs/{sample_text_file.name}"

    def test_upload_with_rename(self, stash: SemStash, sample_text_file: Path) -> None:
        """Upload with rename (no trailing slash)."""
        result = stash.upload(sample_text_file, target="/docs/readme.txt")

        assert result.key == "docs/readme.txt"
        assert result.path == "/docs/readme.txt"

    def test_upload_with_tags(self, stash: SemStash, sample_text_file: Path) -> None:
        """Upload with tags."""
        result = stash.upload(sample_text_file, target="/", tags=["test", "document"])

        assert result.key == sample_text_file.name

    def test_upload_with_metadata(self, stash: SemStash, sample_text_file: Path) -> None:
        """Upload with custom metadata."""
        result = stash.upload(
            sample_text_file,
            target="/",
            metadata={"author": "test", "version": "1.0"},
        )

        assert result.key == sample_text_file.name

    def test_upload_image(self, stash: SemStash, sample_image_file: Path) -> None:
        """Upload image file."""
        result = stash.upload(sample_image_file, target="/")

        assert result.content_type == "image/png"

    def test_upload_nonexistent_file(self, stash: SemStash, tmp_path: Path) -> None:
        """Upload nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            stash.upload(tmp_path / "nonexistent.txt", target="/")

    def test_upload_unsupported_type(self, stash: SemStash, tmp_path: Path) -> None:
        """Upload unsupported type raises error."""
        binary_file = tmp_path / "data.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        with pytest.raises(UnsupportedContentTypeError):
            stash.upload(binary_file, target="/")

    def test_upload_existing_raises_error(self, stash: SemStash, sample_text_file: Path) -> None:
        """Upload to existing key raises ContentExistsError by default."""
        # First upload succeeds
        stash.upload(sample_text_file, target="/")

        # Second upload to same path fails
        with pytest.raises(ContentExistsError) as exc_info:
            stash.upload(sample_text_file, target="/")

        assert "already exists" in str(exc_info.value)

    def test_upload_existing_with_force(self, stash: SemStash, sample_text_file: Path) -> None:
        """Upload with force=True overwrites existing content."""
        # First upload
        result1 = stash.upload(sample_text_file, target="/")

        # Second upload with force=True succeeds
        result2 = stash.upload(sample_text_file, target="/", force=True)

        assert result2.key == result1.key


class TestSemStashQuery:
    """Tests for query functionality."""

    def test_basic_query(
        self,
        stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Basic text query returns valid results with proper scores and metadata."""
        from helpers import assert_valid_query_results, assert_valid_search_result

        # Upload some content first
        stash.upload(sample_text_file, target="/")

        results = stash.query("test query", top_k=5)

        assert isinstance(results, list)
        assert_valid_query_results(results, min_count=1, expected_keys=[sample_text_file.name])

        # Verify the specific result has valid data
        matched = next(r for r in results if r.key == sample_text_file.name)
        assert_valid_search_result(
            matched,
            expected_key=sample_text_file.name,
            expected_content_type="text/plain",
            require_file_size=True,
        )

    def test_query_with_filters(
        self,
        stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Query with content type filter returns valid results."""
        from helpers import assert_valid_query_results

        stash.upload(sample_text_file, target="/")

        results = stash.query(
            "test query",
            content_type="text/plain",
            top_k=5,
        )

        assert isinstance(results, list)
        assert_valid_query_results(results, min_count=1)

    def test_query_without_urls(
        self,
        stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Query without including URLs still returns valid scores."""
        from helpers import assert_valid_query_results

        stash.upload(sample_text_file, target="/")

        results = stash.query("test query", include_url=False)

        assert isinstance(results, list)
        assert_valid_query_results(results, min_count=1)
        # URL should not be set
        for r in results:
            assert r.url is None, "URL should be None when include_url=False"

    def test_query_with_tags(
        self,
        stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Query with tag filter returns valid results."""
        from helpers import assert_valid_query_results

        stash.upload(sample_text_file, target="/", tags=["test", "sample"])

        results = stash.query("test query", tags=["test"])

        assert isinstance(results, list)
        assert_valid_query_results(results, min_count=1)

    def test_query_with_path_filter(
        self,
        stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Query with path filter returns only matching results."""
        from helpers import assert_valid_query_results

        stash.upload(sample_text_file, target="/docs/")

        # Query with path filter should find it
        results = stash.query("test query", path="/docs/")

        assert isinstance(results, list)
        assert_valid_query_results(results, min_count=1)
        # Verify path prefix
        for r in results:
            assert r.path.startswith("/docs/")


class TestBuildFilterExpression:
    """Tests for _build_filter_expression method."""

    def test_no_filters(self, stash: SemStash) -> None:
        """Returns None when no filters provided."""
        result = stash._build_filter_expression()
        assert result is None

    def test_content_type_only(self, stash: SemStash) -> None:
        """Returns $eq filter for content_type."""
        result = stash._build_filter_expression(content_type="image/jpeg")
        assert result == {"content_type": {"$eq": "image/jpeg"}}

    def test_tags_only(self, stash: SemStash) -> None:
        """Returns $in filter for tags."""
        result = stash._build_filter_expression(tags=["vacation", "beach"])
        assert result == {"tags": {"$in": ["vacation", "beach"]}}

    def test_content_type_and_tags(self, stash: SemStash) -> None:
        """Returns $and filter for both content_type and tags."""
        result = stash._build_filter_expression(
            content_type="image/jpeg",
            tags=["vacation"],
        )
        assert result == {
            "$and": [
                {"content_type": {"$eq": "image/jpeg"}},
                {"tags": {"$in": ["vacation"]}},
            ]
        }

    def test_path_filter_not_in_expression(self, stash: SemStash) -> None:
        """Path filtering is done client-side, not in filter expression."""
        # Path is not passed to _build_filter_expression since S3 Vectors
        # doesn't support prefix matching operators like $startsWith
        result = stash._build_filter_expression()
        assert result is None

    def test_all_filters_without_path(self, stash: SemStash) -> None:
        """Returns $and filter with content_type and tags (path filtered client-side)."""
        result = stash._build_filter_expression(
            content_type="text/plain",
            tags=["doc"],
        )
        assert result == {
            "$and": [
                {"content_type": {"$eq": "text/plain"}},
                {"tags": {"$in": ["doc"]}},
            ]
        }


class TestSemStashGet:
    """Tests for get functionality."""

    def test_get_existing(self, stash: SemStash, sample_text_file: Path) -> None:
        """Get existing content by path."""
        stash.upload(sample_text_file, target="/")

        result = stash.get(f"/{sample_text_file.name}")

        assert isinstance(result, GetResult)
        assert result.key == sample_text_file.name
        assert result.path == f"/{sample_text_file.name}"
        assert result.url is not None

    def test_get_nested_path(self, stash: SemStash, sample_text_file: Path) -> None:
        """Get content from nested path."""
        stash.upload(sample_text_file, target="/docs/")

        result = stash.get(f"/docs/{sample_text_file.name}")

        assert isinstance(result, GetResult)
        assert result.key == f"docs/{sample_text_file.name}"
        assert result.path == f"/docs/{sample_text_file.name}"

    def test_get_nonexistent(self, stash: SemStash) -> None:
        """Get nonexistent content raises error."""
        with pytest.raises(ContentNotFoundError):
            stash.get("/nonexistent.txt")


class TestSemStashDelete:
    """Tests for delete functionality."""

    def test_delete_existing(self, stash: SemStash, sample_text_file: Path) -> None:
        """Delete existing content by path."""
        stash.upload(sample_text_file, target="/")

        result = stash.delete(f"/{sample_text_file.name}")

        assert isinstance(result, DeleteResult)
        assert result.key == sample_text_file.name
        assert result.deleted is True

    def test_delete_nested_path(self, stash: SemStash, sample_text_file: Path) -> None:
        """Delete content from nested path."""
        stash.upload(sample_text_file, target="/docs/")

        result = stash.delete(f"/docs/{sample_text_file.name}")

        assert isinstance(result, DeleteResult)
        assert result.key == f"docs/{sample_text_file.name}"
        assert result.deleted is True


class TestSemStashBrowse:
    """Tests for browse functionality."""

    def test_browse_root(
        self, stash: SemStash, sample_text_file: Path, sample_json_file: Path
    ) -> None:
        """Browse root path."""
        stash.upload(sample_text_file, target="/")
        stash.upload(sample_json_file, target="/")

        result = stash.browse("/")

        assert isinstance(result, BrowseResult)
        assert result.total >= 0
        # Verify items have paths
        for item in result.items:
            assert item.path.startswith("/")

    def test_browse_folder(self, stash: SemStash, sample_text_file: Path) -> None:
        """Browse folder path."""
        stash.upload(sample_text_file, target="/docs/")

        result = stash.browse("/docs/")

        assert isinstance(result, BrowseResult)
        for item in result.items:
            assert item.path.startswith("/docs/")


class TestSemStashStats:
    """Tests for statistics."""

    def test_get_stats(self, stash: SemStash, sample_text_file: Path) -> None:
        """Get storage statistics."""
        stash.upload(sample_text_file, target="/")

        stats = stash.get_stats()

        assert isinstance(stats, UsageStats)
        assert stats.content_count >= 0
        assert stats.dimension == 3072


class TestSemStashCheck:
    """Tests for consistency check functionality."""

    def test_check_empty_storage(self, stash: SemStash) -> None:
        """Check empty storage is consistent."""
        from semstash.models import CheckResult

        result = stash.check()

        assert isinstance(result, CheckResult)
        assert result.content_count == 0
        assert result.vector_count == 0
        assert result.is_consistent is True
        assert len(result.orphaned_vectors) == 0
        assert len(result.missing_vectors) == 0

    def test_check_consistent_storage(self, stash: SemStash, sample_text_file: Path) -> None:
        """Check storage with matching content and vectors."""
        from semstash.models import CheckResult

        stash.upload(sample_text_file, target="/")

        result = stash.check()

        assert isinstance(result, CheckResult)
        assert result.content_count == 1
        assert result.vector_count == 1
        assert result.is_consistent is True


class TestSemStashSync:
    """Tests for sync functionality."""

    def test_sync_no_changes_needed(self, stash: SemStash, sample_text_file: Path) -> None:
        """Sync when storage is already consistent."""
        from semstash.models import SyncResult

        stash.upload(sample_text_file, target="/")

        result = stash.sync()

        assert isinstance(result, SyncResult)
        assert result.deleted_count == 0
        assert result.created_count == 0
        assert len(result.failed_keys) == 0


class TestSemStashDestroy:
    """Tests for destroy functionality."""

    def test_destroy_empty_storage(
        self,
        mock_content_storage: ContentStorage,
        mock_vector_storage: VectorStorage,
        mock_embedding_generator: Any,
    ) -> None:
        """Destroy empty storage without force."""
        from semstash.models import DestroyResult

        client = SemStash(
            bucket="test-bucket",
            _content_storage=mock_content_storage,
            _vector_storage=mock_vector_storage,
            _embedding_generator=mock_embedding_generator,
            auto_init=False,
        )

        result = client.destroy()

        assert isinstance(result, DestroyResult)
        assert result.bucket == "test-bucket"
        assert result.destroyed is True

    def test_destroy_requires_force_when_not_empty(
        self,
        stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Destroy with content requires force=True."""
        from semstash.exceptions import StorageError

        stash.upload(sample_text_file, target="/")

        with pytest.raises(StorageError) as exc_info:
            stash.destroy(force=False)

        assert "not empty" in str(exc_info.value)
        assert "force=True" in str(exc_info.value)

    def test_destroy_with_force(
        self,
        stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Destroy with force=True deletes everything."""
        from semstash.models import DestroyResult

        stash.upload(sample_text_file, target="/")

        result = stash.destroy(force=True)

        assert isinstance(result, DestroyResult)
        assert result.content_deleted >= 1
        assert result.vectors_deleted >= 1

    def test_destroy_not_found(
        self,
        mock_content_storage_empty: ContentStorage,
        mock_vector_storage_empty: VectorStorage,
        mock_embedding_generator: Any,
    ) -> None:
        """Destroy raises error if storage doesn't exist."""
        from semstash.exceptions import BucketNotFoundError

        client = SemStash(
            bucket="nonexistent-bucket",
            _content_storage=mock_content_storage_empty,
            _vector_storage=mock_vector_storage_empty,
            _embedding_generator=mock_embedding_generator,
            auto_init=False,
        )

        with pytest.raises(BucketNotFoundError):
            client.destroy()
