"""Tests for storage module."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from semstash.exceptions import (
    ContentExistsError,
    ContentNotFoundError,
    DimensionError,
    NotInitializedError,
)
from semstash.models import StorageItem
from semstash.storage import ContentStorage, VectorStorage


class TestContentStorageInit:
    """Tests for ContentStorage initialization."""

    def test_default_values(self, mock_s3: Any) -> None:
        """Storage initializes with defaults."""
        storage = ContentStorage("test-bucket", client=mock_s3)

        assert storage.bucket == "test-bucket"
        assert storage.region == "us-east-1"

    def test_custom_region(self, mock_s3: Any) -> None:
        """Storage accepts custom region."""
        storage = ContentStorage("test-bucket", region="us-west-2", client=mock_s3)

        assert storage.region == "us-west-2"


class TestContentStorageBucket:
    """Tests for bucket operations."""

    def test_create_bucket(self, mock_s3: Any) -> None:
        """Bucket is created if it doesn't exist."""
        storage = ContentStorage("new-bucket", client=mock_s3)

        created = storage.ensure_bucket_exists()

        assert created is True
        storage._bucket_verified = False  # Reset for next check
        created = storage.ensure_bucket_exists()
        assert created is False  # Already exists

    def test_bucket_already_exists(self, mock_s3: Any, s3_bucket: str) -> None:
        """Existing bucket is detected."""
        storage = ContentStorage(s3_bucket, client=mock_s3)

        created = storage.ensure_bucket_exists()

        assert created is False


class TestContentStorageUpload:
    """Tests for file upload operations."""

    def test_upload_text_file(self, mock_s3: Any, s3_bucket: str, sample_text_file: Path) -> None:
        """Upload text file to S3."""
        storage = ContentStorage(s3_bucket, client=mock_s3)

        item = storage.upload(sample_text_file)

        assert isinstance(item, StorageItem)
        assert item.key == sample_text_file.name
        assert item.content_type == "text/plain"
        assert item.file_size > 0

    def test_upload_with_custom_key(
        self, mock_s3: Any, s3_bucket: str, sample_text_file: Path
    ) -> None:
        """Upload with custom key."""
        storage = ContentStorage(s3_bucket, client=mock_s3)

        item = storage.upload(sample_text_file, key="custom/path/file.txt")

        assert item.key == "custom/path/file.txt"

    def test_upload_with_metadata(
        self, mock_s3: Any, s3_bucket: str, sample_text_file: Path
    ) -> None:
        """Upload with custom metadata."""
        storage = ContentStorage(s3_bucket, client=mock_s3)
        metadata = {"author": "test", "version": "1.0"}

        item = storage.upload(sample_text_file, metadata=metadata)

        assert item.metadata == metadata

    def test_upload_nonexistent_file(self, mock_s3: Any, s3_bucket: str, tmp_path: Path) -> None:
        """Upload nonexistent file raises error."""
        storage = ContentStorage(s3_bucket, client=mock_s3)

        with pytest.raises(FileNotFoundError):
            storage.upload(tmp_path / "nonexistent.txt")

    def test_upload_image(self, mock_s3: Any, s3_bucket: str, sample_image_file: Path) -> None:
        """Upload image file."""
        storage = ContentStorage(s3_bucket, client=mock_s3)

        item = storage.upload(sample_image_file)

        assert item.content_type == "image/png"

    def test_upload_existing_raises_error(
        self, mock_s3: Any, s3_bucket: str, sample_text_file: Path
    ) -> None:
        """Upload to existing key raises ContentExistsError by default."""
        storage = ContentStorage(s3_bucket, client=mock_s3)

        # First upload succeeds
        storage.upload(sample_text_file)

        # Second upload to same key fails
        with pytest.raises(ContentExistsError) as exc_info:
            storage.upload(sample_text_file)

        assert "already exists" in str(exc_info.value)
        assert "force=True" in str(exc_info.value)

    def test_upload_existing_with_force(
        self, mock_s3: Any, s3_bucket: str, sample_text_file: Path
    ) -> None:
        """Upload with force=True overwrites existing content."""
        storage = ContentStorage(s3_bucket, client=mock_s3)

        # First upload
        item1 = storage.upload(sample_text_file)

        # Second upload with force=True succeeds
        item2 = storage.upload(sample_text_file, force=True)

        assert item2.key == item1.key
        assert storage.exists(item1.key)


class TestContentStorageDownload:
    """Tests for file download operations."""

    def test_download_file(
        self, mock_s3: Any, s3_bucket: str, sample_text_file: Path, tmp_path: Path
    ) -> None:
        """Download file from S3."""
        storage = ContentStorage(s3_bucket, client=mock_s3)
        storage.upload(sample_text_file)

        dest = tmp_path / "downloaded.txt"
        result = storage.download(sample_text_file.name, dest)

        assert result == dest
        assert dest.exists()
        assert dest.read_text() == sample_text_file.read_text()

    def test_download_nonexistent(self, mock_s3: Any, s3_bucket: str, tmp_path: Path) -> None:
        """Download nonexistent file raises error."""
        storage = ContentStorage(s3_bucket, client=mock_s3)

        with pytest.raises(ContentNotFoundError):
            storage.download("nonexistent.txt", tmp_path / "dest.txt")


class TestContentStorageDelete:
    """Tests for file deletion operations."""

    def test_delete_file(self, mock_s3: Any, s3_bucket: str, sample_text_file: Path) -> None:
        """Delete file from S3."""
        storage = ContentStorage(s3_bucket, client=mock_s3)
        storage.upload(sample_text_file)

        result = storage.delete(sample_text_file.name)

        assert result is True
        assert storage.exists(sample_text_file.name) is False


class TestContentStoragePresignedUrl:
    """Tests for presigned URL generation."""

    def test_generate_url(self, mock_s3: Any, s3_bucket: str, sample_text_file: Path) -> None:
        """Generate presigned URL."""
        storage = ContentStorage(s3_bucket, client=mock_s3)
        storage.upload(sample_text_file)

        url = storage.get_presigned_url(sample_text_file.name)

        assert isinstance(url, str)
        assert s3_bucket in url


class TestContentStorageMetadata:
    """Tests for metadata operations."""

    def test_get_metadata(self, mock_s3: Any, s3_bucket: str, sample_text_file: Path) -> None:
        """Get object metadata."""
        storage = ContentStorage(s3_bucket, client=mock_s3)
        storage.upload(sample_text_file)

        item = storage.get_metadata(sample_text_file.name)

        assert isinstance(item, StorageItem)
        assert item.key == sample_text_file.name
        assert item.file_size > 0

    def test_get_metadata_nonexistent(self, mock_s3: Any, s3_bucket: str) -> None:
        """Get metadata for nonexistent object raises error."""
        storage = ContentStorage(s3_bucket, client=mock_s3)

        with pytest.raises(ContentNotFoundError):
            storage.get_metadata("nonexistent.txt")


class TestContentStorageList:
    """Tests for listing operations."""

    def test_list_objects(
        self, mock_s3: Any, s3_bucket: str, sample_text_file: Path, sample_json_file: Path
    ) -> None:
        """List objects in bucket."""
        storage = ContentStorage(s3_bucket, client=mock_s3)
        storage.upload(sample_text_file)
        storage.upload(sample_json_file)

        items, token = storage.list_objects()

        assert len(items) == 2
        assert token is None  # No pagination needed

    def test_list_with_prefix(self, mock_s3: Any, s3_bucket: str, sample_text_file: Path) -> None:
        """List objects with prefix filter."""
        storage = ContentStorage(s3_bucket, client=mock_s3)
        storage.upload(sample_text_file, key="docs/file.txt")
        storage.upload(sample_text_file, key="images/file.txt")

        items, _ = storage.list_objects(prefix="docs/")

        assert len(items) == 1
        assert items[0].key == "docs/file.txt"

    def test_list_empty_bucket(self, mock_s3: Any, s3_bucket: str) -> None:
        """List objects in empty bucket."""
        storage = ContentStorage(s3_bucket, client=mock_s3)

        items, token = storage.list_objects()

        assert items == []
        assert token is None


class TestContentStorageExists:
    """Tests for existence check."""

    def test_exists_true(self, mock_s3: Any, s3_bucket: str, sample_text_file: Path) -> None:
        """Object exists returns True."""
        storage = ContentStorage(s3_bucket, client=mock_s3)
        storage.upload(sample_text_file)

        assert storage.exists(sample_text_file.name) is True

    def test_exists_false(self, mock_s3: Any, s3_bucket: str) -> None:
        """Nonexistent object returns False."""
        storage = ContentStorage(s3_bucket, client=mock_s3)

        assert storage.exists("nonexistent.txt") is False


class TestVectorStorageInit:
    """Tests for VectorStorage initialization."""

    def test_default_values(self, mock_s3vectors: MagicMock) -> None:
        """Storage initializes with defaults."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)

        assert storage.bucket == "test-vectors"
        assert storage.region == "us-east-1"
        assert storage.dimension == 3072
        assert storage.index_name == "default-index"

    def test_custom_dimension(self, mock_s3vectors: MagicMock) -> None:
        """Storage accepts custom dimension."""
        storage = VectorStorage("test-vectors", dimension=1024, client=mock_s3vectors)

        assert storage.dimension == 1024

    def test_invalid_dimension(self, mock_s3vectors: MagicMock) -> None:
        """Invalid dimension raises error."""
        with pytest.raises(DimensionError):
            VectorStorage("test-vectors", dimension=512, client=mock_s3vectors)


class TestVectorStorageIndex:
    """Tests for index operations."""

    def test_create_index(self, mock_s3vectors: MagicMock) -> None:
        """Index is created if it doesn't exist."""
        storage = VectorStorage("new-vectors", client=mock_s3vectors)

        created = storage.ensure_index_exists()

        assert created is True
        mock_s3vectors.create_vector_bucket.assert_called()
        mock_s3vectors.create_index.assert_called()


class TestVectorStoragePutVector:
    """Tests for vector storage operations."""

    def test_put_vector(self, mock_s3vectors: MagicMock) -> None:
        """Store vector."""
        storage = VectorStorage("test-vectors", dimension=3072, client=mock_s3vectors)
        storage._initialized = True  # Simulate initialization

        storage.put_vector("doc-1", [0.1] * 3072)

        mock_s3vectors.put_vectors.assert_called()

    def test_put_vector_with_metadata(self, mock_s3vectors: MagicMock) -> None:
        """Store vector with metadata."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)
        storage._initialized = True

        metadata = {"title": "Test Document"}
        storage.put_vector("doc-1", [0.1] * 3072, metadata=metadata)

        mock_s3vectors.put_vectors.assert_called()

    def test_put_vector_not_initialized(self, mock_s3vectors: MagicMock) -> None:
        """Put vector without initialization raises error."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)

        with pytest.raises(NotInitializedError):
            storage.put_vector("doc-1", [0.1] * 3072)


class TestVectorStorageQuery:
    """Tests for vector query operations."""

    def test_query(self, mock_s3vectors: MagicMock) -> None:
        """Query for similar vectors."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)
        storage._initialized = True

        results = storage.query([0.1] * 3072, top_k=5)

        assert isinstance(results, list)
        mock_s3vectors.query_vectors.assert_called()

    def test_query_not_initialized(self, mock_s3vectors: MagicMock) -> None:
        """Query without initialization raises error."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)

        with pytest.raises(NotInitializedError):
            storage.query([0.1] * 3072)


class TestVectorStorageDelete:
    """Tests for vector deletion operations."""

    def test_delete_vector(self, mock_s3vectors: MagicMock) -> None:
        """Delete vector."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)
        storage._initialized = True

        result = storage.delete_vector("doc-1")

        assert result is True
        mock_s3vectors.delete_vectors.assert_called()

    def test_delete_vectors_batch(self, mock_s3vectors: MagicMock) -> None:
        """Delete multiple vectors in batch."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)
        storage._initialized = True

        result = storage.delete_vectors_batch(["doc-1", "doc-2", "doc-3"])

        assert result == 3
        mock_s3vectors.delete_vectors.assert_called_once()

    def test_delete_vectors_batch_empty(self, mock_s3vectors: MagicMock) -> None:
        """Delete empty batch returns 0."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)
        storage._initialized = True

        result = storage.delete_vectors_batch([])

        assert result == 0
        mock_s3vectors.delete_vectors.assert_not_called()

    def test_delete_all_vectors(self, mock_s3vectors: MagicMock) -> None:
        """Delete all vectors in index."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)
        storage._initialized = True

        # Add some vectors first
        storage.put_vector("doc-1", [0.1] * 3072)
        storage.put_vector("doc-2", [0.2] * 3072)

        result = storage.delete_all_vectors()

        assert result == 2

    def test_delete_all_vectors_empty_index(self, mock_s3vectors: MagicMock) -> None:
        """Delete all vectors in empty index returns 0."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)
        storage._initialized = True

        result = storage.delete_all_vectors()

        assert result == 0

    def test_delete_all_vectors_not_initialized(self, mock_s3vectors: MagicMock) -> None:
        """Delete all vectors without initialization raises error."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)

        with pytest.raises(NotInitializedError):
            storage.delete_all_vectors()


class TestVectorStorageListKeys:
    """Tests for listing vector keys."""

    def test_list_all_keys(self, mock_s3vectors: MagicMock) -> None:
        """List all vector keys in index."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)
        storage._initialized = True

        # Add some vectors
        storage.put_vector("doc-1", [0.1] * 3072)
        storage.put_vector("doc-2", [0.2] * 3072)

        keys = storage.list_all_keys()

        assert isinstance(keys, set)
        assert "doc-1" in keys
        assert "doc-2" in keys

    def test_list_all_keys_empty(self, mock_s3vectors: MagicMock) -> None:
        """List keys in empty index returns empty set."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)
        storage._initialized = True

        keys = storage.list_all_keys()

        assert keys == set()

    def test_list_all_keys_not_initialized(self, mock_s3vectors: MagicMock) -> None:
        """List keys without initialization raises error."""
        storage = VectorStorage("test-vectors", client=mock_s3vectors)

        with pytest.raises(NotInitializedError):
            storage.list_all_keys()
