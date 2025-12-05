"""Tests for web API module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from semstash.exceptions import ContentExistsError
from semstash.models import (
    BrowseResult,
    CheckResult,
    DeleteResult,
    DestroyResult,
    GetResult,
    InitResult,
    SearchResult,
    StorageItem,
    SyncResult,
    UploadResult,
    UsageStats,
)
from semstash.web import app

client = TestClient(app)


class TestWebHealth:
    """Tests for health endpoint."""

    def test_health_check(self) -> None:
        """Health endpoint returns status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestWebInit:
    """Tests for init endpoint."""

    @patch("semstash.web.SemStash")
    def test_init_success(self, mock_stash_class: MagicMock) -> None:
        """Init endpoint creates storage."""
        mock_stash = MagicMock()
        mock_stash.init.return_value = InitResult(
            bucket="test-bucket",
            vector_bucket="test-bucket-vectors",
            region="us-east-1",
            dimension=3072,
        )
        mock_stash_class.return_value = mock_stash

        response = client.post(
            "/init",
            data={"bucket": "test-bucket"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["bucket"] == "test-bucket"


class TestWebUpload:
    """Tests for upload endpoint."""

    @patch("semstash.web.get_stash")
    def test_upload_file_to_root(self, mock_get_stash: MagicMock) -> None:
        """Upload endpoint uploads file to root."""
        mock_stash = MagicMock()

        mock_stash.upload.return_value = UploadResult(
            key="test.txt",
            path="/test.txt",
            content_type="text/plain",
            file_size=100,
            dimension=3072,
        )
        mock_get_stash.return_value = mock_stash

        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"test content", "text/plain")},
            data={"target": "/"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["key"] == "test.txt"
        assert data["path"] == "/test.txt"
        # Verify target was passed
        mock_stash.upload.assert_called_once()
        call_kwargs = mock_stash.upload.call_args.kwargs
        assert call_kwargs.get("target") == "/"

    @patch("semstash.web.get_stash")
    def test_upload_file_to_folder(self, mock_get_stash: MagicMock) -> None:
        """Upload endpoint uploads file to folder preserving filename."""
        mock_stash = MagicMock()

        mock_stash.upload.return_value = UploadResult(
            key="docs/test.txt",
            path="/docs/test.txt",
            content_type="text/plain",
            file_size=100,
            dimension=3072,
        )
        mock_get_stash.return_value = mock_stash

        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"test content", "text/plain")},
            data={"target": "/docs/"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["path"] == "/docs/test.txt"
        # Verify target was passed
        mock_stash.upload.assert_called_once()
        call_kwargs = mock_stash.upload.call_args.kwargs
        assert call_kwargs.get("target") == "/docs/"

    @patch("semstash.web.get_stash")
    def test_upload_with_tags(self, mock_get_stash: MagicMock) -> None:
        """Upload endpoint accepts tags."""
        mock_stash = MagicMock()

        mock_stash.upload.return_value = UploadResult(
            key="test.txt",
            path="/test.txt",
            content_type="text/plain",
            file_size=100,
            dimension=3072,
        )
        mock_get_stash.return_value = mock_stash

        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"content", "text/plain")},
            data={"target": "/", "tags": "tag1,tag2"},
        )

        assert response.status_code == 200

    @patch("semstash.web.get_stash")
    def test_upload_existing_returns_409(self, mock_get_stash: MagicMock) -> None:
        """Upload to existing path returns 409 conflict."""
        mock_stash = MagicMock()
        mock_stash.upload.side_effect = ContentExistsError(
            "Content already exists at '/test.txt'. Use force=True to overwrite."
        )
        mock_get_stash.return_value = mock_stash

        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"test content", "text/plain")},
            data={"target": "/"},
        )

        assert response.status_code == 409
        data = response.json()
        assert "already exists" in data["detail"]
        assert "force=true" in data["detail"]

    @patch("semstash.web.get_stash")
    def test_upload_with_force(self, mock_get_stash: MagicMock) -> None:
        """Upload with force=true overwrites existing content."""
        mock_stash = MagicMock()

        mock_stash.upload.return_value = UploadResult(
            key="test.txt",
            path="/test.txt",
            content_type="text/plain",
            file_size=100,
            dimension=3072,
        )
        mock_get_stash.return_value = mock_stash

        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"test content", "text/plain")},
            data={"target": "/", "force": "true"},
        )

        assert response.status_code == 200
        mock_stash.upload.assert_called_once()
        # Verify force=True was passed
        call_kwargs = mock_stash.upload.call_args.kwargs
        assert call_kwargs.get("force") is True


class TestWebQuery:
    """Tests for query endpoint."""

    @patch("semstash.web.get_stash")
    def test_query_success(self, mock_get_stash: MagicMock) -> None:
        """Query endpoint returns results."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = [
            SearchResult(
                key="photo.jpg",
                path="/photo.jpg",
                score=0.95,
                distance=0.05,
                content_type="image/jpeg",
                file_size=1024,
                url="https://example.com/photo.jpg",
            ),
        ]
        mock_get_stash.return_value = mock_stash

        response = client.get("/query?q=sunset+beach")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert data["results"][0]["key"] == "photo.jpg"
        assert data["results"][0]["path"] == "/photo.jpg"

    @patch("semstash.web.get_stash")
    def test_query_with_filters(self, mock_get_stash: MagicMock) -> None:
        """Query endpoint accepts filters."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = []
        mock_get_stash.return_value = mock_stash

        response = client.get("/query?q=test&top_k=5&content_type=image/jpeg")

        assert response.status_code == 200

    @patch("semstash.web.get_stash")
    def test_query_with_tags(self, mock_get_stash: MagicMock) -> None:
        """Query endpoint filters by tags."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = [
            SearchResult(
                key="vacation.jpg",
                path="/vacation.jpg",
                score=0.90,
                distance=0.10,
                content_type="image/jpeg",
                file_size=2048,
                url="https://example.com/vacation.jpg",
            ),
        ]
        mock_get_stash.return_value = mock_stash

        response = client.get("/query?q=beach&tag=vacation&tag=summer")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        # Verify tags were passed to query
        mock_stash.query.assert_called_once()
        call_kwargs = mock_stash.query.call_args.kwargs
        assert call_kwargs.get("tags") == ["vacation", "summer"]

    @patch("semstash.web.get_stash")
    def test_query_with_path_filter(self, mock_get_stash: MagicMock) -> None:
        """Query endpoint filters by path prefix."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = [
            SearchResult(
                key="docs/readme.txt",
                path="/docs/readme.txt",
                score=0.85,
                distance=0.15,
                content_type="text/plain",
                file_size=512,
                url="https://example.com/docs/readme.txt",
            ),
        ]
        mock_get_stash.return_value = mock_stash

        response = client.get("/query?q=readme&path=/docs/")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["path"] == "/docs/readme.txt"
        # Verify path was passed to query
        mock_stash.query.assert_called_once()
        call_kwargs = mock_stash.query.call_args.kwargs
        assert call_kwargs.get("path") == "/docs/"


class TestWebGet:
    """Tests for get endpoint."""

    @patch("semstash.web.get_stash")
    def test_get_success(self, mock_get_stash: MagicMock) -> None:
        """Get endpoint returns content info."""
        mock_stash = MagicMock()
        mock_stash.get.return_value = GetResult(
            key="photo.jpg",
            path="/photo.jpg",
            content_type="image/jpeg",
            file_size=1024,
            created_at=datetime.now(),
            url="https://example.com/photo.jpg",
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/content/photo.jpg")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["key"] == "photo.jpg"
        assert data["path"] == "/photo.jpg"
        # Verify path was passed to get
        mock_stash.get.assert_called_once()
        call_args = mock_stash.get.call_args[0]
        assert call_args[0] == "/photo.jpg"

    @patch("semstash.web.get_stash")
    def test_get_nested_path(self, mock_get_stash: MagicMock) -> None:
        """Get endpoint handles nested paths."""
        mock_stash = MagicMock()
        mock_stash.get.return_value = GetResult(
            key="docs/report.pdf",
            path="/docs/report.pdf",
            content_type="application/pdf",
            file_size=2048,
            created_at=datetime.now(),
            url="https://example.com/docs/report.pdf",
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/content/docs/report.pdf")

        assert response.status_code == 200
        data = response.json()
        assert data["key"] == "docs/report.pdf"
        assert data["path"] == "/docs/report.pdf"
        # Verify path was passed to get
        mock_stash.get.assert_called_once()
        call_args = mock_stash.get.call_args[0]
        assert call_args[0] == "/docs/report.pdf"


class TestWebDelete:
    """Tests for delete endpoint."""

    @patch("semstash.web.get_stash")
    def test_delete_success(self, mock_get_stash: MagicMock) -> None:
        """Delete endpoint removes content by path."""
        mock_stash = MagicMock()
        mock_stash.delete.return_value = DeleteResult(key="photo.jpg", deleted=True)
        mock_get_stash.return_value = mock_stash

        response = client.delete("/content/photo.jpg")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted"] is True
        # Verify path was passed to delete
        mock_stash.delete.assert_called_once()
        call_args = mock_stash.delete.call_args[0]
        assert call_args[0] == "/photo.jpg"

    @patch("semstash.web.get_stash")
    def test_delete_nested_path(self, mock_get_stash: MagicMock) -> None:
        """Delete endpoint handles nested paths."""
        mock_stash = MagicMock()
        mock_stash.delete.return_value = DeleteResult(key="docs/readme.txt", deleted=True)
        mock_get_stash.return_value = mock_stash

        response = client.delete("/content/docs/readme.txt")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        # Verify path was passed to delete
        mock_stash.delete.assert_called_once()
        call_args = mock_stash.delete.call_args[0]
        assert call_args[0] == "/docs/readme.txt"


class TestWebBrowse:
    """Tests for browse endpoint."""

    @patch("semstash.web.get_stash")
    def test_browse_empty(self, mock_get_stash: MagicMock) -> None:
        """Browse endpoint returns empty list at root."""
        mock_stash = MagicMock()
        mock_stash.browse.return_value = BrowseResult(items=[], total=0)
        mock_get_stash.return_value = mock_stash

        response = client.get("/browse/")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 0
        # Verify path was passed
        mock_stash.browse.assert_called_once()
        call_kwargs = mock_stash.browse.call_args.kwargs
        assert call_kwargs.get("path") == "/"

    @patch("semstash.web.get_stash")
    def test_browse_with_content(self, mock_get_stash: MagicMock) -> None:
        """Browse endpoint returns content at root."""
        mock_stash = MagicMock()
        mock_stash.browse.return_value = BrowseResult(
            items=[
                StorageItem(
                    key="photo.jpg",
                    path="/photo.jpg",
                    content_type="image/jpeg",
                    file_size=1024,
                    created_at=datetime.now(),
                ),
            ],
            total=1,
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/browse/")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["path"] == "/photo.jpg"

    @patch("semstash.web.get_stash")
    def test_browse_folder(self, mock_get_stash: MagicMock) -> None:
        """Browse endpoint handles folder paths."""
        mock_stash = MagicMock()
        mock_stash.browse.return_value = BrowseResult(
            items=[
                StorageItem(
                    key="docs/readme.txt",
                    path="/docs/readme.txt",
                    content_type="text/plain",
                    file_size=512,
                    created_at=datetime.now(),
                ),
            ],
            total=1,
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/browse/docs/")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["path"] == "/docs/readme.txt"
        # Verify path was passed
        mock_stash.browse.assert_called_once()
        call_kwargs = mock_stash.browse.call_args.kwargs
        assert call_kwargs.get("path") == "/docs/"

    @patch("semstash.web.get_stash")
    def test_browse_with_pagination(self, mock_get_stash: MagicMock) -> None:
        """Browse endpoint returns next_token for pagination."""
        mock_stash = MagicMock()
        mock_stash.browse.return_value = BrowseResult(
            items=[
                StorageItem(
                    key="photo1.jpg",
                    path="/photo1.jpg",
                    content_type="image/jpeg",
                    file_size=1024,
                    created_at=datetime.now(),
                ),
            ],
            total=1,
            next_token="token123",
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/browse/?limit=1")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["next_token"] == "token123"

    @patch("semstash.web.get_stash")
    def test_browse_with_next_token(self, mock_get_stash: MagicMock) -> None:
        """Browse endpoint accepts next_token for continuation."""
        mock_stash = MagicMock()
        mock_stash.browse.return_value = BrowseResult(
            items=[
                StorageItem(
                    key="photo2.jpg",
                    path="/photo2.jpg",
                    content_type="image/jpeg",
                    file_size=2048,
                    created_at=datetime.now(),
                ),
            ],
            total=1,
            next_token=None,
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/browse/?next_token=token123")

        assert response.status_code == 200
        data = response.json()
        assert data["next_token"] is None
        # Verify the token was passed to the client
        mock_stash.browse.assert_called_once()
        call_kwargs = mock_stash.browse.call_args.kwargs
        assert call_kwargs.get("continuation_token") == "token123"


class TestWebStats:
    """Tests for stats endpoint."""

    @patch("semstash.web.get_stash")
    def test_stats_success(self, mock_get_stash: MagicMock) -> None:
        """Stats endpoint returns statistics."""
        mock_stash = MagicMock()
        mock_stash.get_stats.return_value = UsageStats(
            content_count=10,
            vector_count=10,
            storage_bytes=1024 * 1024,
            dimension=3072,
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content_count"] == 10


class TestWebCheck:
    """Tests for check endpoint."""

    @patch("semstash.web.get_stash")
    def test_check_consistent(self, mock_get_stash: MagicMock) -> None:
        """Check endpoint returns consistent status."""
        mock_stash = MagicMock()
        mock_stash.check.return_value = CheckResult(
            content_count=10,
            vector_count=10,
            orphaned_vectors=[],
            missing_vectors=[],
            is_consistent=True,
            message="Storage is consistent",
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/check")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["is_consistent"] is True
        assert data["content_count"] == 10
        assert data["vector_count"] == 10

    @patch("semstash.web.get_stash")
    def test_check_inconsistent(self, mock_get_stash: MagicMock) -> None:
        """Check endpoint returns inconsistent status."""
        mock_stash = MagicMock()
        mock_stash.check.return_value = CheckResult(
            content_count=5,
            vector_count=7,
            orphaned_vectors=["orphan1.jpg", "orphan2.jpg"],
            missing_vectors=["missing.jpg"],
            is_consistent=False,
            message="Found inconsistencies",
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/check")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["is_consistent"] is False
        assert len(data["orphaned_vectors"]) == 2
        assert len(data["missing_vectors"]) == 1


class TestWebSync:
    """Tests for sync endpoint."""

    @patch("semstash.web.get_stash")
    def test_sync_success(self, mock_get_stash: MagicMock) -> None:
        """Sync endpoint synchronizes storage."""
        mock_stash = MagicMock()
        mock_stash.sync.return_value = SyncResult(
            deleted_vectors=["orphan1.jpg"],
            created_vectors=["missing.jpg"],
            failed_keys=[],
            deleted_count=1,
            created_count=1,
            message="Sync complete",
        )
        mock_get_stash.return_value = mock_stash

        response = client.post("/sync")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] == 1
        assert data["created_count"] == 1

    @patch("semstash.web.get_stash")
    def test_sync_with_options(self, mock_get_stash: MagicMock) -> None:
        """Sync endpoint respects options."""
        mock_stash = MagicMock()
        mock_stash.sync.return_value = SyncResult(
            deleted_vectors=[],
            created_vectors=["missing.jpg"],
            failed_keys=[],
            deleted_count=0,
            created_count=1,
            message="Sync complete",
        )
        mock_get_stash.return_value = mock_stash

        response = client.post("/sync?delete_orphaned=false&create_missing=true")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_stash.sync.assert_called_with(
            delete_orphaned=False,
            create_missing=True,
        )


class TestWebDestroy:
    """Tests for destroy endpoint."""

    @patch("semstash.web.get_stash")
    def test_destroy_success(self, mock_get_stash: MagicMock) -> None:
        """Destroy endpoint removes storage."""
        mock_stash = MagicMock()
        mock_stash.destroy.return_value = DestroyResult(
            bucket="test-bucket",
            vector_bucket="test-bucket-vectors",
            content_deleted=5,
            vectors_deleted=5,
            bucket_deleted=True,
            vector_bucket_deleted=True,
            destroyed=True,
            message="Stash destroyed",
        )
        mock_get_stash.return_value = mock_stash

        response = client.delete("/destroy?force=true")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["destroyed"] is True
        assert data["content_deleted"] == 5
        mock_stash.destroy.assert_called_with(force=True)

    @patch("semstash.web.get_stash")
    def test_destroy_requires_force(self, mock_get_stash: MagicMock) -> None:
        """Destroy endpoint requires force for non-empty storage."""
        from semstash.exceptions import StorageError

        mock_stash = MagicMock()
        mock_stash.destroy.side_effect = StorageError("Storage is not empty")
        mock_get_stash.return_value = mock_stash

        response = client.delete("/destroy")

        assert response.status_code == 400
        data = response.json()
        assert "force=true" in data["detail"]


class TestWebUI:
    """Tests for web UI routes."""

    @patch("semstash.web.get_stash")
    def test_ui_dashboard(self, mock_get_stash: MagicMock) -> None:
        """Dashboard page renders with stats."""
        mock_stash = MagicMock()
        mock_stash.get_stats.return_value = UsageStats(
            content_count=10,
            vector_count=10,
            storage_bytes=1024 * 1024,
            dimension=3072,
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/ui/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "SemStash" in response.text
        assert "Dashboard" in response.text

    def test_ui_upload_page(self) -> None:
        """Upload page renders."""
        response = client.get("/ui/upload")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Upload" in response.text
        assert "dropzone" in response.text

    @patch("semstash.web.get_stash")
    def test_ui_browse_page(self, mock_get_stash: MagicMock) -> None:
        """Browse page renders with content list."""
        mock_stash = MagicMock()
        mock_stash.browse.return_value = BrowseResult(
            items=[
                StorageItem(
                    key="photo.jpg",
                    path="/photo.jpg",
                    content_type="image/jpeg",
                    file_size=1024,
                    created_at=datetime.now(),
                ),
            ],
            total=1,
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/ui/browse")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Browse" in response.text
        assert "photo.jpg" in response.text

    @patch("semstash.web.get_stash")
    def test_ui_browse_empty(self, mock_get_stash: MagicMock) -> None:
        """Browse page shows empty state."""
        mock_stash = MagicMock()
        mock_stash.browse.return_value = BrowseResult(items=[], total=0)
        mock_get_stash.return_value = mock_stash

        response = client.get("/ui/browse")

        assert response.status_code == 200
        assert "No content found" in response.text

    @patch("semstash.web.get_stash")
    def test_ui_browse_with_filters(self, mock_get_stash: MagicMock) -> None:
        """Browse page accepts filters."""
        mock_stash = MagicMock()
        mock_stash.browse.return_value = BrowseResult(items=[], total=0)
        mock_get_stash.return_value = mock_stash

        response = client.get("/ui/browse?path=/photos/&content_type=image")

        assert response.status_code == 200
        mock_stash.browse.assert_called_once()

    def test_ui_search_page(self) -> None:
        """Search page renders without query."""
        response = client.get("/ui/search")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Search" in response.text

    @patch("semstash.web.get_stash")
    def test_ui_search_with_query(self, mock_get_stash: MagicMock) -> None:
        """Search page renders with results."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = [
            SearchResult(
                key="beach.jpg",
                path="/beach.jpg",
                score=0.95,
                distance=0.05,
                content_type="image/jpeg",
                file_size=2048,
                url="https://example.com/beach.jpg",
            ),
        ]
        mock_get_stash.return_value = mock_stash

        response = client.get("/ui/search?q=sunset+beach")

        assert response.status_code == 200
        assert "beach.jpg" in response.text
        assert "95%" in response.text  # Score display

    @patch("semstash.web.get_stash")
    def test_ui_search_no_results(self, mock_get_stash: MagicMock) -> None:
        """Search page shows no results message."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = []
        mock_get_stash.return_value = mock_stash

        response = client.get("/ui/search?q=nonexistent")

        assert response.status_code == 200
        assert "No results found" in response.text

    @patch("semstash.web.get_stash")
    def test_ui_content_page(self, mock_get_stash: MagicMock) -> None:
        """Content detail page renders."""
        mock_stash = MagicMock()
        mock_stash.get.return_value = GetResult(
            key="photo.jpg",
            path="/photo.jpg",
            content_type="image/jpeg",
            file_size=1024,
            created_at=datetime.now(),
            url="https://example.com/photo.jpg",
            tags=["vacation", "beach"],
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/ui/content/photo.jpg")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "photo.jpg" in response.text
        assert "image/jpeg" in response.text
        assert "vacation" in response.text
        assert "beach" in response.text

    @patch("semstash.web.get_stash")
    def test_ui_content_nested_path(self, mock_get_stash: MagicMock) -> None:
        """Content detail page handles nested paths."""
        mock_stash = MagicMock()
        mock_stash.get.return_value = GetResult(
            key="docs/report.pdf",
            path="/docs/report.pdf",
            content_type="application/pdf",
            file_size=2048,
            created_at=datetime.now(),
            url="https://example.com/docs/report.pdf",
        )
        mock_get_stash.return_value = mock_stash

        response = client.get("/ui/content/docs/report.pdf")

        assert response.status_code == 200
        assert "docs/report.pdf" in response.text

    @patch("semstash.web.get_stash")
    def test_ui_content_not_found(self, mock_get_stash: MagicMock) -> None:
        """Content detail page handles missing content."""
        from semstash.exceptions import ContentNotFoundError

        mock_stash = MagicMock()
        mock_stash.get.side_effect = ContentNotFoundError("Not found")
        mock_get_stash.return_value = mock_stash

        response = client.get("/ui/content/missing.jpg")

        assert response.status_code == 404
