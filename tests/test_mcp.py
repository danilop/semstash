"""Tests for MCP server module using FastMCP."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semstash.mcp_server import (
    browse,
    delete,
    get,
    init,
    query,
    stats,
    upload,
)
from semstash.models import (
    BrowseResult,
    DeleteResult,
    GetResult,
    InitResult,
    SearchResult,
    StorageItem,
    UploadResult,
    UsageStats,
)


class TestMCPInit:
    """Tests for init tool."""

    @patch("semstash.mcp_server.get_cached_stash")
    def test_init_success(self, mock_get_stash: MagicMock) -> None:
        """Init tool initializes storage."""
        mock_stash = MagicMock()
        mock_stash.init.return_value = InitResult(
            bucket="test-bucket",
            vector_bucket="test-bucket-vectors",
            region="us-east-1",
            dimension=3072,
        )
        mock_get_stash.return_value = mock_stash

        result = init()

        data = json.loads(result)
        assert data["bucket"] == "test-bucket"
        assert data["vector_bucket"] == "test-bucket-vectors"
        assert data["region"] == "us-east-1"
        assert data["dimension"] == 3072
        assert data["message"] == "Storage initialized successfully"


class TestMCPUpload:
    """Tests for upload tool."""

    @patch("semstash.mcp_server.get_cached_stash")
    def test_upload_success(self, mock_get_stash: MagicMock, sample_text_file: Path) -> None:
        """Upload tool uploads file."""
        mock_stash = MagicMock()
        mock_stash.upload.return_value = UploadResult(
            key=sample_text_file.name,
            content_type="text/plain",
            file_size=100,
            dimension=3072,
        )
        mock_get_stash.return_value = mock_stash

        result = upload(file_path=str(sample_text_file))

        data = json.loads(result)
        assert data["key"] == sample_text_file.name
        assert data["content_type"] == "text/plain"

    def test_upload_file_not_found(self) -> None:
        """Upload tool raises exception for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            upload(file_path="/nonexistent/file.txt")

    @patch("semstash.mcp_server.get_cached_stash")
    def test_upload_with_force(self, mock_get_stash: MagicMock, sample_text_file: Path) -> None:
        """Upload with force=True overwrites existing content."""
        mock_stash = MagicMock()
        mock_stash.upload.return_value = UploadResult(
            key=sample_text_file.name,
            content_type="text/plain",
            file_size=100,
            dimension=3072,
        )
        mock_get_stash.return_value = mock_stash

        result = upload(file_path=str(sample_text_file), force=True)

        data = json.loads(result)
        assert data["key"] == sample_text_file.name
        # Verify force=True was passed
        mock_stash.upload.assert_called_once()
        call_kwargs = mock_stash.upload.call_args.kwargs
        assert call_kwargs.get("force") is True


class TestMCPQuery:
    """Tests for query tool."""

    @patch("semstash.mcp_server.get_cached_stash")
    def test_query_success(self, mock_get_stash: MagicMock) -> None:
        """Query tool finds results."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = [
            SearchResult(
                key="photo.jpg",
                score=0.95,
                distance=0.05,
                content_type="image/jpeg",
                file_size=1024,
                url="https://example.com/photo.jpg",
            ),
        ]
        mock_get_stash.return_value = mock_stash

        result = query(query_text="sunset beach")

        data = json.loads(result)
        assert data["query"] == "sunset beach"
        assert data["count"] == 1
        assert data["results"][0]["key"] == "photo.jpg"

    @patch("semstash.mcp_server.get_cached_stash")
    def test_query_empty_results(self, mock_get_stash: MagicMock) -> None:
        """Query tool handles no results."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = []
        mock_get_stash.return_value = mock_stash

        result = query(query_text="nonexistent content")

        data = json.loads(result)
        assert data["count"] == 0
        assert data["results"] == []

    @patch("semstash.mcp_server.get_cached_stash")
    def test_query_with_tags(self, mock_get_stash: MagicMock) -> None:
        """Query tool filters by tags."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = [
            SearchResult(
                key="vacation.jpg",
                score=0.90,
                distance=0.10,
                content_type="image/jpeg",
                file_size=2048,
                url="https://example.com/vacation.jpg",
            ),
        ]
        mock_get_stash.return_value = mock_stash

        result = query(query_text="beach", tags=["vacation", "summer"])

        data = json.loads(result)
        assert data["count"] == 1
        # Verify tags were passed to query
        mock_stash.query.assert_called_once()
        call_kwargs = mock_stash.query.call_args.kwargs
        assert call_kwargs.get("tags") == ["vacation", "summer"]


class TestMCPGet:
    """Tests for get tool."""

    @patch("semstash.mcp_server.get_cached_stash")
    def test_get_success(self, mock_get_stash: MagicMock) -> None:
        """Get tool gets content."""
        from datetime import datetime

        mock_stash = MagicMock()
        mock_stash.get.return_value = GetResult(
            key="photo.jpg",
            content_type="image/jpeg",
            file_size=1024,
            created_at=datetime.now(),
            url="https://example.com/photo.jpg",
        )
        mock_get_stash.return_value = mock_stash

        result = get(key="photo.jpg")

        data = json.loads(result)
        assert data["key"] == "photo.jpg"
        assert data["content_type"] == "image/jpeg"
        assert "url" in data


class TestMCPDelete:
    """Tests for delete tool."""

    @patch("semstash.mcp_server.get_cached_stash")
    def test_delete_success(self, mock_get_stash: MagicMock) -> None:
        """Delete tool removes content."""
        mock_stash = MagicMock()
        mock_stash.delete.return_value = DeleteResult(key="photo.jpg", deleted=True)
        mock_get_stash.return_value = mock_stash

        result = delete(key="photo.jpg")

        data = json.loads(result)
        assert data["key"] == "photo.jpg"
        assert data["deleted"] is True


class TestMCPBrowse:
    """Tests for browse tool."""

    @patch("semstash.mcp_server.get_cached_stash")
    def test_browse_success(self, mock_get_stash: MagicMock) -> None:
        """Browse tool lists content."""
        from datetime import datetime

        mock_stash = MagicMock()
        mock_stash.browse.return_value = BrowseResult(
            items=[
                StorageItem(
                    key="photo.jpg",
                    content_type="image/jpeg",
                    file_size=1024,
                    created_at=datetime.now(),
                ),
            ],
            total=1,
        )
        mock_get_stash.return_value = mock_stash

        result = browse()

        data = json.loads(result)
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["key"] == "photo.jpg"

    @patch("semstash.mcp_server.get_cached_stash")
    def test_browse_with_prefix(self, mock_get_stash: MagicMock) -> None:
        """Browse tool filters by prefix."""
        from datetime import datetime

        mock_stash = MagicMock()
        mock_stash.browse.return_value = BrowseResult(
            items=[
                StorageItem(
                    key="images/photo.jpg",
                    content_type="image/jpeg",
                    file_size=1024,
                    created_at=datetime.now(),
                ),
            ],
            total=1,
        )
        mock_get_stash.return_value = mock_stash

        result = browse(prefix="images/")

        data = json.loads(result)
        assert data["total"] == 1
        mock_stash.browse.assert_called_with(prefix="images/", limit=20)


class TestMCPStats:
    """Tests for stats tool."""

    @patch("semstash.mcp_server.get_cached_stash")
    def test_stats_success(self, mock_get_stash: MagicMock) -> None:
        """Stats tool returns statistics."""
        mock_stash = MagicMock()
        mock_stash.get_stats.return_value = UsageStats(
            content_count=10,
            vector_count=10,
            storage_bytes=1024 * 1024,
            dimension=3072,
        )
        mock_get_stash.return_value = mock_stash

        result = stats()

        data = json.loads(result)
        assert data["content_count"] == 10
        assert data["vector_count"] == 10
        assert data["storage_bytes"] == 1024 * 1024
