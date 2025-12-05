"""Tests for CLI module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from semstash.cli import app
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

runner = CliRunner()


class TestCliHelp:
    """Tests for CLI help output."""

    def test_main_help(self) -> None:
        """Main help shows all commands."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "init" in result.output
        assert "upload" in result.output
        assert "query" in result.output
        assert "get" in result.output
        assert "delete" in result.output
        assert "browse" in result.output
        assert "stats" in result.output
        assert "check" in result.output
        assert "sync" in result.output
        assert "destroy" in result.output


class TestCliInit:
    """Tests for init command."""

    @patch("semstash.cli.SemStash")
    def test_init_success(self, mock_stash_class: MagicMock) -> None:
        """Init command creates storage."""
        mock_stash = MagicMock()
        mock_stash.init.return_value = InitResult(
            bucket="test-bucket",
            vector_bucket="test-bucket-vectors",
            region="us-east-1",
            dimension=3072,
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["init", "test-bucket"])

        assert result.exit_code == 0
        assert "test-bucket" in result.output

    @patch("semstash.cli.SemStash")
    def test_init_json_output(self, mock_stash_class: MagicMock) -> None:
        """Init command with JSON output."""
        mock_stash = MagicMock()
        mock_stash.init.return_value = InitResult(
            bucket="test-bucket",
            vector_bucket="test-bucket-vectors",
            region="us-east-1",
            dimension=3072,
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["init", "test-bucket", "--output", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["bucket"] == "test-bucket"
        assert data["dimension"] == 3072


class TestCliUpload:
    """Tests for upload command."""

    @patch("semstash.cli.SemStash")
    def test_upload_file_to_root(self, mock_stash_class: MagicMock, sample_text_file: Path) -> None:
        """Upload command uploads file to root."""
        mock_stash = MagicMock()

        mock_stash.upload.return_value = UploadResult(
            key=sample_text_file.name,
            path=f"/{sample_text_file.name}",
            content_type="text/plain",
            file_size=100,
            dimension=3072,
        )
        mock_stash_class.return_value = mock_stash

        # New syntax: semstash upload <stash> <file> <target>
        result = runner.invoke(app, ["upload", "test-bucket", str(sample_text_file), "/"])

        assert result.exit_code == 0
        assert sample_text_file.name in result.output
        mock_stash.upload.assert_called_once()
        # Verify target was passed
        call_kwargs = mock_stash.upload.call_args[1]
        assert call_kwargs.get("target") == "/"

    @patch("semstash.cli.SemStash")
    def test_upload_file_to_folder(
        self, mock_stash_class: MagicMock, sample_text_file: Path
    ) -> None:
        """Upload command uploads file to folder."""
        mock_stash = MagicMock()

        mock_stash.upload.return_value = UploadResult(
            key=f"docs/{sample_text_file.name}",
            path=f"/docs/{sample_text_file.name}",
            content_type="text/plain",
            file_size=100,
            dimension=3072,
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["upload", "test-bucket", str(sample_text_file), "/docs/"])

        assert result.exit_code == 0
        call_kwargs = mock_stash.upload.call_args[1]
        assert call_kwargs.get("target") == "/docs/"

    def test_upload_nonexistent_file(self) -> None:
        """Upload nonexistent file shows error."""
        result = runner.invoke(app, ["upload", "test-bucket", "/nonexistent/file.txt", "/"])

        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("semstash.cli.SemStash")
    def test_upload_existing_shows_error(
        self, mock_stash_class: MagicMock, sample_text_file: Path
    ) -> None:
        """Upload to existing path shows error with force hint."""
        mock_stash = MagicMock()
        mock_stash.upload.side_effect = ContentExistsError(
            "Content already exists at '/sample.txt'. Use force=True to overwrite."
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["upload", "test-bucket", str(sample_text_file), "/"])

        assert result.exit_code == 1
        assert "already exists" in result.output
        assert "--force" in result.output

    @patch("semstash.cli.SemStash")
    def test_upload_with_force(self, mock_stash_class: MagicMock, sample_text_file: Path) -> None:
        """Upload with --force succeeds."""
        mock_stash = MagicMock()
        mock_result = MagicMock()
        mock_result.key = sample_text_file.name
        mock_result.path = f"/{sample_text_file.name}"
        mock_result.content_type = "text/plain"
        mock_result.file_size = 100
        mock_result.dimension = 3072
        mock_stash.upload.return_value = mock_result
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["upload", "test-bucket", str(sample_text_file), "/", "--force"])

        assert result.exit_code == 0
        mock_stash.upload.assert_called_once()
        # Verify force=True was passed
        call_kwargs = mock_stash.upload.call_args[1]
        assert call_kwargs.get("force") is True

    @patch("semstash.cli.SemStash")
    def test_upload_multiple_files(
        self, mock_stash_class: MagicMock, sample_text_file: Path, sample_image_file: Path
    ) -> None:
        """Upload command handles multiple files to folder."""
        mock_stash = MagicMock()
        # Return different results for each file
        mock_stash.upload.side_effect = [
            UploadResult(
                key=sample_text_file.name,
                path=f"/{sample_text_file.name}",
                content_type="text/plain",
                file_size=100,
                dimension=3072,
            ),
            UploadResult(
                key=sample_image_file.name,
                path=f"/{sample_image_file.name}",
                content_type="image/png",
                file_size=200,
                dimension=3072,
            ),
        ]
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(
            app, ["upload", "test-bucket", str(sample_text_file), str(sample_image_file), "/"]
        )

        assert result.exit_code == 0
        assert mock_stash.upload.call_count == 2
        assert "Uploaded 2 files" in result.output


class TestCliQuery:
    """Tests for query command."""

    @patch("semstash.cli.SemStash")
    def test_query_basic(self, mock_stash_class: MagicMock) -> None:
        """Query command finds results."""
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
        mock_stash_class.return_value = mock_stash

        # New syntax: semstash query <stash> <query>
        result = runner.invoke(app, ["query", "test-bucket", "sunset beach"])

        assert result.exit_code == 0
        assert "/photo.jpg" in result.output
        assert "0.95" in result.output

    @patch("semstash.cli.SemStash")
    def test_query_json_output(self, mock_stash_class: MagicMock) -> None:
        """Query with JSON output."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = [
            SearchResult(
                key="doc.txt",
                path="/doc.txt",
                score=0.85,
                distance=0.15,
            ),
        ]
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["query", "test", "test query", "--output", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["query"] == "test query"
        assert len(data["results"]) == 1
        assert data["results"][0]["path"] == "/doc.txt"

    @patch("semstash.cli.SemStash")
    def test_query_with_tags(self, mock_stash_class: MagicMock) -> None:
        """Query command filters by tags."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = [
            SearchResult(
                key="vacation.jpg",
                path="/vacation.jpg",
                score=0.90,
                distance=0.10,
                content_type="image/jpeg",
            ),
        ]
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(
            app, ["query", "test-bucket", "beach", "--tag", "vacation", "--tag", "summer"]
        )

        assert result.exit_code == 0
        # Verify tags were passed to query
        mock_stash.query.assert_called_once()
        call_kwargs = mock_stash.query.call_args.kwargs
        assert call_kwargs.get("tags") == ["vacation", "summer"]

    @patch("semstash.cli.SemStash")
    def test_query_with_path_filter(self, mock_stash_class: MagicMock) -> None:
        """Query command filters by path prefix."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = [
            SearchResult(
                key="docs/readme.txt",
                path="/docs/readme.txt",
                score=0.85,
                distance=0.15,
            ),
        ]
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["query", "test-bucket", "readme", "--path", "/docs/"])

        assert result.exit_code == 0
        # Verify path was passed to query
        mock_stash.query.assert_called_once()
        call_kwargs = mock_stash.query.call_args.kwargs
        assert call_kwargs.get("path") == "/docs/"

    @patch("semstash.cli.SemStash")
    def test_query_urls_output(self, mock_stash_class: MagicMock) -> None:
        """Query with --urls outputs presigned URLs only."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = [
            SearchResult(
                key="photo1.jpg",
                path="/photo1.jpg",
                score=0.95,
                distance=0.05,
                url="https://bucket.s3.amazonaws.com/photo1.jpg?signed=1",
            ),
            SearchResult(
                key="photo2.jpg",
                path="/photo2.jpg",
                score=0.90,
                distance=0.10,
                url="https://bucket.s3.amazonaws.com/photo2.jpg?signed=2",
            ),
        ]
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["query", "test", "photos", "--urls"])

        assert result.exit_code == 0
        # Should output only URLs, one per line
        lines = result.output.strip().split("\n")
        assert len(lines) == 2
        assert "https://bucket.s3.amazonaws.com/photo1.jpg?signed=1" in lines[0]
        assert "https://bucket.s3.amazonaws.com/photo2.jpg?signed=2" in lines[1]
        # Should NOT contain table formatting
        assert "Score" not in result.output

    @patch("semstash.cli.SemStash")
    def test_query_with_expiry(self, mock_stash_class: MagicMock) -> None:
        """Query with --expiry passes expiry to client."""
        mock_stash = MagicMock()
        mock_stash.query.return_value = [
            SearchResult(
                key="photo.jpg",
                path="/photo.jpg",
                score=0.95,
                distance=0.05,
                url="https://bucket.s3.amazonaws.com/photo.jpg?expires=7200",
            ),
        ]
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["query", "test", "photos", "--expiry", "7200"])

        assert result.exit_code == 0
        mock_stash.query.assert_called_once()
        call_kwargs = mock_stash.query.call_args.kwargs
        assert call_kwargs.get("url_expiry") == 7200


class TestCliGet:
    """Tests for get command."""

    @patch("semstash.cli.SemStash")
    def test_get_success(self, mock_stash_class: MagicMock) -> None:
        """Get command shows content info by path."""
        from datetime import datetime

        mock_stash = MagicMock()
        mock_stash.get.return_value = GetResult(
            key="photo.jpg",
            path="/photo.jpg",
            content_type="image/jpeg",
            file_size=2048,
            created_at=datetime.now(),
            url="https://example.com/photo.jpg",
        )
        mock_stash_class.return_value = mock_stash

        # Syntax: semstash get <stash> <path>
        result = runner.invoke(app, ["get", "test-bucket", "/photo.jpg"])

        assert result.exit_code == 0
        assert "/photo.jpg" in result.output
        assert "image/jpeg" in result.output
        # Verify path was passed to get
        mock_stash.get.assert_called_once()
        assert mock_stash.get.call_args[0][0] == "/photo.jpg"

    @patch("semstash.cli.SemStash")
    def test_get_nested_path(self, mock_stash_class: MagicMock) -> None:
        """Get command handles nested paths."""
        from datetime import datetime

        mock_stash = MagicMock()
        mock_stash.get.return_value = GetResult(
            key="docs/readme.txt",
            path="/docs/readme.txt",
            content_type="text/plain",
            file_size=1024,
            created_at=datetime.now(),
            url="https://example.com/docs/readme.txt",
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["get", "test-bucket", "/docs/readme.txt"])

        assert result.exit_code == 0
        assert "/docs/readme.txt" in result.output

    @patch("semstash.cli.SemStash")
    def test_get_multiple_paths(self, mock_stash_class: MagicMock) -> None:
        """Get command accepts multiple paths."""
        from datetime import datetime

        mock_stash = MagicMock()
        mock_stash.get.side_effect = [
            GetResult(
                key="photo1.jpg",
                path="/photo1.jpg",
                content_type="image/jpeg",
                file_size=1024,
                created_at=datetime.now(),
                url="https://example.com/photo1.jpg",
            ),
            GetResult(
                key="photo2.jpg",
                path="/photo2.jpg",
                content_type="image/png",
                file_size=2048,
                created_at=datetime.now(),
                url="https://example.com/photo2.jpg",
            ),
        ]
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["get", "test-bucket", "/photo1.jpg", "/photo2.jpg"])

        assert result.exit_code == 0
        assert "/photo1.jpg" in result.output
        assert "/photo2.jpg" in result.output
        assert mock_stash.get.call_count == 2

    @patch("semstash.cli.SemStash")
    def test_get_multiple_json_output(self, mock_stash_class: MagicMock) -> None:
        """Get multiple paths returns array in JSON mode."""
        from datetime import datetime

        mock_stash = MagicMock()
        mock_stash.get.side_effect = [
            GetResult(
                key="photo1.jpg",
                path="/photo1.jpg",
                content_type="image/jpeg",
                file_size=1024,
                created_at=datetime.now(),
                url="https://example.com/photo1.jpg",
            ),
            GetResult(
                key="photo2.jpg",
                path="/photo2.jpg",
                content_type="image/png",
                file_size=2048,
                created_at=datetime.now(),
                url="https://example.com/photo2.jpg",
            ),
        ]
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(
            app, ["get", "test-bucket", "/photo1.jpg", "/photo2.jpg", "-o", "json"]
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["count"] == 2
        assert len(data["items"]) == 2
        assert data["items"][0]["path"] == "/photo1.jpg"

    @patch("semstash.cli.SemStash")
    def test_get_with_expiry(self, mock_stash_class: MagicMock) -> None:
        """Get with --expiry passes expiry to client."""
        from datetime import datetime

        mock_stash = MagicMock()
        mock_stash.get.return_value = GetResult(
            key="photo.jpg",
            path="/photo.jpg",
            content_type="image/jpeg",
            file_size=2048,
            created_at=datetime.now(),
            url="https://example.com/photo.jpg?expires=7200",
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["get", "test-bucket", "/photo.jpg", "--expiry", "7200"])

        assert result.exit_code == 0
        mock_stash.get.assert_called_once()
        call_kwargs = mock_stash.get.call_args.kwargs
        assert call_kwargs.get("url_expiry") == 7200


class TestCliDelete:
    """Tests for delete command."""

    @patch("semstash.cli.SemStash")
    def test_delete_with_confirm(self, mock_stash_class: MagicMock) -> None:
        """Delete command with --yes skips confirmation."""
        mock_stash = MagicMock()
        mock_stash.delete.return_value = DeleteResult(key="photo.jpg", deleted=True)
        mock_stash_class.return_value = mock_stash

        # Syntax: semstash delete <stash> <path> --yes
        result = runner.invoke(app, ["delete", "test-bucket", "/photo.jpg", "--yes"])

        assert result.exit_code == 0
        assert "Deleted" in result.output
        # Verify path was passed to delete
        mock_stash.delete.assert_called_once()
        assert mock_stash.delete.call_args[0][0] == "/photo.jpg"

    @patch("semstash.cli.SemStash")
    def test_delete_nested_path(self, mock_stash_class: MagicMock) -> None:
        """Delete command handles nested paths."""
        mock_stash = MagicMock()
        mock_stash.delete.return_value = DeleteResult(key="docs/readme.txt", deleted=True)
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["delete", "test-bucket", "/docs/readme.txt", "--yes"])

        assert result.exit_code == 0
        mock_stash.delete.assert_called_once()
        assert mock_stash.delete.call_args[0][0] == "/docs/readme.txt"

    @patch("semstash.cli.SemStash")
    def test_delete_abort(self, mock_stash_class: MagicMock) -> None:
        """Delete command can be aborted."""
        result = runner.invoke(app, ["delete", "test-bucket", "/photo.jpg"], input="n\n")

        assert result.exit_code == 1

    @patch("semstash.cli.SemStash")
    def test_delete_multiple_paths(self, mock_stash_class: MagicMock) -> None:
        """Delete command accepts multiple paths."""
        mock_stash = MagicMock()
        mock_stash.delete.side_effect = [
            DeleteResult(key="photo1.jpg", deleted=True),
            DeleteResult(key="photo2.jpg", deleted=True),
        ]
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(
            app, ["delete", "test-bucket", "/photo1.jpg", "/photo2.jpg", "--yes"]
        )

        assert result.exit_code == 0
        assert "/photo1.jpg" in result.output
        assert "/photo2.jpg" in result.output
        assert mock_stash.delete.call_count == 2

    @patch("semstash.cli.SemStash")
    def test_delete_multiple_json_output(self, mock_stash_class: MagicMock) -> None:
        """Delete multiple paths returns array in JSON mode."""
        mock_stash = MagicMock()
        mock_stash.delete.side_effect = [
            DeleteResult(key="photo1.jpg", deleted=True),
            DeleteResult(key="photo2.jpg", deleted=True),
        ]
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(
            app,
            ["delete", "test-bucket", "/photo1.jpg", "/photo2.jpg", "--yes", "-o", "json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["count"] == 2
        assert len(data["deleted"]) == 2


class TestCliBrowse:
    """Tests for browse command."""

    @patch("semstash.cli.SemStash")
    def test_browse_empty(self, mock_stash_class: MagicMock) -> None:
        """Browse with no content at root."""
        mock_stash = MagicMock()
        mock_stash.browse.return_value = BrowseResult(items=[], total=0)
        mock_stash_class.return_value = mock_stash

        # New syntax: semstash browse <stash> <path>
        result = runner.invoke(app, ["browse", "test-bucket", "/"])

        assert result.exit_code == 0
        assert "No content found" in result.output
        # Verify path was passed
        mock_stash.browse.assert_called_once()
        call_kwargs = mock_stash.browse.call_args.kwargs
        assert call_kwargs.get("path") == "/"

    @patch("semstash.cli.SemStash")
    def test_browse_with_content(self, mock_stash_class: MagicMock) -> None:
        """Browse with content items at root."""
        from datetime import datetime

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
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["browse", "test-bucket", "/"])

        assert result.exit_code == 0
        assert "photo.jpg" in result.output
        assert "/photo.jpg" in result.output

    @patch("semstash.cli.SemStash")
    def test_browse_folder(self, mock_stash_class: MagicMock) -> None:
        """Browse content in a folder."""
        from datetime import datetime

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
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["browse", "test-bucket", "/docs/"])

        assert result.exit_code == 0
        assert "/docs/readme.txt" in result.output
        # Verify path was passed
        mock_stash.browse.assert_called_once()
        call_kwargs = mock_stash.browse.call_args.kwargs
        assert call_kwargs.get("path") == "/docs/"


class TestCliStats:
    """Tests for stats command."""

    @patch("semstash.cli.SemStash")
    def test_stats(self, mock_stash_class: MagicMock) -> None:
        """Stats command shows statistics."""
        mock_stash = MagicMock()
        mock_stash.get_stats.return_value = UsageStats(
            content_count=10,
            vector_count=10,
            storage_bytes=1024 * 1024,
            dimension=3072,
        )
        mock_stash_class.return_value = mock_stash

        # New syntax: semstash stats <stash>
        result = runner.invoke(app, ["stats", "test-bucket"])

        assert result.exit_code == 0
        assert "10" in result.output
        assert "3072" in result.output


class TestCliCheck:
    """Tests for check command."""

    @patch("semstash.cli.SemStash")
    def test_check_consistent(self, mock_stash_class: MagicMock) -> None:
        """Check command shows consistent status."""
        mock_stash = MagicMock()
        mock_stash.check.return_value = CheckResult(
            content_count=5,
            vector_count=5,
            orphaned_vectors=[],
            missing_vectors=[],
            is_consistent=True,
        )
        mock_stash_class.return_value = mock_stash

        # New syntax: semstash check <stash>
        result = runner.invoke(app, ["check", "test-bucket"])

        assert result.exit_code == 0
        assert "consistent" in result.output.lower()
        assert "5" in result.output

    @patch("semstash.cli.SemStash")
    def test_check_inconsistent(self, mock_stash_class: MagicMock) -> None:
        """Check command shows inconsistencies."""
        mock_stash = MagicMock()
        mock_stash.check.return_value = CheckResult(
            content_count=3,
            vector_count=5,
            orphaned_vectors=["orphan1.jpg", "orphan2.jpg"],
            missing_vectors=["missing1.txt"],
            is_consistent=False,
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["check", "test-bucket"])

        assert result.exit_code == 0
        assert "Orphaned" in result.output
        assert "Missing" in result.output
        assert "sync" in result.output.lower()

    @patch("semstash.cli.SemStash")
    def test_check_json_output(self, mock_stash_class: MagicMock) -> None:
        """Check command with JSON output."""
        mock_stash = MagicMock()
        mock_stash.check.return_value = CheckResult(
            content_count=3,
            vector_count=3,
            orphaned_vectors=[],
            missing_vectors=[],
            is_consistent=True,
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["check", "test-bucket", "--output", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["content_count"] == 3
        assert data["is_consistent"] is True


class TestCliSync:
    """Tests for sync command."""

    @patch("semstash.cli.SemStash")
    def test_sync_already_consistent(self, mock_stash_class: MagicMock) -> None:
        """Sync command when already consistent."""
        mock_stash = MagicMock()
        mock_stash.check.return_value = CheckResult(
            content_count=5,
            vector_count=5,
            orphaned_vectors=[],
            missing_vectors=[],
            is_consistent=True,
        )
        mock_stash_class.return_value = mock_stash

        # New syntax: semstash sync <stash> --yes
        result = runner.invoke(app, ["sync", "test-bucket", "--yes"])

        assert result.exit_code == 0
        assert "consistent" in result.output.lower()
        mock_stash.sync.assert_not_called()

    @patch("semstash.cli.SemStash")
    def test_sync_with_changes(self, mock_stash_class: MagicMock) -> None:
        """Sync command with changes."""
        mock_stash = MagicMock()
        mock_stash.check.return_value = CheckResult(
            content_count=3,
            vector_count=5,
            orphaned_vectors=["orphan1.jpg"],
            missing_vectors=["missing1.txt"],
            is_consistent=False,
        )
        mock_stash.sync.return_value = SyncResult(
            deleted_vectors=["orphan1.jpg"],
            created_vectors=["missing1.txt"],
            failed_keys=[],
            deleted_count=1,
            created_count=1,
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["sync", "test-bucket", "--yes"])

        assert result.exit_code == 0
        assert "1" in result.output
        mock_stash.sync.assert_called_once()

    @patch("semstash.cli.SemStash")
    def test_sync_abort(self, mock_stash_class: MagicMock) -> None:
        """Sync command can be aborted."""
        mock_stash = MagicMock()
        mock_stash.check.return_value = CheckResult(
            content_count=3,
            vector_count=5,
            orphaned_vectors=["orphan1.jpg"],
            missing_vectors=[],
            is_consistent=False,
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["sync", "test-bucket"], input="n\n")

        assert result.exit_code == 1
        mock_stash.sync.assert_not_called()

    @patch("semstash.cli.SemStash")
    def test_sync_json_output(self, mock_stash_class: MagicMock) -> None:
        """Sync command with JSON output."""
        mock_stash = MagicMock()
        mock_stash.check.return_value = CheckResult(
            content_count=5,
            vector_count=5,
            orphaned_vectors=[],
            missing_vectors=[],
            is_consistent=True,
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["sync", "test-bucket", "--output", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["deleted_count"] == 0
        assert data["created_count"] == 0


class TestCliDestroy:
    """Tests for destroy command."""

    @patch("semstash.cli.SemStash")
    def test_destroy_with_yes(self, mock_stash_class: MagicMock) -> None:
        """Destroy command with --yes skips confirmation."""
        mock_stash = MagicMock()
        mock_stash._content_storage._bucket = "test-bucket"
        mock_stash.destroy.return_value = DestroyResult(
            bucket="test-bucket",
            vector_bucket="test-bucket-vectors",
            content_deleted=5,
            vectors_deleted=5,
            bucket_deleted=True,
            vector_bucket_deleted=True,
            destroyed=True,
        )
        mock_stash_class.return_value = mock_stash

        # New syntax: semstash destroy <stash> --yes --force
        result = runner.invoke(app, ["destroy", "test-bucket", "--yes", "--force"])

        assert result.exit_code == 0
        assert "destroyed" in result.output.lower()
        mock_stash.destroy.assert_called_once_with(force=True)

    @patch("semstash.cli.SemStash")
    def test_destroy_abort(self, mock_stash_class: MagicMock) -> None:
        """Destroy command can be aborted."""
        mock_stash = MagicMock()
        mock_stash._content_storage._bucket = "test-bucket"
        mock_stash.get_stats.return_value = UsageStats(
            content_count=5,
            vector_count=5,
            storage_bytes=1024,
            dimension=3072,
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["destroy", "test-bucket"], input="n\n")

        assert result.exit_code == 1
        mock_stash.destroy.assert_not_called()

    @patch("semstash.cli.SemStash")
    def test_destroy_json_output(self, mock_stash_class: MagicMock) -> None:
        """Destroy command with JSON output."""
        mock_stash = MagicMock()
        mock_stash.destroy.return_value = DestroyResult(
            bucket="test-bucket",
            vector_bucket="test-bucket-vectors",
            content_deleted=0,
            vectors_deleted=0,
            bucket_deleted=True,
            vector_bucket_deleted=True,
            destroyed=True,
        )
        mock_stash_class.return_value = mock_stash

        result = runner.invoke(app, ["destroy", "test-bucket", "--yes", "--output", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["bucket"] == "test-bucket"
        assert data["destroyed"] is True
