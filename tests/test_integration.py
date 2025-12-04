"""Integration tests for semstash with real AWS services.

These tests require:
    - Valid AWS credentials configured
    - Permission to create/delete S3 buckets and S3 Vectors resources
    - Access to Bedrock Nova embeddings model

Run with: pytest --use-aws -m integration
Preserve resources for debugging: pytest --use-aws --preserve-aws -m integration
"""

from pathlib import Path

import httpx
import pytest

from helpers import assert_valid_query_results, assert_valid_search_result
from semstash import SemStash


@pytest.mark.integration
class TestIntegrationUploadQuery:
    """Integration tests for upload and query workflow."""

    def test_upload_text_and_query(
        self,
        integration_stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Upload a text file and query for it."""
        # Upload file
        result = integration_stash.upload(sample_text_file)
        assert result.key == sample_text_file.name
        assert result.content_type == "text/plain"

        # Query for content (matches actual text file content)
        query_results = integration_stash.query("sample text for testing semantic storage", top_k=5)

        # Use helper to validate all results have proper scores and metadata
        assert_valid_query_results(
            query_results, min_count=1, expected_keys=[sample_text_file.name]
        )

        # Verify the specific matched result
        matched_result = next(r for r in query_results if r.key == sample_text_file.name)
        assert_valid_search_result(
            matched_result,
            expected_key=sample_text_file.name,
            expected_content_type="text/plain",
            require_file_size=True,
        )

    def test_upload_image_and_query(
        self,
        integration_stash: SemStash,
        sample_image_file: Path,
    ) -> None:
        """Upload an image file and query for it."""
        # Upload image
        result = integration_stash.upload(sample_image_file)
        assert result.key == sample_image_file.name
        assert result.content_type == "image/png"

        # Query for image content (1x1 transparent pixel - minimal PNG)
        # For a minimal test image, any visual query will have low similarity
        # Just verify the query returns valid results with proper metadata
        query_results = integration_stash.query("small transparent image", top_k=5)
        assert_valid_query_results(query_results, min_count=1)


@pytest.mark.integration
class TestIntegrationBrowse:
    """Integration tests for browse functionality."""

    def test_browse_after_uploads(
        self,
        integration_stash: SemStash,
        sample_text_file: Path,
        sample_json_file: Path,
    ) -> None:
        """Browse content after uploading multiple files."""

        # Upload multiple files
        integration_stash.upload(sample_text_file)
        integration_stash.upload(sample_json_file)

        # Browse all content
        result = integration_stash.browse()
        assert len(result.items) == 2
        assert result.next_token is None  # No pagination needed

        # Check file names
        keys = [item.key for item in result.items]
        assert sample_text_file.name in keys
        assert sample_json_file.name in keys


@pytest.mark.integration
class TestIntegrationGetDelete:
    """Integration tests for get and delete functionality."""

    def test_get_content(
        self,
        integration_stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Get uploaded content."""

        # Upload file
        integration_stash.upload(sample_text_file)

        # Get metadata and URL
        result = integration_stash.get(sample_text_file.name)

        assert result.key == sample_text_file.name
        assert result.content_type == "text/plain"
        assert result.url is not None
        assert "https://" in result.url  # Should be a presigned S3 URL

    def test_delete_content(
        self,
        integration_stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Delete uploaded content."""

        # Upload file
        integration_stash.upload(sample_text_file)

        # Verify it exists
        browse_result = integration_stash.browse()
        assert len(browse_result.items) == 1

        # Delete it
        result = integration_stash.delete(sample_text_file.name)
        assert result.deleted is True

        # Verify it's gone
        browse_result = integration_stash.browse()
        assert len(browse_result.items) == 0


@pytest.mark.integration
class TestIntegrationCheckSync:
    """Integration tests for check and sync functionality."""

    def test_check_consistent_storage(
        self,
        integration_stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Check returns consistent state after normal operations."""

        # Upload file
        integration_stash.upload(sample_text_file)

        # Check consistency
        result = integration_stash.check()
        assert result.is_consistent is True
        assert result.content_count == 1
        assert result.vector_count == 1
        assert len(result.orphaned_vectors) == 0
        assert len(result.missing_vectors) == 0


@pytest.mark.integration
class TestIntegrationOpenExisting:
    """Integration tests for opening existing storage."""

    def test_open_existing_stash(
        self,
        integration_stash: SemStash,
        integration_bucket_name: str,
        sample_text_file: Path,
    ) -> None:
        """Open an existing stash without re-initializing."""
        # Stash is already initialized by the session fixture
        integration_stash.upload(sample_text_file)

        # Open existing stash with a new client
        stash2 = SemStash(integration_bucket_name)
        stash2.open()  # Should detect existing config

        # Should be able to query (matches actual text file content)
        results = stash2.query("sample text for testing semantic storage", top_k=5)
        assert len(results) >= 1

        # Should have same dimension
        assert stash2.dimension == 256


@pytest.mark.integration
class TestIntegrationContentVerification:
    """Integration tests for content verification (download and compare)."""

    def test_download_text_content_matches(
        self,
        integration_stash: SemStash,
        sample_text_file: Path,
        tmp_path: Path,
    ) -> None:
        """Download text file and verify content matches original."""

        # Upload file
        integration_stash.upload(sample_text_file)

        # Get presigned URL
        result = integration_stash.get(sample_text_file.name)

        # Download content using presigned URL
        response = httpx.get(result.url)
        assert response.status_code == 200

        # Save to tmp and compare
        downloaded_path = tmp_path / "downloaded.txt"
        downloaded_path.write_bytes(response.content)

        # Verify content matches
        original_content = sample_text_file.read_bytes()
        downloaded_content = downloaded_path.read_bytes()
        assert downloaded_content == original_content

    def test_download_image_content_matches(
        self,
        integration_stash: SemStash,
        sample_image_file: Path,
        tmp_path: Path,
    ) -> None:
        """Download image file and verify content matches original."""

        # Upload image
        integration_stash.upload(sample_image_file)

        # Get presigned URL
        result = integration_stash.get(sample_image_file.name)

        # Download content using presigned URL
        response = httpx.get(result.url)
        assert response.status_code == 200

        # Save to tmp and compare
        downloaded_path = tmp_path / "downloaded.png"
        downloaded_path.write_bytes(response.content)

        # Verify content matches
        original_content = sample_image_file.read_bytes()
        downloaded_content = downloaded_path.read_bytes()
        assert downloaded_content == original_content

    def test_download_json_content_matches(
        self,
        integration_stash: SemStash,
        sample_json_file: Path,
        tmp_path: Path,
    ) -> None:
        """Download JSON file and verify content matches original."""

        # Upload JSON
        integration_stash.upload(sample_json_file)

        # Get presigned URL
        result = integration_stash.get(sample_json_file.name)

        # Download content using presigned URL
        response = httpx.get(result.url)
        assert response.status_code == 200

        # Save to tmp and compare
        downloaded_path = tmp_path / "downloaded.json"
        downloaded_path.write_bytes(response.content)

        # Verify content matches
        original_content = sample_json_file.read_bytes()
        downloaded_content = downloaded_path.read_bytes()
        assert downloaded_content == original_content


@pytest.mark.integration
class TestIntegrationTagFiltering:
    """Integration tests for tag filtering with native S3 Vectors support."""

    def test_upload_with_tags_and_filter(
        self,
        integration_stash: SemStash,
        sample_text_file: Path,
        sample_json_file: Path,
    ) -> None:
        """Upload files with tags and filter by tag in query."""
        # Upload with different tags
        integration_stash.upload(sample_text_file, tags=["documentation", "readme"])
        integration_stash.upload(sample_json_file, tags=["config", "settings"])

        # Query with tag filter - should only return the text file
        results = integration_stash.query("sample content", tags=["documentation"])
        assert_valid_query_results(results, min_count=1, expected_keys=[sample_text_file.name])
        keys = [r.key for r in results]
        # JSON file should not be in filtered results (different tags)
        assert sample_json_file.name not in keys

    def test_query_with_multiple_tags(
        self,
        integration_stash: SemStash,
        sample_text_file: Path,
        sample_json_file: Path,
    ) -> None:
        """Query with multiple tags uses OR logic (any match)."""
        # Upload with different tags
        integration_stash.upload(sample_text_file, tags=["type-a", "category-1"])
        integration_stash.upload(sample_json_file, tags=["type-b", "category-2"])

        # Query with multiple tags - should return files matching any tag
        results = integration_stash.query("content", tags=["type-a", "type-b"])
        assert_valid_query_results(
            results,
            min_count=2,
            expected_keys=[sample_text_file.name, sample_json_file.name],
        )

    def test_query_with_nonexistent_tag(
        self,
        integration_stash: SemStash,
        sample_text_file: Path,
    ) -> None:
        """Query with non-existent tag returns no results."""
        # Upload file with specific tags
        integration_stash.upload(sample_text_file, tags=["existing-tag"])

        # Query with a tag that doesn't exist
        results = integration_stash.query("sample content", tags=["nonexistent-tag"])
        # Should return empty list since no content matches the tag
        assert len(results) == 0


@pytest.mark.integration
class TestIntegrationDocuments:
    """Integration tests for document upload and search (PDF, DOCX, PPTX, XLSX)."""

    def test_upload_pdf_and_query(
        self,
        integration_stash: SemStash,
        sample_pdf_file: Path,
    ) -> None:
        """Upload PDF and query for its content."""
        # Upload PDF (contains text about machine learning)
        result = integration_stash.upload(sample_pdf_file)
        assert result.key == sample_pdf_file.name
        assert result.content_type == "application/pdf"

        # Query for content that's in the PDF
        query_results = integration_stash.query(
            "machine learning neural networks deep learning", top_k=5
        )
        assert_valid_query_results(
            query_results, min_count=1, expected_keys=[sample_pdf_file.name]
        )

    def test_upload_docx_and_query(
        self,
        integration_stash: SemStash,
        sample_docx_file: Path,
    ) -> None:
        """Upload DOCX and query for its content."""
        # Upload DOCX (contains text about semantic storage architecture)
        result = integration_stash.upload(sample_docx_file)
        assert result.key == sample_docx_file.name
        assert "wordprocessingml" in result.content_type

        # Query for content that's in the DOCX
        query_results = integration_stash.query("semantic storage architecture embeddings", top_k=5)
        assert_valid_query_results(
            query_results, min_count=1, expected_keys=[sample_docx_file.name]
        )

    def test_upload_pptx_and_query(
        self,
        integration_stash: SemStash,
        sample_pptx_file: Path,
    ) -> None:
        """Upload PPTX and query for its content."""
        # Upload PPTX (contains text about REST API design)
        result = integration_stash.upload(sample_pptx_file)
        assert result.key == sample_pptx_file.name
        assert "presentationml" in result.content_type

        # Query for content that's in the PPTX
        query_results = integration_stash.query("REST API design patterns", top_k=5)
        assert_valid_query_results(
            query_results, min_count=1, expected_keys=[sample_pptx_file.name]
        )

    def test_upload_xlsx_and_query(
        self,
        integration_stash: SemStash,
        sample_xlsx_file: Path,
    ) -> None:
        """Upload XLSX and query for its content."""
        # Upload XLSX (contains quarterly sales data)
        result = integration_stash.upload(sample_xlsx_file)
        assert result.key == sample_xlsx_file.name
        assert "spreadsheetml" in result.content_type

        # Query for content that's in the XLSX (converted to CSV/text)
        query_results = integration_stash.query("quarterly sales revenue financial data", top_k=5)
        assert_valid_query_results(
            query_results, min_count=1, expected_keys=[sample_xlsx_file.name]
        )

    def test_download_pdf_content_matches(
        self,
        integration_stash: SemStash,
        sample_pdf_file: Path,
        tmp_path: Path,
    ) -> None:
        """Download PDF file and verify content matches original."""
        # Upload PDF
        integration_stash.upload(sample_pdf_file)

        # Get presigned URL
        result = integration_stash.get(sample_pdf_file.name)
        assert result.content_type == "application/pdf"

        # Download content using presigned URL
        response = httpx.get(result.url)
        assert response.status_code == 200

        # Save to tmp and compare
        downloaded_path = tmp_path / "downloaded.pdf"
        downloaded_path.write_bytes(response.content)

        # Verify content matches
        original_content = sample_pdf_file.read_bytes()
        downloaded_content = downloaded_path.read_bytes()
        assert downloaded_content == original_content

    def test_download_docx_content_matches(
        self,
        integration_stash: SemStash,
        sample_docx_file: Path,
        tmp_path: Path,
    ) -> None:
        """Download DOCX file and verify content matches original."""
        # Upload DOCX
        integration_stash.upload(sample_docx_file)

        # Get presigned URL
        result = integration_stash.get(sample_docx_file.name)

        # Download content using presigned URL
        response = httpx.get(result.url)
        assert response.status_code == 200

        # Save to tmp and compare
        downloaded_path = tmp_path / "downloaded.docx"
        downloaded_path.write_bytes(response.content)

        # Verify content matches
        original_content = sample_docx_file.read_bytes()
        downloaded_content = downloaded_path.read_bytes()
        assert downloaded_content == original_content

    def test_cross_modality_query(
        self,
        integration_stash: SemStash,
        sample_pdf_file: Path,
        sample_docx_file: Path,
        sample_jpg_file: Path,
    ) -> None:
        """Upload multiple document types and query across them."""
        # Upload all documents (they all contain the same sample.jpg image)
        integration_stash.upload(sample_pdf_file)
        integration_stash.upload(sample_docx_file)
        integration_stash.upload(sample_jpg_file)

        # Query for landscape/nature content (common in the image)
        query_results = integration_stash.query("landscape nature scenery", top_k=5)

        # At minimum the JPG should match with valid scores
        assert_valid_query_results(
            query_results, min_count=1, expected_keys=[sample_jpg_file.name]
        )
