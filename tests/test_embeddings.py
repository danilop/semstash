"""Tests for embeddings module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from semstash.embeddings import (
    MODALITY_MAP,
    EmbeddingGenerator,
    get_supported_content_types,
    is_supported_content_type,
)
from semstash.exceptions import (
    DimensionError,
    UnsupportedContentTypeError,
)


class TestEmbeddingGeneratorInit:
    """Tests for EmbeddingGenerator initialization."""

    def test_default_values(self, mock_bedrock: MagicMock) -> None:
        """Generator initializes with defaults."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        assert gen.region == "us-east-1"
        assert gen.dimension == 3072

    def test_custom_dimension(self, mock_bedrock: MagicMock) -> None:
        """Generator accepts custom dimension."""
        gen = EmbeddingGenerator(dimension=1024, client=mock_bedrock)

        assert gen.dimension == 1024

    def test_invalid_dimension_rejected(self, mock_bedrock: MagicMock) -> None:
        """Invalid dimensions are rejected."""
        with pytest.raises(DimensionError) as exc_info:
            EmbeddingGenerator(dimension=512, client=mock_bedrock)

        assert "512" in str(exc_info.value)

    def test_any_region_accepted(self, mock_bedrock: MagicMock) -> None:
        """Any region is accepted (AWS gives error if unsupported)."""
        gen = EmbeddingGenerator(region="eu-west-1", client=mock_bedrock)
        assert gen.region == "eu-west-1"


class TestEmbedText:
    """Tests for text embedding generation."""

    def test_basic_text(self, mock_bedrock: MagicMock) -> None:
        """Generate embedding for basic text."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        embedding = gen.embed_text("Hello, world!")

        assert isinstance(embedding, list)
        assert len(embedding) == 3072  # Default dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_custom_dimension(self, mock_bedrock: MagicMock) -> None:
        """Text embedding respects dimension setting."""
        gen = EmbeddingGenerator(dimension=256, client=mock_bedrock)

        embedding = gen.embed_text("Hello, world!")

        assert len(embedding) == 256

    def test_empty_text(self, mock_bedrock: MagicMock) -> None:
        """Empty text can be embedded."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        embedding = gen.embed_text("")

        assert isinstance(embedding, list)


class TestEmbedImage:
    """Tests for image embedding generation."""

    def test_basic_image(self, mock_bedrock: MagicMock, sample_image_file: Path) -> None:
        """Generate embedding for image."""
        gen = EmbeddingGenerator(client=mock_bedrock)
        image_data = sample_image_file.read_bytes()

        embedding = gen.embed_image(image_data)

        assert isinstance(embedding, list)
        assert len(embedding) == 3072


class TestEmbedAudio:
    """Tests for audio embedding generation."""

    def test_basic_audio(self, mock_bedrock: MagicMock) -> None:
        """Generate embedding for audio."""
        gen = EmbeddingGenerator(client=mock_bedrock)
        # Minimal audio bytes (not valid audio, but works for mock)
        audio_data = b"fake audio data"

        embedding = gen.embed_audio(audio_data, "audio/mp3")

        assert isinstance(embedding, list)
        assert len(embedding) == 3072

    def test_mpeg_format(self, mock_bedrock: MagicMock) -> None:
        """audio/mpeg is converted to mp3 format."""
        gen = EmbeddingGenerator(client=mock_bedrock)
        audio_data = b"fake audio data"

        embedding = gen.embed_audio(audio_data, "audio/mpeg")

        assert isinstance(embedding, list)


class TestEmbedVideo:
    """Tests for video embedding generation."""

    def test_basic_video(self, mock_bedrock: MagicMock) -> None:
        """Generate embedding for video."""
        gen = EmbeddingGenerator(client=mock_bedrock)
        video_data = b"fake video data"

        embedding = gen.embed_video(video_data, "video/mp4")

        assert isinstance(embedding, list)
        assert len(embedding) == 3072

    def test_quicktime_format(self, mock_bedrock: MagicMock) -> None:
        """video/quicktime is converted to mov format."""
        gen = EmbeddingGenerator(client=mock_bedrock)
        video_data = b"fake video data"

        embedding = gen.embed_video(video_data, "video/quicktime")

        assert isinstance(embedding, list)


class TestEmbedPdf:
    """Tests for PDF embedding generation."""

    def test_pdf_document(self, mock_bedrock: MagicMock, tmp_path: Path) -> None:
        """Generate embedding for PDF document."""
        gen = EmbeddingGenerator(client=mock_bedrock)
        # Create a minimal valid PDF
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Hello World")
        pdf_bytes = doc.tobytes()
        doc.close()

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(pdf_bytes)

        embedding = gen.embed_file(pdf_file, content_type="application/pdf")

        assert isinstance(embedding, list)
        assert len(embedding) == 3072


class TestEmbedOffice:
    """Tests for Office document embedding generation (DOCX, PPTX)."""

    def test_docx_document(self, mock_bedrock: MagicMock, tmp_path: Path) -> None:
        """Generate embedding for DOCX document."""
        import docx

        gen = EmbeddingGenerator(client=mock_bedrock)

        # Create a real DOCX file
        doc = docx.Document()
        doc.add_heading("Test Document", 0)
        doc.add_paragraph("This is a test paragraph with some content.")
        doc.add_paragraph("Here is another paragraph for testing.")

        docx_file = tmp_path / "test.docx"
        doc.save(str(docx_file))

        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        embedding = gen.embed_file(docx_file, content_type=content_type)

        assert isinstance(embedding, list)
        assert len(embedding) == 3072

    def test_pptx_presentation(self, mock_bedrock: MagicMock, tmp_path: Path) -> None:
        """Generate embedding for PPTX presentation."""
        from pptx import Presentation

        gen = EmbeddingGenerator(client=mock_bedrock)

        # Create a real PPTX file
        prs = Presentation()
        slide_layout = prs.slide_layouts[0]  # Title slide
        slide = prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]

        title.text = "Test Presentation"
        subtitle.text = "This is the subtitle with test content"

        # Add another slide
        slide_layout = prs.slide_layouts[1]  # Title and content
        slide = prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        title.text = "Second Slide"

        pptx_file = tmp_path / "test.pptx"
        prs.save(str(pptx_file))

        content_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        embedding = gen.embed_file(pptx_file, content_type=content_type)

        assert isinstance(embedding, list)
        assert len(embedding) == 3072


class TestEmbedSpreadsheet:
    """Tests for spreadsheet embedding generation (XLSX)."""

    def test_xlsx_spreadsheet(self, mock_bedrock: MagicMock, tmp_path: Path) -> None:
        """Generate embedding for XLSX spreadsheet."""
        import openpyxl

        gen = EmbeddingGenerator(client=mock_bedrock)

        # Create a minimal valid XLSX
        wb = openpyxl.Workbook()
        ws = wb.active
        assert ws is not None
        ws["A1"] = "Name"
        ws["B1"] = "Value"
        ws["A2"] = "Test"
        ws["B2"] = 42

        xlsx_file = tmp_path / "test.xlsx"
        wb.save(xlsx_file)
        wb.close()

        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        embedding = gen.embed_file(xlsx_file, content_type=content_type)

        assert isinstance(embedding, list)
        assert len(embedding) == 3072


class TestEmbedFile:
    """Tests for file-based embedding generation."""

    def test_text_file(self, mock_bedrock: MagicMock, sample_text_file: Path) -> None:
        """Embed text file."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        embedding = gen.embed_file(sample_text_file)

        assert isinstance(embedding, list)
        assert len(embedding) == 3072

    def test_image_file(self, mock_bedrock: MagicMock, sample_image_file: Path) -> None:
        """Embed image file."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        embedding = gen.embed_file(sample_image_file)

        assert isinstance(embedding, list)
        assert len(embedding) == 3072

    def test_json_file(self, mock_bedrock: MagicMock, sample_json_file: Path) -> None:
        """Embed JSON file as text."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        embedding = gen.embed_file(sample_json_file)

        assert isinstance(embedding, list)

    def test_custom_content_type(self, mock_bedrock: MagicMock, sample_text_file: Path) -> None:
        """Content type can be overridden."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        # Treat text file as image
        embedding = gen.embed_file(sample_text_file, content_type="image/png")

        assert isinstance(embedding, list)

    def test_nonexistent_file(self, mock_bedrock: MagicMock, tmp_path: Path) -> None:
        """Nonexistent file raises FileNotFoundError."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        with pytest.raises(FileNotFoundError):
            gen.embed_file(tmp_path / "nonexistent.txt")

    def test_unsupported_type(self, mock_bedrock: MagicMock, tmp_path: Path) -> None:
        """Unsupported content type raises exception."""
        gen = EmbeddingGenerator(client=mock_bedrock)
        binary_file = tmp_path / "data.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        with pytest.raises(UnsupportedContentTypeError):
            gen.embed_file(binary_file, content_type="application/octet-stream")


class TestGetModality:
    """Tests for modality detection."""

    def test_text_types(self, mock_bedrock: MagicMock) -> None:
        """Text content types return text modality."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        for content_type in ("text/plain", "text/html", "application/json"):
            assert gen.get_modality(content_type) == "text"

    def test_image_types(self, mock_bedrock: MagicMock) -> None:
        """Image content types return image modality."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        for content_type in ("image/jpeg", "image/png", "image/gif"):
            assert gen.get_modality(content_type) == "image"

    def test_audio_types(self, mock_bedrock: MagicMock) -> None:
        """Audio content types return audio modality."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        for content_type in ("audio/mpeg", "audio/wav", "audio/ogg"):
            assert gen.get_modality(content_type) == "audio"

    def test_video_types(self, mock_bedrock: MagicMock) -> None:
        """Video content types return video modality."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        for content_type in ("video/mp4", "video/quicktime", "video/webm"):
            assert gen.get_modality(content_type) == "video"

    def test_generic_text(self, mock_bedrock: MagicMock) -> None:
        """Any text/* type returns text modality."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        assert gen.get_modality("text/x-custom") == "text"

    def test_unsupported_type(self, mock_bedrock: MagicMock) -> None:
        """Unsupported type raises exception."""
        gen = EmbeddingGenerator(client=mock_bedrock)

        with pytest.raises(UnsupportedContentTypeError):
            gen.get_modality("application/octet-stream")


class TestSupportedContentTypes:
    """Tests for content type utilities."""

    def test_get_supported_content_types(self) -> None:
        """get_supported_content_types returns sorted list."""
        types = get_supported_content_types()

        assert isinstance(types, list)
        assert types == sorted(types)
        assert len(types) == len(MODALITY_MAP)

    def test_is_supported_exact_match(self) -> None:
        """is_supported_content_type works for exact matches."""
        assert is_supported_content_type("image/jpeg") is True
        assert is_supported_content_type("audio/mp3") is True
        assert is_supported_content_type("video/mp4") is True

    def test_is_supported_text_prefix(self) -> None:
        """is_supported_content_type works for text/* types."""
        assert is_supported_content_type("text/plain") is True
        assert is_supported_content_type("text/x-custom") is True

    def test_is_unsupported(self) -> None:
        """is_supported_content_type returns False for unsupported."""
        assert is_supported_content_type("application/octet-stream") is False
        assert is_supported_content_type("application/x-custom") is False
