"""Nova Multimodal Embeddings generation.

Generates embeddings using Amazon Nova for text, images, audio, and video.
Supports configurable dimensions (256, 384, 1024, 3072).

Document support:
- PDF: Rendered to image using PyMuPDF, embedded with DOCUMENT_IMAGE detail level
- DOCX: Text extracted using python-docx, embedded as text
- PPTX: Text extracted using python-pptx, embedded as text
- XLSX: Converted to CSV text using openpyxl, embedded as text
"""

import base64
import io
import json
import mimetypes
from pathlib import Path
from typing import Any

import boto3
import docx
import fitz  # PyMuPDF
import openpyxl
from botocore.exceptions import ClientError
from pptx import Presentation

from semstash.config import (
    DEFAULT_DIMENSION,
    DEFAULT_REGION,
    NOVA_EMBEDDINGS_MODEL,
    SUPPORTED_DIMENSIONS,
)
from semstash.exceptions import (
    DimensionError,
    EmbeddingError,
    UnsupportedContentTypeError,
)

# Content type to modality mapping
MODALITY_MAP = {
    # Text types
    "text/plain": "text",
    "text/html": "text",
    "text/markdown": "text",
    "text/csv": "text",
    "application/json": "text",
    "application/xml": "text",
    # Image types
    "image/jpeg": "image",
    "image/png": "image",
    "image/gif": "image",
    "image/webp": "image",
    # Audio types
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
    "audio/wav": "audio",
    "audio/ogg": "audio",
    "audio/flac": "audio",
    # Video types
    "video/mp4": "video",
    "video/quicktime": "video",
    "video/x-msvideo": "video",
    "video/webm": "video",
    # PDF documents (rendered to image via PyMuPDF)
    "application/pdf": "pdf",
    # Office documents (text extracted via python-docx/python-pptx)
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "office",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "office",
    # Spreadsheet types (converted to text/CSV)
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "spreadsheet",
}

# Format mappings for media types (content_type -> Nova format string)
IMAGE_FORMAT_MAP = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

AUDIO_FORMAT_MAP = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/ogg": "ogg",
    "audio/flac": "wav",  # Fallback for unsupported formats
}

VIDEO_FORMAT_MAP = {
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/x-msvideo": "avi",
    "video/webm": "webm",
    "video/x-matroska": "mkv",
    "video/x-flv": "flv",
    "video/mpeg": "mpeg",
    "video/3gpp": "3gp",
    "video/x-ms-wmv": "wmv",
}


class EmbeddingGenerator:
    """Generates embeddings using Amazon Nova Multimodal.

    Example:
        generator = EmbeddingGenerator(dimension=1024)

        # Generate text embedding
        embedding = generator.embed_text("Hello world")

        # Generate image embedding
        embedding = generator.embed_file(Path("photo.jpg"))

        # Get modality for a file
        modality = generator.get_modality("image/jpeg")  # "image"
    """

    def __init__(
        self,
        region: str = DEFAULT_REGION,
        dimension: int = DEFAULT_DIMENSION,
        client: Any | None = None,
    ) -> None:
        """Initialize embedding generator.

        Args:
            region: AWS region.
            dimension: Embedding dimension (256, 384, 1024, 3072).
            client: Optional boto3 bedrock-runtime client (for testing).

        Raises:
            DimensionError: If dimension is not supported.
        """
        if dimension not in SUPPORTED_DIMENSIONS:
            raise DimensionError(
                f"Invalid dimension {dimension}. "
                f"Supported: {', '.join(str(d) for d in SUPPORTED_DIMENSIONS)}"
            )

        self.region = region
        self.dimension = dimension
        self._client = client

    @property
    def client(self) -> Any:
        """Get or create Bedrock Runtime client."""
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text content.

        Args:
            text: Text content to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        request = self._build_request(
            {
                "text": {"truncationMode": "END", "value": text},
            }
        )
        return self._invoke_model(request)

    def embed_image(self, image_data: bytes, content_type: str = "image/jpeg") -> list[float]:
        """Generate embedding for image content.

        Args:
            image_data: Raw image bytes.
            content_type: MIME type of the image.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        request = self._build_request(
            {
                "image": {
                    "detailLevel": "STANDARD_IMAGE",
                    "format": IMAGE_FORMAT_MAP.get(content_type, "jpeg"),
                    "source": {"bytes": base64.b64encode(image_data).decode("utf-8")},
                },
            }
        )
        return self._invoke_model(request)

    def embed_audio(self, audio_data: bytes, content_type: str) -> list[float]:
        """Generate embedding for audio content.

        Args:
            audio_data: Raw audio bytes.
            content_type: MIME type of the audio.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        request = self._build_request(
            {
                "audio": {
                    "format": AUDIO_FORMAT_MAP.get(content_type, "mp3"),
                    "source": {"bytes": base64.b64encode(audio_data).decode("utf-8")},
                },
            }
        )
        return self._invoke_model(request)

    def embed_video(self, video_data: bytes, content_type: str) -> list[float]:
        """Generate embedding for video content.

        Args:
            video_data: Raw video bytes.
            content_type: MIME type of the video.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        request = self._build_request(
            {
                "video": {
                    "format": VIDEO_FORMAT_MAP.get(content_type, "mp4"),
                    "embeddingMode": "AUDIO_VIDEO_COMBINED",
                    "source": {"bytes": base64.b64encode(video_data).decode("utf-8")},
                },
            }
        )
        return self._invoke_model(request)

    def embed_pdf(self, pdf_data: bytes) -> list[float]:
        """Generate embedding for PDF content.

        Uses PyMuPDF to render PDF pages to images, then embeds with
        the Nova DOCUMENT_IMAGE detail level for better text interpretation.
        For multi-page documents, embeds the first page.

        Args:
            pdf_data: Raw PDF bytes.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            if doc.page_count == 0:
                raise EmbeddingError("PDF has no pages")
            # Render first page to PNG at 2x resolution for better text recognition
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            image_data = pix.tobytes("png")
            doc.close()
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to render PDF to image: {e}") from e

        request = self._build_request(
            {
                "image": {
                    "detailLevel": "DOCUMENT_IMAGE",
                    "format": "png",
                    "source": {"bytes": base64.b64encode(image_data).decode("utf-8")},
                },
            }
        )
        return self._invoke_model(request)

    def embed_office(self, office_data: bytes, content_type: str) -> list[float]:
        """Generate embedding for Office document content (DOCX, PPTX).

        Extracts text from Office documents and embeds as text.
        This captures semantic content for search without external dependencies.

        Args:
            office_data: Raw Office document bytes.
            content_type: MIME type to determine document format.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            # Content type constants for readability
            docx_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            pptx_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            extractors = {
                docx_type: self._extract_docx_text,
                pptx_type: self._extract_pptx_text,
            }
            extractor = extractors.get(content_type)
            if not extractor:
                raise EmbeddingError(f"Unsupported Office format: {content_type}")

            text = extractor(office_data)
            if not text.strip():
                raise EmbeddingError("Document contains no text")
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to extract text from document: {e}") from e

        return self.embed_text(text)

    def _extract_docx_text(self, docx_data: bytes) -> str:
        """Extract text from DOCX document.

        Args:
            docx_data: Raw DOCX bytes.

        Returns:
            Extracted text content.
        """
        doc = docx.Document(io.BytesIO(docx_data))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)

    def _extract_pptx_text(self, pptx_data: bytes) -> str:
        """Extract text from PPTX presentation.

        Args:
            pptx_data: Raw PPTX bytes.

        Returns:
            Extracted text content.
        """
        prs = Presentation(io.BytesIO(pptx_data))
        text_parts = []

        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text)

            if slide_text:
                text_parts.append(f"=== Slide {slide_num} ===")
                text_parts.extend(slide_text)

        return "\n".join(text_parts)

    def embed_spreadsheet(self, spreadsheet_data: bytes) -> list[float]:
        """Generate embedding for spreadsheet content (XLSX).

        Converts XLSX to CSV text format and embeds as text.
        This preserves the tabular data structure for semantic search.

        Args:
            spreadsheet_data: Raw XLSX file bytes.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            wb = openpyxl.load_workbook(io.BytesIO(spreadsheet_data), data_only=True)
            text_parts = []
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                text_parts.append(f"=== Sheet: {sheet_name} ===")
                for row in sheet.iter_rows(values_only=True):
                    row_values = [str(cell) if cell is not None else "" for cell in row]
                    if any(v.strip() for v in row_values):
                        text_parts.append(",".join(row_values))
            wb.close()
            text = "\n".join(text_parts)
            if not text.strip():
                raise EmbeddingError("Spreadsheet is empty")
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to convert spreadsheet to text: {e}") from e

        return self.embed_text(text)

    def embed_file(self, file_path: Path, content_type: str | None = None) -> list[float]:
        """Generate embedding for a file.

        Automatically detects content type and uses appropriate embedding method.

        Args:
            file_path: Path to the file.
            content_type: Optional MIME type (auto-detected if None).

        Returns:
            Embedding vector as list of floats.

        Raises:
            UnsupportedContentTypeError: If content type is not supported.
            EmbeddingError: If embedding generation fails.
            FileNotFoundError: If file doesn't exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if content_type is None:
            content_type, _ = mimetypes.guess_type(str(file_path))
            content_type = content_type or "application/octet-stream"

        modality = self.get_modality(content_type)

        # Dispatch table for modality handlers
        if modality == "text":
            text = file_path.read_text(encoding="utf-8", errors="replace")
            return self.embed_text(text)

        # Binary modalities - read file once and dispatch
        file_data = file_path.read_bytes()

        handlers = {
            "image": lambda: self.embed_image(file_data, content_type),
            "audio": lambda: self.embed_audio(file_data, content_type),
            "video": lambda: self.embed_video(file_data, content_type),
            "pdf": lambda: self.embed_pdf(file_data),
            "office": lambda: self.embed_office(file_data, content_type),
            "spreadsheet": lambda: self.embed_spreadsheet(file_data),
        }

        handler = handlers.get(modality)
        if handler:
            return handler()

        raise UnsupportedContentTypeError(
            f"Cannot generate embedding for content type: {content_type}"
        )

    def get_modality(self, content_type: str) -> str:
        """Determine modality from content type.

        Args:
            content_type: MIME type string.

        Returns:
            Modality string: "text", "image", "audio", "video", "pdf", "office", or "spreadsheet".

        Raises:
            UnsupportedContentTypeError: If content type is not supported.
        """
        # Check exact match
        if content_type in MODALITY_MAP:
            return MODALITY_MAP[content_type]

        # Check prefix match for text types
        if content_type.startswith("text/"):
            return "text"

        # Unknown type
        raise UnsupportedContentTypeError(
            f"Unsupported content type: {content_type}. "
            f"Supported types: {', '.join(sorted(MODALITY_MAP.keys()))}"
        )

    def _build_request(self, content_params: dict[str, Any]) -> dict[str, Any]:
        """Build Nova embedding request body.

        Args:
            content_params: Content-specific params (text, image, audio, or video).

        Returns:
            Complete request body for invoke_model.
        """
        return {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": self.dimension,
                **content_params,
            },
        }

    def _invoke_model(self, request_body: dict[str, Any]) -> list[float]:
        """Invoke the Nova embeddings model.

        Args:
            request_body: Model request payload.

        Returns:
            Embedding vector.

        Raises:
            EmbeddingError: If model invocation fails.
        """
        try:
            response = self.client.invoke_model(
                modelId=NOVA_EMBEDDINGS_MODEL,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())

            # New response format: {"embeddings": [{"embeddingType": "...", "embedding": [...]}]}
            embeddings = response_body.get("embeddings")
            if not embeddings:
                raise EmbeddingError("No embeddings in model response")

            embedding: list[float] | None = embeddings[0].get("embedding")
            if embedding is None:
                raise EmbeddingError("No embedding vector in model response")

            return embedding

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            raise EmbeddingError(f"Bedrock error ({error_code}): {error_msg}") from e

        except json.JSONDecodeError as e:
            raise EmbeddingError(f"Invalid model response: {e}") from e


def get_supported_content_types() -> list[str]:
    """Get list of all supported content types for embedding.

    Returns:
        Sorted list of MIME type strings.
    """
    return sorted(MODALITY_MAP.keys())


def is_supported_content_type(content_type: str) -> bool:
    """Check if a content type is supported for embedding.

    Args:
        content_type: MIME type string.

    Returns:
        True if supported, False otherwise.
    """
    if content_type in MODALITY_MAP:
        return True
    return content_type.startswith("text/")
