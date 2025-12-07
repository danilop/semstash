"""Configuration management for semstash.

Configuration is loaded with the following precedence (highest to lowest):
1. Explicit parameters passed to SemStash()
2. Environment variables (SEMSTASH_*)
3. Config file (semstash.toml)
4. Default values

Config file search paths:
- ./semstash.toml
- ./.semstash.toml
- ~/.config/semstash/config.toml
- ~/.semstash.toml
"""

import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from semstash.exceptions import DimensionError

# Nova Embeddings model ID
NOVA_EMBEDDINGS_MODEL = "amazon.nova-2-multimodal-embeddings-v1:0"

# Supported embedding dimensions (Nova multimodal embeddings)
SUPPORTED_DIMENSIONS = (256, 384, 1024, 3072)

# Default values
DEFAULT_DIMENSION = 3072
DEFAULT_REGION = "us-east-1"

# Storage constants
DEFAULT_INDEX_NAME = "default-index"
VECTOR_BUCKET_SUFFIX = "-vectors"
DEFAULT_PRESIGNED_URL_EXPIRY = 3600  # 1 hour in seconds

# Default limits for search and browse operations
DEFAULT_SEARCH_TOP_K = 10
DEFAULT_BROWSE_LIMIT = 20

# Nova Async API - Segmented Embedding Configuration
# Text segmentation: max characters per segment (Nova limit: 300 chars min, 50000 max)
DEFAULT_TEXT_SEGMENT_CHARS = 10000  # 10K chars per segment (good for semantic chunks)
MAX_TEXT_SEGMENT_CHARS = 50000  # Nova's maximum
MIN_TEXT_SEGMENT_CHARS = 300  # Nova's minimum

# Audio/Video segmentation: duration per segment in seconds (Nova limit: 5-30s)
DEFAULT_MEDIA_SEGMENT_SECONDS = 30  # Max allowed by Nova
MIN_MEDIA_SEGMENT_SECONDS = 5  # Nova's minimum
MAX_MEDIA_SEGMENT_SECONDS = 30  # Nova's maximum

# Async job polling configuration
ASYNC_POLL_INTERVAL_SECONDS = 2.0  # Initial poll interval
ASYNC_POLL_MAX_INTERVAL_SECONDS = 10.0  # Maximum poll interval (with backoff)
ASYNC_POLL_TIMEOUT_SECONDS = 600  # 10 minute timeout for async jobs

# Size thresholds for using async vs sync API
# Files larger than these use async segmentation for better semantic coverage
TEXT_ASYNC_THRESHOLD_BYTES = 50000  # 50KB text -> use async segmentation
AUDIO_ASYNC_THRESHOLD_BYTES = 5 * 1024 * 1024  # 5MB audio -> use async segmentation
VIDEO_ASYNC_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10MB video -> use async segmentation


class Config(BaseSettings):
    """semstash configuration.

    Automatically loads from environment variables (SEMSTASH_*) and config files.

    Example:
        # Use defaults
        config = Config()

        # Override specific values
        config = Config(bucket="my-bucket", dimension=1024)

        # Or set environment variables:
        # SEMSTASH_BUCKET=my-bucket
        # SEMSTASH_DIMENSION=1024
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMSTASH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # AWS settings
    region: str = Field(default=DEFAULT_REGION, description="AWS region")
    bucket: str | None = Field(default=None, description="S3 bucket for content")

    # Embedding settings
    dimension: int = Field(default=DEFAULT_DIMENSION, description="Embedding dimension")

    # Storage settings
    index_name: str = Field(default=DEFAULT_INDEX_NAME, description="Vector index name")
    presigned_url_expiry: int = Field(
        default=DEFAULT_PRESIGNED_URL_EXPIRY, description="Presigned URL expiry in seconds"
    )

    # Default limits (can be overridden per-call)
    search_top_k: int = Field(default=DEFAULT_SEARCH_TOP_K, description="Search results limit")
    browse_limit: int = Field(default=DEFAULT_BROWSE_LIMIT, description="Browse results limit")

    # Output settings
    output_format: str = Field(default="table", description="CLI output: table, json, plain")

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        """Ensure dimension is one of the supported values."""
        if v not in SUPPORTED_DIMENSIONS:
            raise DimensionError(f"dimension must be one of {SUPPORTED_DIMENSIONS}, got {v}")
        return v

    @property
    def vector_bucket(self) -> str | None:
        """S3 Vectors bucket name (derived from content bucket)."""
        return f"{self.bucket}{VECTOR_BUCKET_SUFFIX}" if self.bucket else None


def find_config_file() -> Path | None:
    """Find the first existing config file.

    Search paths (in order):
    1. ./semstash.toml
    2. ./.semstash.toml
    3. ~/.config/semstash/config.toml
    4. ~/.semstash.toml
    """
    search_paths = [
        Path.cwd() / "semstash.toml",
        Path.cwd() / ".semstash.toml",
        Path.home() / ".config" / "semstash" / "config.toml",
        Path.home() / ".semstash.toml",
    ]

    for path in search_paths:
        if path.exists():
            return path
    return None


def load_toml_config(path: Path | None = None) -> dict[str, Any]:
    """Load configuration from a TOML file.

    Args:
        path: Explicit path, or None to search default locations.

    Returns:
        Configuration dictionary (empty if no file found).
    """
    if path is None:
        path = find_config_file()

    if path is None or not path.exists():
        return {}

    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Flatten nested sections into top-level keys
    # e.g., [aws] region = "us-east-1" becomes region = "us-east-1"
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value

    return flat


def load_config(
    config_path: Path | None = None,
    **overrides: Any,
) -> Config:
    """Load configuration with full precedence chain.

    Precedence (highest to lowest):
    1. Keyword arguments (overrides)
    2. Environment variables (SEMSTASH_*)
    3. Config file (TOML)
    4. Default values

    Args:
        config_path: Explicit path to config file.
        **overrides: Values to override (e.g., bucket="my-bucket").

    Returns:
        Validated Config object.

    Example:
        # Load with defaults
        config = load_config()

        # Override bucket
        config = load_config(bucket="my-bucket")

        # Use specific config file
        config = load_config(config_path=Path("./my-config.toml"))
    """
    # Start with TOML file values
    toml_values = load_toml_config(config_path)

    # Merge with overrides (overrides take precedence)
    merged = {**toml_values}
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    # Pydantic handles env vars automatically
    return Config(**merged)
