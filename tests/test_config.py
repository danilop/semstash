"""Tests for configuration module."""

from pathlib import Path

import pytest

from semstash.config import (
    DEFAULT_REGION,
    SUPPORTED_DIMENSIONS,
    Config,
    find_config_file,
    load_config,
    load_toml_config,
)
from semstash.exceptions import DimensionError


class TestConfig:
    """Tests for Config class."""

    def test_default_values(self) -> None:
        """Config has sensible defaults."""
        config = Config()

        assert config.region == DEFAULT_REGION
        assert config.bucket is None
        assert config.dimension == 3072
        assert config.output_format == "table"

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = Config(
            region="us-east-1",
            bucket="my-bucket",
            dimension=1024,
            output_format="json",
        )

        assert config.bucket == "my-bucket"
        assert config.dimension == 1024
        assert config.output_format == "json"

    def test_vector_bucket_derived(self) -> None:
        """Vector bucket is derived from content bucket."""
        config = Config(bucket="my-bucket")
        assert config.vector_bucket == "my-bucket-vectors"

    def test_vector_bucket_none_when_no_bucket(self) -> None:
        """Vector bucket is None when no content bucket."""
        config = Config()
        assert config.vector_bucket is None

    def test_invalid_dimension_rejected(self) -> None:
        """Invalid dimensions are rejected."""
        with pytest.raises(DimensionError) as exc_info:
            Config(dimension=512)

        assert "512" in str(exc_info.value)
        assert "256, 384, 1024, 3072" in str(exc_info.value)

    def test_valid_dimensions_accepted(self) -> None:
        """All valid dimensions are accepted."""
        for dim in SUPPORTED_DIMENSIONS:
            config = Config(dimension=dim)
            assert config.dimension == dim

    def test_any_region_accepted(self) -> None:
        """Any region is accepted (AWS gives error if unsupported)."""
        config = Config(region="eu-west-1")
        assert config.region == "eu-west-1"

    def test_env_var_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config loads from environment variables."""
        monkeypatch.setenv("SEMSTASH_BUCKET", "env-bucket")
        monkeypatch.setenv("SEMSTASH_DIMENSION", "384")

        config = Config()

        assert config.bucket == "env-bucket"
        assert config.dimension == 384


class TestLoadTomlConfig:
    """Tests for TOML config file loading."""

    def test_load_existing_file(self, sample_config_file: Path) -> None:
        """Load configuration from existing TOML file."""
        config = load_toml_config(sample_config_file)

        assert config["region"] == "us-east-1"
        assert config["bucket"] == "config-bucket"
        assert config["dimension"] == 1024

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Return empty dict for non-existent file."""
        config = load_toml_config(tmp_path / "nonexistent.toml")
        assert config == {}

    def test_load_none_path(self) -> None:
        """Return empty dict when path is None."""
        config = load_toml_config(None)
        assert config == {}

    def test_nested_sections_flattened(self, tmp_path: Path) -> None:
        """Nested TOML sections are flattened."""
        file = tmp_path / "nested.toml"
        file.write_text(
            """
[aws]
region = "us-east-1"

[embeddings]
dimension = 256
"""
        )

        config = load_toml_config(file)

        assert config["region"] == "us-east-1"
        assert config["dimension"] == 256


class TestFindConfigFile:
    """Tests for config file discovery."""

    def test_find_in_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Find config file in current directory."""
        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / "semstash.toml"
        config_file.write_text("bucket = 'test'")

        found = find_config_file()
        assert found == config_file

    def test_find_hidden_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Find hidden config file in current directory."""
        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / ".semstash.toml"
        config_file.write_text("bucket = 'test'")

        found = find_config_file()
        assert found == config_file

    def test_prefer_non_hidden(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prefer non-hidden over hidden config file."""
        monkeypatch.chdir(tmp_path)

        hidden = tmp_path / ".semstash.toml"
        hidden.write_text("bucket = 'hidden'")

        visible = tmp_path / "semstash.toml"
        visible.write_text("bucket = 'visible'")

        found = find_config_file()
        assert found == visible

    def test_no_config_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return None when no config file exists."""
        monkeypatch.chdir(tmp_path)
        found = find_config_file()
        assert found is None


class TestLoadConfig:
    """Tests for the main load_config function."""

    def test_defaults_when_no_sources(self) -> None:
        """Use defaults when no config sources available."""
        config = load_config()

        assert config.region == DEFAULT_REGION
        assert config.dimension == 3072

    def test_toml_values_loaded(self, sample_config_file: Path) -> None:
        """Load values from TOML file."""
        config = load_config(config_path=sample_config_file)

        assert config.bucket == "config-bucket"
        assert config.dimension == 1024

    def test_overrides_take_precedence(self, sample_config_file: Path) -> None:
        """Explicit overrides take precedence over file."""
        config = load_config(
            config_path=sample_config_file,
            bucket="override-bucket",
            dimension=256,
        )

        assert config.bucket == "override-bucket"
        assert config.dimension == 256

    def test_file_values_take_precedence_over_env(
        self, sample_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File values take precedence over env vars (overrides > file > env > defaults)."""
        monkeypatch.setenv("SEMSTASH_DIMENSION", "384")

        config = load_config(config_path=sample_config_file)

        # File value (1024) takes precedence over env var
        assert config.dimension == 1024

    def test_env_vars_used_when_not_in_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env vars are used for values not in config file."""
        # Create a minimal config file without dimension
        file = tmp_path / "minimal.toml"
        file.write_text('bucket = "file-bucket"')

        monkeypatch.setenv("SEMSTASH_DIMENSION", "384")

        config = load_config(config_path=file)

        assert config.bucket == "file-bucket"  # From file
        assert config.dimension == 384  # From env (not in file)

    def test_none_overrides_ignored(self) -> None:
        """None values in overrides are ignored."""
        config = load_config(bucket=None, dimension=None)

        assert config.bucket is None  # Default
        assert config.dimension == 3072  # Default
