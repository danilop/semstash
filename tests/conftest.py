"""Shared test fixtures for semstash.

Provides mocked AWS services and sample test data.

To run integration tests with real AWS:
    pytest --use-aws -m integration

Integration tests require:
    - Valid AWS credentials configured
    - Permission to create/delete S3 buckets and S3 Vectors resources
    - Access to Bedrock Nova embeddings model
"""

import atexit
import io
import json
import os
import signal
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import boto3
import pytest

# Re-export helper functions for use by test files
from helpers import assert_valid_query_results, assert_valid_search_result
from moto import mock_aws

from semstash.config import Config

__all__ = ["assert_valid_query_results", "assert_valid_search_result"]

# --- Robust Cleanup Registry ---
# Track resources to clean up even on interrupt/crash

_cleanup_registry: dict[str, bool] = {}  # bucket_name -> preserve flag
_original_sigint_handler: signal.Handlers | None = None


def _register_for_cleanup(bucket_name: str, preserve: bool = False) -> None:
    """Register a bucket for cleanup on exit."""
    _cleanup_registry[bucket_name] = preserve


def _unregister_cleanup(bucket_name: str) -> None:
    """Remove a bucket from cleanup registry (already cleaned)."""
    _cleanup_registry.pop(bucket_name, None)


def _cleanup_all_resources() -> None:
    """Clean up all registered resources. Called on exit or interrupt."""
    if not _cleanup_registry:
        return

    # Import here to avoid circular imports
    from semstash import SemStash

    for bucket_name, preserve in list(_cleanup_registry.items()):
        if preserve:
            print(f"\n[atexit] Preserving: {bucket_name}")
            continue
        try:
            stash = SemStash(bucket_name, auto_init=False)
            stash.open()
            stash.destroy(force=True)
            print(f"\n[atexit] Cleaned up: {bucket_name}")
        except Exception as e:
            print(f"\n[atexit] Failed to cleanup {bucket_name}: {e}")
        finally:
            _cleanup_registry.pop(bucket_name, None)


def _sigint_handler(signum: int, frame: Any) -> None:
    """Handle Ctrl+C by cleaning up before exit."""
    print("\n\nInterrupted! Cleaning up test resources...")
    _cleanup_all_resources()
    # Call original handler or exit
    if _original_sigint_handler and callable(_original_sigint_handler):
        _original_sigint_handler(signum, frame)
    else:
        raise KeyboardInterrupt


# Register atexit handler for normal exits
atexit.register(_cleanup_all_resources)


def _install_signal_handlers() -> None:
    """Install signal handlers for interrupt cleanup."""
    global _original_sigint_handler
    if _original_sigint_handler is None:
        _original_sigint_handler = signal.signal(signal.SIGINT, _sigint_handler)


# --- Pytest Configuration ---


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line options for integration tests."""
    parser.addoption(
        "--use-aws",
        action="store_true",
        default=False,
        help="Run integration tests with real AWS services",
    )
    parser.addoption(
        "--preserve-aws",
        action="store_true",
        default=False,
        help="Preserve AWS resources after tests (for debugging)",
    )


def _check_aws_credentials() -> bool:
    """Check if valid AWS credentials are available."""
    try:
        import subprocess

        result = subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless --use-aws flag is provided and credentials are valid."""
    use_aws = config.getoption("--use-aws")

    if not use_aws:
        # --use-aws not given: skip integration tests
        skip_integration = pytest.mark.skip(reason="need --use-aws option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
        return

    # --use-aws given: check if credentials are available
    if not _check_aws_credentials():
        skip_no_creds = pytest.mark.skip(
            reason="AWS credentials not available (run 'aws configure' or set AWS_* env vars)"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_no_creds)


@pytest.fixture
def use_aws(request: pytest.FixtureRequest) -> bool:
    """Return True if --use-aws flag is set."""
    return bool(request.config.getoption("--use-aws"))


# --- Environment Setup ---


@pytest.fixture(autouse=True)
def aws_credentials(request: pytest.FixtureRequest) -> Generator[None]:
    """Set mock AWS credentials for unit tests.

    When --use-aws is set, this fixture does nothing and allows
    real AWS credentials to be used.
    """
    if request.config.getoption("--use-aws", default=False):
        # Use real credentials for integration tests
        yield
        return

    # Mock credentials for unit tests
    env_vars = {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SECURITY_TOKEN": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }
    original = {k: os.environ.get(k) for k in env_vars}

    os.environ.update(env_vars)
    yield

    # Restore original values
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# --- Integration Test Fixtures ---


@pytest.fixture(scope="session")
def integration_bucket_name() -> str:
    """Generate unique bucket name for integration tests (session-scoped)."""
    return f"semstash-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def agent_bucket_name() -> str:
    """Generate unique bucket name for agent tests (session-scoped)."""
    return f"semstash-agent-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def preserve_aws(request: pytest.FixtureRequest) -> bool:
    """Return True if --preserve-aws flag is set (session-scoped)."""
    return bool(request.config.getoption("--preserve-aws", default=False))


@pytest.fixture(scope="session")
def _session_stash(
    integration_bucket_name: str,
    preserve_aws: bool,
) -> Generator[Any]:
    """Session-scoped stash - created once, destroyed at end of all tests.

    This fixture handles:
    - Creating S3 bucket and vector index ONCE for all tests
    - Automatic cleanup after ALL tests complete (success or failure)
    - Option to preserve resources with --preserve-aws flag
    - Robust cleanup on interrupt via atexit/signal handlers
    """
    from semstash import SemStash
    from semstash.exceptions import AlreadyExistsError

    # Install signal handlers for interrupt cleanup
    _install_signal_handlers()

    # Register for cleanup BEFORE creating resources
    _register_for_cleanup(integration_bucket_name, preserve_aws)

    stash = SemStash(integration_bucket_name, dimension=256)
    try:
        stash.init()  # Initialize once for all tests
        print(f"\n>>> Created integration test resources: {integration_bucket_name}")
    except AlreadyExistsError:
        # Bucket already exists (e.g., from agent tests), just open it
        stash.open()
        print(f"\n>>> Opened existing integration test resources: {integration_bucket_name}")

    yield stash

    # Cleanup at end of session (after all tests)
    if preserve_aws:
        print(f"\n--preserve-aws: Keeping resources for {integration_bucket_name}")
    else:
        try:
            stash.destroy(force=True)
            print(f"\n>>> Cleaned up integration test resources: {integration_bucket_name}")
            # Unregister since we cleaned up successfully
            _unregister_cleanup(integration_bucket_name)
        except Exception as e:
            # Log but don't fail - atexit handler will retry
            print(f"\nWarning: Failed to cleanup {integration_bucket_name}: {e}")


@pytest.fixture
def integration_stash(_session_stash: Any) -> Generator[Any]:
    """Function-scoped stash that clears content before each test.

    Uses the session-scoped stash (S3 bucket + vector index created once),
    but clears all content before each test for isolation.

    Usage:
        def test_something(integration_stash):
            # Stash is already initialized, just use it
            integration_stash.upload(some_file)
    """
    stash = _session_stash

    # Clear all content before the test for isolation
    try:
        browse_result = stash.browse("/")
        for item in browse_result.items:
            stash.delete(item.path)
    except Exception:
        pass  # Ignore errors (e.g., if no content exists)

    yield stash


@pytest.fixture
def real_s3_client() -> Any:
    """Create real S3 client for integration tests."""
    return boto3.client("s3", region_name="us-east-1")


@pytest.fixture
def real_s3vectors_client() -> Any:
    """Create real S3 Vectors client for integration tests."""
    return boto3.client("s3vectors", region_name="us-east-1")


@pytest.fixture
def real_bedrock_client() -> Any:
    """Create real Bedrock Runtime client for integration tests."""
    return boto3.client("bedrock-runtime", region_name="us-east-1")


@pytest.fixture(scope="class", autouse=False)
def agent_stash_cleanup(agent_bucket_name: str, preserve_aws: bool) -> Generator[None]:
    """Cleanup agent test bucket after agent test class completes.

    Uses robust cleanup registry for interrupt handling.
    """
    from semstash import SemStash

    # Install signal handlers and register for cleanup
    _install_signal_handlers()
    _register_for_cleanup(agent_bucket_name, preserve_aws)

    yield  # Run tests first

    # Cleanup after agent tests
    if preserve_aws:
        print(f"\n--preserve-aws: Keeping agent resources for {agent_bucket_name}")
    else:
        try:
            stash = SemStash(agent_bucket_name, auto_init=False)
            stash.open()
            stash.destroy(force=True)
            print(f"\n>>> Cleaned up agent test resources: {agent_bucket_name}")
            # Unregister since we cleaned up successfully
            _unregister_cleanup(agent_bucket_name)
        except Exception as e:
            # Log but don't fail - atexit handler will retry
            print(f"\nWarning: Failed to cleanup agent bucket {agent_bucket_name}: {e}")


# --- S3 Mocking (using moto) ---


@pytest.fixture
def mock_s3() -> Generator[Any]:
    """Mock S3 using moto."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        yield client


@pytest.fixture
def s3_bucket(mock_s3: Any) -> str:
    """Create a test S3 bucket."""
    bucket_name = "test-bucket"
    mock_s3.create_bucket(Bucket=bucket_name)
    return bucket_name


# --- S3 Vectors Mocking (manual mock - not in moto) ---


@pytest.fixture
def mock_s3vectors() -> MagicMock:
    """Mock S3 Vectors client.

    S3 Vectors is in preview and not supported by moto,
    so we create a manual mock with in-memory storage.
    """
    from botocore.exceptions import ClientError

    mock = MagicMock()
    vector_store: dict[str, dict[str, Any]] = {}
    buckets: set[str] = set()
    indexes: dict[str, dict[str, Any]] = {}

    def create_vector_bucket(vectorBucketName: str, **kwargs: Any) -> dict[str, Any]:
        buckets.add(vectorBucketName)
        return {"vectorBucketArn": f"arn:aws:s3vectors:us-east-1:123456789012:{vectorBucketName}"}

    def create_index(
        vectorBucketName: str, indexName: str, dimension: int, **kwargs: Any
    ) -> dict[str, Any]:
        key = f"{vectorBucketName}/{indexName}"
        indexes[key] = {"dimension": dimension, "vectorCount": 0}
        return {"indexArn": f"arn:aws:s3vectors:us-east-1:123456789012:{key}"}

    def put_vectors(
        vectorBucketName: str, indexName: str, vectors: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        for v in vectors:
            key = f"{vectorBucketName}/{indexName}/{v['key']}"
            vector_store[key] = v
        return {}

    def query_vectors(
        vectorBucketName: str,
        indexName: str,
        queryVector: dict[str, list[float]] | None = None,
        topK: int = 10,
        returnDistance: bool = False,
        returnMetadata: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Return stored vectors (simple mock, no actual similarity)
        # Mirrors real S3 Vectors API: only return distance/metadata if requested
        prefix = f"{vectorBucketName}/{indexName}/"
        matching = [v for k, v in vector_store.items() if k.startswith(prefix)]
        results = matching[:topK]
        vectors = []
        for v in results:
            result: dict[str, Any] = {"key": v["key"]}
            if returnDistance:
                result["distance"] = 0.1  # Mock distance
            if returnMetadata:
                result["metadata"] = v.get("metadata", {})
            vectors.append(result)
        return {"vectors": vectors}

    def delete_vectors(
        vectorBucketName: str, indexName: str, keys: list[str], **kwargs: Any
    ) -> dict[str, Any]:
        for k in keys:
            full_key = f"{vectorBucketName}/{indexName}/{k}"
            vector_store.pop(full_key, None)
        return {}

    def get_index(vectorBucketName: str, indexName: str, **kwargs: Any) -> dict[str, Any]:
        key = f"{vectorBucketName}/{indexName}"
        if key not in indexes:
            # Raise ClientError like the real API
            error_response = {
                "Error": {"Code": "ResourceNotFoundException", "Message": "Index not found"}
            }
            raise ClientError(error_response, "GetIndex")
        return {"dimension": indexes[key]["dimension"], "vectorCount": len(vector_store)}

    def list_vectors(
        vectorBucketName: str,
        indexName: str,
        maxResults: int = 1000,
        nextToken: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # List all vectors in the index
        prefix = f"{vectorBucketName}/{indexName}/"
        matching = [{"key": v["key"]} for k, v in vector_store.items() if k.startswith(prefix)]
        return {"vectors": matching}

    def delete_index(vectorBucketName: str, indexName: str, **kwargs: Any) -> dict[str, Any]:
        key = f"{vectorBucketName}/{indexName}"
        indexes.pop(key, None)
        # Also delete all vectors in this index
        prefix = f"{vectorBucketName}/{indexName}/"
        to_delete = [k for k in vector_store if k.startswith(prefix)]
        for k in to_delete:
            vector_store.pop(k, None)
        return {}

    def delete_vector_bucket(vectorBucketName: str, **kwargs: Any) -> dict[str, Any]:
        buckets.discard(vectorBucketName)
        return {}

    mock.create_vector_bucket = MagicMock(side_effect=create_vector_bucket)
    mock.create_index = MagicMock(side_effect=create_index)
    mock.put_vectors = MagicMock(side_effect=put_vectors)
    mock.query_vectors = MagicMock(side_effect=query_vectors)
    mock.delete_vectors = MagicMock(side_effect=delete_vectors)
    mock.get_index = MagicMock(side_effect=get_index)
    mock.list_vectors = MagicMock(side_effect=list_vectors)
    mock.delete_index = MagicMock(side_effect=delete_index)
    mock.delete_vector_bucket = MagicMock(side_effect=delete_vector_bucket)

    return mock


# --- Bedrock Mocking ---


@pytest.fixture
def mock_bedrock() -> MagicMock:
    """Mock Bedrock Runtime for embedding generation."""
    mock = MagicMock()

    def invoke_model(body: str, modelId: str, **kwargs: Any) -> dict[str, Any]:
        request = json.loads(body)

        # Determine dimension from request (new format uses singleEmbeddingParams)
        dimension = 3072  # Default
        if "singleEmbeddingParams" in request:
            dimension = request["singleEmbeddingParams"].get("embeddingDimension", 3072)

        # Determine embedding type based on request content
        embedding_type = "TEXT"
        params = request.get("singleEmbeddingParams", {})
        if "image" in params:
            embedding_type = "IMAGE"
        elif "audio" in params:
            embedding_type = "AUDIO"
        elif "video" in params:
            embedding_type = "AUDIO_VIDEO_COMBINED"

        # Generate fake embedding (deterministic based on input for consistency)
        embedding = [0.1] * dimension

        # New response format with embeddings array
        response_body = json.dumps(
            {
                "embeddings": [
                    {
                        "embeddingType": embedding_type,
                        "embedding": embedding,
                    }
                ]
            }
        )
        return {"body": io.BytesIO(response_body.encode())}

    mock.invoke_model = MagicMock(side_effect=invoke_model)

    return mock


# --- Pricing API Mocking ---


@pytest.fixture
def mock_pricing() -> MagicMock:
    """Mock AWS Pricing API."""
    mock = MagicMock()

    def get_products(ServiceCode: str, **kwargs: Any) -> dict[str, Any]:
        # Return mock pricing data
        if ServiceCode == "AmazonS3":
            return {
                "PriceList": [
                    json.dumps(
                        {
                            "terms": {
                                "OnDemand": {
                                    "xxx": {
                                        "priceDimensions": {
                                            "yyy": {"pricePerUnit": {"USD": "0.023"}}
                                        }
                                    }
                                }
                            }
                        }
                    )
                ]
            }
        return {"PriceList": []}

    mock.get_products = MagicMock(side_effect=get_products)

    return mock


# --- Configuration Fixtures ---


@pytest.fixture
def test_config() -> Config:
    """Create a test configuration."""
    return Config(
        region="us-east-1",
        bucket="test-bucket",
        dimension=1024,
    )


# --- Sample File Fixtures ---


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a sample text file."""
    file = tmp_path / "sample.txt"
    file.write_text("This is sample text content for testing semantic storage.")
    return file


@pytest.fixture
def sample_json_file(tmp_path: Path) -> Path:
    """Create a sample JSON file."""
    file = tmp_path / "sample.json"
    file.write_text('{"key": "value", "items": [1, 2, 3]}')
    return file


@pytest.fixture
def sample_image_file(tmp_path: Path) -> Path:
    """Create a minimal valid PNG file (1x1 transparent pixel)."""
    file = tmp_path / "sample.png"
    # Minimal valid PNG: 1x1 transparent pixel
    png_data = bytes(
        [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,  # PNG signature
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,  # IHDR chunk
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,  # 1x1
            0x08,
            0x06,
            0x00,
            0x00,
            0x00,
            0x1F,
            0x15,
            0xC4,
            0x89,  # 8-bit RGBA
            0x00,
            0x00,
            0x00,
            0x0A,
            0x49,
            0x44,
            0x41,
            0x54,  # IDAT chunk
            0x78,
            0x9C,
            0x63,
            0x00,
            0x01,
            0x00,
            0x00,
            0x05,
            0x00,
            0x01,  # compressed data
            0x0D,
            0x0A,
            0x2D,
            0xB4,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,
            0x44,  # IEND
            0xAE,
            0x42,
            0x60,
            0x82,
        ]
    )
    file.write_bytes(png_data)
    return file


@pytest.fixture
def sample_config_file(tmp_path: Path) -> Path:
    """Create a sample config file."""
    file = tmp_path / "semstash.toml"
    file.write_text(
        """
[aws]
region = "us-east-1"
bucket = "config-bucket"

[embeddings]
dimension = 1024

[output]
format = "json"
"""
    )
    return file


# --- Sample Document Fixtures (for integration tests) ---


def _get_sample_file(filename: str) -> Path | None:
    """Get a sample file from tests/samples directory if it exists."""
    samples_dir = Path(__file__).parent / "samples"
    sample_file = samples_dir / filename
    if sample_file.exists():
        return sample_file
    return None


@pytest.fixture
def sample_pdf_file() -> Path:
    """Get sample PDF file from tests/samples/.

    Skips test if sample file doesn't exist.
    """
    sample = _get_sample_file("sample.pdf")
    if sample is None:
        pytest.skip("sample.pdf not found in tests/samples/")
    return sample


@pytest.fixture
def sample_docx_file() -> Path:
    """Get sample DOCX file from tests/samples/.

    Skips test if sample file doesn't exist.
    """
    sample = _get_sample_file("sample.docx")
    if sample is None:
        pytest.skip("sample.docx not found in tests/samples/")
    return sample


@pytest.fixture
def sample_pptx_file() -> Path:
    """Get sample PPTX file from tests/samples/.

    Skips test if sample file doesn't exist.
    """
    sample = _get_sample_file("sample.pptx")
    if sample is None:
        pytest.skip("sample.pptx not found in tests/samples/")
    return sample


@pytest.fixture
def sample_xlsx_file() -> Path:
    """Get sample XLSX file from tests/samples/.

    Skips test if sample file doesn't exist.
    """
    sample = _get_sample_file("sample.xlsx")
    if sample is None:
        pytest.skip("sample.xlsx not found in tests/samples/")
    return sample


@pytest.fixture
def sample_jpg_file() -> Path:
    """Get sample JPG file from tests/samples/.

    Skips test if sample file doesn't exist.
    """
    sample = _get_sample_file("sample.jpg")
    if sample is None:
        pytest.skip("sample.jpg not found in tests/samples/")
    return sample
