"""Storage operations for S3 and S3 Vectors.

Handles content storage in S3 and vector storage in S3 Vectors.
Provides unified interface for storing and retrieving semantic content.
"""

import json
import mimetypes
from datetime import datetime
from itertools import batched
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from semstash.config import (
    DEFAULT_DIMENSION,
    DEFAULT_INDEX_NAME,
    DEFAULT_PRESIGNED_URL_EXPIRY,
    DEFAULT_REGION,
    SUPPORTED_DIMENSIONS,
)
from semstash.exceptions import (
    ContentExistsError,
    ContentNotFoundError,
    DimensionError,
    NotInitializedError,
    StorageError,
)
from semstash.models import SearchResult, StashConfig, StorageItem
from semstash.utils import key_to_path

# Stash config file path in S3 bucket
STASH_CONFIG_KEY = ".semstash/config.json"


def _get_error_code(error: ClientError) -> str:
    """Extract error code from a ClientError response."""
    return error.response.get("Error", {}).get("Code", "")


class ContentStorage:
    """Manages content storage in Amazon S3.

    Example:
        storage = ContentStorage("my-bucket")

        # Upload content
        storage.upload(Path("document.pdf"), "docs/document.pdf")

        # Get content URL
        url = storage.get_presigned_url("docs/document.pdf")

        # List content
        for item in storage.list_objects(prefix="docs/"):
            print(f"{item.key}: {item.size} bytes")
    """

    def __init__(
        self,
        bucket: str,
        region: str = DEFAULT_REGION,
        client: Any | None = None,
    ) -> None:
        """Initialize content storage.

        Args:
            bucket: S3 bucket name.
            region: AWS region.
            client: Optional boto3 S3 client (for testing).
        """
        self.bucket = bucket
        self.region = region
        self._client = client
        self._bucket_verified = False

    @property
    def client(self) -> Any:
        """Get or create S3 client."""
        if self._client is None:
            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def ensure_bucket_exists(self) -> bool:
        """Ensure bucket exists, create if needed.

        Returns:
            True if bucket was created, False if already existed.

        Raises:
            StorageError: If bucket creation fails.
        """
        if self._bucket_verified:
            return False

        try:
            self.client.head_bucket(Bucket=self.bucket)
            self._bucket_verified = True
            return False
        except ClientError as e:
            error_code = _get_error_code(e)
            if error_code in ("404", "NoSuchBucket"):
                return self._create_bucket()
            if error_code in ("403", "Forbidden"):
                raise StorageError(
                    f"Bucket name '{self.bucket}' is not available. "
                    "S3 bucket names are globally unique. "
                    "Please choose a different name."
                ) from e
            raise StorageError(f"Failed to access bucket {self.bucket}: {e}") from e

    def _create_bucket(self) -> bool:
        """Create the S3 bucket.

        Returns:
            True if created successfully.

        Raises:
            StorageError: If creation fails.
        """
        try:
            # us-east-1 doesn't need LocationConstraint
            if self.region == "us-east-1":
                self.client.create_bucket(Bucket=self.bucket)
            else:
                self.client.create_bucket(
                    Bucket=self.bucket,
                    CreateBucketConfiguration={"LocationConstraint": self.region},
                )
            self._bucket_verified = True
            return True
        except ClientError as e:
            raise StorageError(f"Failed to create bucket {self.bucket}: {e}") from e

    def upload(
        self,
        file_path: Path,
        key: str | None = None,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        force: bool = False,
    ) -> StorageItem:
        """Upload a file to S3.

        Args:
            file_path: Local file path.
            key: S3 object key (defaults to filename).
            content_type: MIME type (auto-detected if None).
            metadata: Optional custom metadata.
            force: If True, overwrite existing content. If False (default),
                raise ContentExistsError if key already exists.

        Returns:
            StorageItem with upload details.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ContentExistsError: If key exists and force=False.
            StorageError: If upload fails.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if key is None:
            key = file_path.name

        if content_type is None:
            content_type, _ = mimetypes.guess_type(str(file_path))
            content_type = content_type or "application/octet-stream"

        self.ensure_bucket_exists()

        # Check if content already exists (unless force=True)
        if not force and self.exists(key):
            raise ContentExistsError(
                f"Content already exists at '{key}'. Use force=True to overwrite."
            )

        try:
            extra_args: dict[str, Any] = {"ContentType": content_type}
            if metadata:
                extra_args["Metadata"] = metadata

            self.client.upload_file(
                str(file_path),
                self.bucket,
                key,
                ExtraArgs=extra_args,
            )

            file_size = file_path.stat().st_size
            return StorageItem(
                key=key,
                path=key_to_path(key),
                content_type=content_type,
                file_size=file_size,
                created_at=datetime.now(),
                metadata=metadata or {},
            )

        except ClientError as e:
            raise StorageError(f"Failed to upload {key}: {e}") from e

    def download(self, key: str, destination: Path) -> Path:
        """Download an object from S3.

        Args:
            key: S3 object key.
            destination: Local destination path.

        Returns:
            Path to downloaded file.

        Raises:
            ContentNotFoundError: If object doesn't exist.
            StorageError: If download fails.
        """
        try:
            self.client.download_file(self.bucket, key, str(destination))
            return destination
        except ClientError as e:
            if _get_error_code(e) in ("404", "NoSuchKey"):
                raise ContentNotFoundError(f"Content not found: {key}") from e
            raise StorageError(f"Failed to download {key}: {e}") from e

    def delete(self, key: str) -> bool:
        """Delete an object from S3.

        Args:
            key: S3 object key.

        Returns:
            True if deleted successfully.

        Raises:
            StorageError: If deletion fails.
        """
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            raise StorageError(f"Failed to delete {key}: {e}") from e

    def get_presigned_url(self, key: str, expiry: int = DEFAULT_PRESIGNED_URL_EXPIRY) -> str:
        """Generate a presigned URL for an object.

        Args:
            key: S3 object key.
            expiry: URL expiration time in seconds.

        Returns:
            Presigned URL string.

        Raises:
            StorageError: If URL generation fails.
        """
        try:
            url: str = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expiry,
            )
            return url
        except ClientError as e:
            raise StorageError(f"Failed to generate URL for {key}: {e}") from e

    def get_metadata(self, key: str) -> StorageItem:
        """Get metadata for an object.

        Args:
            key: S3 object key.

        Returns:
            StorageItem with object metadata.

        Raises:
            ContentNotFoundError: If object doesn't exist.
            StorageError: If metadata retrieval fails.
        """
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
            return StorageItem(
                key=key,
                path=key_to_path(key),
                content_type=response.get("ContentType", "application/octet-stream"),
                file_size=response.get("ContentLength", 0),
                created_at=response.get("LastModified", datetime.now()),
                metadata=response.get("Metadata") or {},
            )
        except ClientError as e:
            if _get_error_code(e) in ("404", "NoSuchKey"):
                raise ContentNotFoundError(f"Content not found: {key}") from e
            raise StorageError(f"Failed to get metadata for {key}: {e}") from e

    def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> tuple[list[StorageItem], str | None]:
        """List objects in the bucket.

        Args:
            prefix: Filter by key prefix.
            max_keys: Maximum number of objects to return.
            continuation_token: Token for pagination.

        Returns:
            Tuple of (items list, next continuation token or None).

        Raises:
            StorageError: If listing fails.
        """
        try:
            params: dict[str, Any] = {
                "Bucket": self.bucket,
                "MaxKeys": max_keys,
            }
            if prefix:
                params["Prefix"] = prefix
            if continuation_token:
                params["ContinuationToken"] = continuation_token

            response = self.client.list_objects_v2(**params)

            items = []
            for obj in response.get("Contents", []):
                key = obj["Key"]
                # Exclude internal files from browse results
                if key.startswith(".semstash/"):
                    continue
                items.append(
                    StorageItem(
                        key=key,
                        path=key_to_path(key),
                        content_type="",  # Not available in list response
                        file_size=obj["Size"],
                        created_at=obj["LastModified"],
                    )
                )

            next_token = response.get("NextContinuationToken")
            return items, next_token

        except ClientError as e:
            raise StorageError(f"Failed to list objects: {e}") from e

    def exists(self, key: str) -> bool:
        """Check if an object exists.

        Args:
            key: S3 object key.

        Returns:
            True if object exists, False otherwise.
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def bucket_exists(self) -> bool:
        """Check if the bucket exists.

        Returns:
            True if bucket exists, False otherwise.
        """
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError:
            return False

    def list_all_keys(self) -> set[str]:
        """List all object keys in the bucket.

        Returns all keys without full metadata for efficiency.
        Used for consistency checking. Excludes internal files like config.

        Returns:
            Set of all object keys.

        Raises:
            StorageError: If listing fails.
        """
        keys: set[str] = set()
        continuation_token: str | None = None

        while True:
            try:
                params: dict[str, Any] = {
                    "Bucket": self.bucket,
                    "MaxKeys": 1000,
                }
                if continuation_token:
                    params["ContinuationToken"] = continuation_token

                response = self.client.list_objects_v2(**params)

                for obj in response.get("Contents", []):
                    key = obj["Key"]
                    # Exclude internal files
                    if not key.startswith(".semstash/"):
                        keys.add(key)

                if not response.get("IsTruncated", False):
                    break

                continuation_token = response.get("NextContinuationToken")

            except ClientError as e:
                raise StorageError(f"Failed to list objects: {e}") from e

        return keys

    def delete_all_objects(self) -> int:
        """Delete all objects in the bucket.

        Returns:
            Number of objects deleted.

        Raises:
            StorageError: If deletion fails.
        """
        deleted_count = 0
        continuation_token: str | None = None

        while True:
            try:
                params: dict[str, Any] = {
                    "Bucket": self.bucket,
                    "MaxKeys": 1000,
                }
                if continuation_token:
                    params["ContinuationToken"] = continuation_token

                response = self.client.list_objects_v2(**params)
                objects = response.get("Contents", [])

                if objects:
                    delete_objects = [{"Key": obj["Key"]} for obj in objects]
                    self.client.delete_objects(
                        Bucket=self.bucket,
                        Delete={"Objects": delete_objects},
                    )
                    deleted_count += len(objects)

                if not response.get("IsTruncated", False):
                    break

                continuation_token = response.get("NextContinuationToken")

            except ClientError as e:
                raise StorageError(f"Failed to delete objects: {e}") from e

        return deleted_count

    def delete_bucket(self) -> bool:
        """Delete the S3 bucket.

        Bucket must be empty before deletion.

        Returns:
            True if deleted successfully.

        Raises:
            StorageError: If deletion fails (e.g., bucket not empty).
        """
        try:
            self.client.delete_bucket(Bucket=self.bucket)
            self._bucket_verified = False
            return True
        except ClientError as e:
            raise StorageError(f"Failed to delete bucket {self.bucket}: {e}") from e

    def save_config(self, config: StashConfig) -> None:
        """Save stash configuration to S3.

        Stores the config at .semstash/config.json in the bucket.

        Args:
            config: StashConfig to save.

        Raises:
            StorageError: If save fails.
        """
        self.ensure_bucket_exists()
        try:
            config_json = config.model_dump_json(indent=2)
            self.client.put_object(
                Bucket=self.bucket,
                Key=STASH_CONFIG_KEY,
                Body=config_json.encode("utf-8"),
                ContentType="application/json",
            )
        except ClientError as e:
            raise StorageError(f"Failed to save config: {e}") from e

    def load_config(self) -> StashConfig | None:
        """Load stash configuration from S3.

        Returns:
            StashConfig if found, None otherwise.

        Raises:
            StorageError: If load fails (not including missing config).
        """
        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=STASH_CONFIG_KEY,
            )
            config_json = response["Body"].read().decode("utf-8")
            config_data = json.loads(config_json)
            return StashConfig(**config_data)
        except ClientError as e:
            if _get_error_code(e) in ("404", "NoSuchKey"):
                return None
            raise StorageError(f"Failed to load config: {e}") from e

    def config_exists(self) -> bool:
        """Check if stash config exists in bucket.

        Returns:
            True if config exists, False otherwise.
        """
        return self.exists(STASH_CONFIG_KEY)


class VectorStorage:
    """Manages vector storage in Amazon S3 Vectors.

    Example:
        storage = VectorStorage("my-bucket-vectors")

        # Initialize index
        storage.ensure_index_exists()

        # Store vector
        storage.put_vector("doc-1", [0.1, 0.2, ...], {"title": "Doc 1"})

        # Query similar vectors
        results = storage.query([0.1, 0.2, ...], top_k=5)
        for result in results:
            print(f"{result.key}: {result.score}")
    """

    def __init__(
        self,
        bucket: str,
        region: str = DEFAULT_REGION,
        dimension: int = DEFAULT_DIMENSION,
        index_name: str = DEFAULT_INDEX_NAME,
        client: Any | None = None,
    ) -> None:
        """Initialize vector storage.

        Args:
            bucket: S3 Vectors bucket name.
            region: AWS region (must support S3 Vectors).
            dimension: Embedding dimension.
            index_name: Name for the vector index.
            client: Optional boto3 s3vectors client (for testing).

        Raises:
            DimensionError: If dimension is not supported.
        """
        if dimension not in SUPPORTED_DIMENSIONS:
            raise DimensionError(
                f"Invalid dimension {dimension}. "
                f"Supported: {', '.join(str(d) for d in SUPPORTED_DIMENSIONS)}"
            )

        self.bucket = bucket
        self.region = region
        self.dimension = dimension
        self.index_name = index_name
        self._client = client
        self._initialized = False

    @property
    def client(self) -> Any:
        """Get or create S3 Vectors client."""
        if self._client is None:
            self._client = boto3.client("s3vectors", region_name=self.region)
        return self._client

    def ensure_index_exists(self) -> bool:
        """Ensure vector bucket and index exist, create if needed.

        Returns:
            True if created, False if already existed.

        Raises:
            StorageError: If creation fails.
        """
        if self._initialized:
            return False

        try:
            # Check if index exists
            self.client.get_index(
                vectorBucketName=self.bucket,
                indexName=self.index_name,
            )
            self._initialized = True
            return False
        except ClientError as e:
            code = _get_error_code(e)
            if "NotFound" in code or "NoSuch" in code:
                return self._create_index()
            raise StorageError(f"Failed to access index: {e}") from e

    def _create_index(self) -> bool:
        """Create vector bucket and index.

        Returns:
            True if created successfully.

        Raises:
            StorageError: If creation fails.
        """
        try:
            # Create vector bucket
            try:
                self.client.create_vector_bucket(vectorBucketName=self.bucket)
            except ClientError as e:
                # Bucket may already exist
                if "AlreadyExists" not in str(e):
                    raise

            # Create index
            self.client.create_index(
                vectorBucketName=self.bucket,
                indexName=self.index_name,
                dimension=self.dimension,
                distanceMetric="cosine",
                dataType="float32",
            )
            self._initialized = True
            return True

        except ClientError as e:
            raise StorageError(f"Failed to create index: {e}") from e

    def put_vector(
        self,
        key: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a vector with optional metadata.

        Args:
            key: Unique identifier for the vector.
            vector: Embedding vector.
            metadata: Optional metadata dictionary.

        Raises:
            NotInitializedError: If index not initialized.
            StorageError: If storage fails.
        """
        if not self._initialized:
            raise NotInitializedError(
                "Vector index not initialized. Call ensure_index_exists() first."
            )

        try:
            vector_data: dict[str, Any] = {
                "key": key,
                "data": {"float32": vector},
            }
            if metadata:
                vector_data["metadata"] = metadata

            self.client.put_vectors(
                vectorBucketName=self.bucket,
                indexName=self.index_name,
                vectors=[vector_data],
            )
        except ClientError as e:
            raise StorageError(f"Failed to store vector {key}: {e}") from e

    def put_vectors_batch(
        self,
        vectors: list[tuple[str, list[float], dict[str, Any] | None]],
    ) -> int:
        """Store multiple vectors efficiently in batches.

        Uses S3 Vectors batch API to store up to 500 vectors per request.
        Automatically batches larger lists.

        Args:
            vectors: List of (key, vector, metadata) tuples.

        Returns:
            Number of vectors stored.

        Raises:
            NotInitializedError: If index not initialized.
            StorageError: If storage fails.

        Example:
            vectors = [
                ("doc.pdf#page=1", [0.1, 0.2, ...], {"page": 1}),
                ("doc.pdf#page=2", [0.3, 0.4, ...], {"page": 2}),
            ]
            count = storage.put_vectors_batch(vectors)
        """
        if not self._initialized:
            raise NotInitializedError(
                "Vector index not initialized. Call ensure_index_exists() first."
            )

        if not vectors:
            return 0

        stored_count = 0
        # S3 Vectors allows up to 500 vectors per put_vectors call
        batch_size = 500

        for batch in batched(vectors, batch_size, strict=False):
            try:
                vector_data_list = []
                for key, vector, metadata in batch:
                    vector_data: dict[str, Any] = {
                        "key": key,
                        "data": {"float32": vector},
                    }
                    if metadata:
                        vector_data["metadata"] = metadata
                    vector_data_list.append(vector_data)

                self.client.put_vectors(
                    vectorBucketName=self.bucket,
                    indexName=self.index_name,
                    vectors=vector_data_list,
                )
                stored_count += len(vector_data_list)

            except ClientError as e:
                raise StorageError(f"Failed to store vector batch: {e}") from e

        return stored_count

    def get_vector(self, key: str) -> tuple[list[float], dict[str, Any] | None]:
        """Retrieve a vector by key.

        Args:
            key: Vector key.

        Returns:
            Tuple of (vector, metadata).

        Raises:
            ContentNotFoundError: If vector doesn't exist.
            StorageError: If retrieval fails.
        """
        try:
            response = self.client.get_vectors(
                vectorBucketName=self.bucket,
                indexName=self.index_name,
                keys=[key],
            )
            vectors = response.get("vectors", [])
            if not vectors:
                raise ContentNotFoundError(f"Vector not found: {key}")

            vector_data = vectors[0]
            vector = vector_data.get("data", {}).get("float32", [])
            metadata = vector_data.get("metadata")
            return vector, metadata

        except ClientError as e:
            if "NotFound" in _get_error_code(e):
                raise ContentNotFoundError(f"Vector not found: {key}") from e
            raise StorageError(f"Failed to get vector {key}: {e}") from e

    def delete_vector(self, key: str) -> bool:
        """Delete a vector by key.

        Args:
            key: Vector key.

        Returns:
            True if deleted successfully.

        Raises:
            StorageError: If deletion fails.
        """
        try:
            self.client.delete_vectors(
                vectorBucketName=self.bucket,
                indexName=self.index_name,
                keys=[key],
            )
            return True
        except ClientError as e:
            raise StorageError(f"Failed to delete vector {key}: {e}") from e

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter_expression: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query for similar vectors.

        Args:
            vector: Query vector.
            top_k: Maximum number of results.
            filter_expression: Optional metadata filter.

        Returns:
            List of SearchResult sorted by similarity.

        Raises:
            NotInitializedError: If index not initialized.
            StorageError: If query fails.
        """
        if not self._initialized:
            raise NotInitializedError(
                "Vector index not initialized. Call ensure_index_exists() first."
            )

        try:
            params: dict[str, Any] = {
                "vectorBucketName": self.bucket,
                "indexName": self.index_name,
                "queryVector": {"float32": vector},
                "topK": top_k,
                "returnDistance": True,
                "returnMetadata": True,
            }
            if filter_expression:
                params["filter"] = filter_expression

            response = self.client.query_vectors(**params)

            results = []
            for match in response.get("vectors", []):
                # Convert distance to similarity score (1 - distance for cosine)
                distance = match.get("distance", 0)
                score = 1 - distance

                results.append(
                    SearchResult(
                        key=match["key"],
                        score=score,
                        distance=distance,
                        metadata=match.get("metadata") or {},
                    )
                )

            return results

        except ClientError as e:
            raise StorageError(f"Vector query failed: {e}") from e

    def get_stats(self) -> dict[str, Any]:
        """Get index configuration.

        Returns:
            Dictionary with dimension and distance_metric.

        Raises:
            StorageError: If retrieval fails.
        """
        try:
            response = self.client.get_index(
                vectorBucketName=self.bucket,
                indexName=self.index_name,
            )
            # Index details are nested under 'index' key
            index_info = response.get("index", {})

            return {
                "dimension": index_info.get("dimension", self.dimension),
                "distance_metric": index_info.get("distanceMetric", "cosine"),
            }
        except ClientError as e:
            raise StorageError(f"Failed to get index stats: {e}") from e

    def index_exists(self) -> bool:
        """Check if the vector index exists.

        Returns:
            True if index exists, False otherwise.
        """
        try:
            self.client.get_index(
                vectorBucketName=self.bucket,
                indexName=self.index_name,
            )
            return True
        except ClientError:
            return False

    def list_all_keys(self) -> set[str]:
        """List all vector keys in the index.

        Returns all keys for consistency checking.

        Returns:
            Set of all vector keys.

        Raises:
            NotInitializedError: If index not initialized.
            StorageError: If listing fails.
        """
        if not self._initialized:
            raise NotInitializedError(
                "Vector index not initialized. Call ensure_index_exists() first."
            )

        keys: set[str] = set()
        next_token: str | None = None

        while True:
            try:
                params: dict[str, Any] = {
                    "vectorBucketName": self.bucket,
                    "indexName": self.index_name,
                    "maxResults": 1000,
                }
                if next_token:
                    params["nextToken"] = next_token

                response = self.client.list_vectors(**params)

                for vector in response.get("vectors", []):
                    keys.add(vector["key"])

                next_token = response.get("nextToken")
                if not next_token:
                    break

            except ClientError as e:
                raise StorageError(f"Failed to list vectors: {e}") from e

        return keys

    def delete_vectors_batch(self, keys: list[str]) -> int:
        """Delete multiple vectors by keys.

        Args:
            keys: List of vector keys to delete.

        Returns:
            Number of vectors deleted.

        Raises:
            StorageError: If deletion fails.
        """
        if not keys:
            return 0

        try:
            self.client.delete_vectors(
                vectorBucketName=self.bucket,
                indexName=self.index_name,
                keys=keys,
            )
            return len(keys)
        except ClientError as e:
            raise StorageError(f"Failed to delete vectors: {e}") from e

    def delete_all_vectors(self) -> int:
        """Delete all vectors in the index.

        Returns:
            Number of vectors deleted.

        Raises:
            NotInitializedError: If index not initialized.
            StorageError: If deletion fails.
        """
        if not self._initialized:
            raise NotInitializedError("Vector index not initialized.")

        keys = list(self.list_all_keys())
        if not keys:
            return 0

        # Delete in batches of 100 using itertools.batched
        deleted_count = 0
        for batch in batched(keys, 100, strict=False):
            deleted_count += self.delete_vectors_batch(list(batch))

        return deleted_count

    def delete_index(self) -> bool:
        """Delete the vector index.

        Returns:
            True if deleted successfully.

        Raises:
            StorageError: If deletion fails.
        """
        try:
            self.client.delete_index(
                vectorBucketName=self.bucket,
                indexName=self.index_name,
            )
            self._initialized = False
            return True
        except ClientError as e:
            raise StorageError(f"Failed to delete index: {e}") from e

    def delete_vector_bucket(self) -> bool:
        """Delete the S3 Vectors bucket.

        Index must be deleted first.

        Returns:
            True if deleted successfully.

        Raises:
            StorageError: If deletion fails.
        """
        try:
            self.client.delete_vector_bucket(vectorBucketName=self.bucket)
            return True
        except ClientError as e:
            raise StorageError(f"Failed to delete vector bucket {self.bucket}: {e}") from e
