"""Custom exceptions for semstash.

All exceptions inherit from SemStashError for easy catching.
Each exception provides helpful, actionable error messages.
"""


class SemStashError(Exception):
    """Base exception for all semstash errors.

    Example:
        try:
            stash.upload("file.txt")
        except SemStashError as e:
            print(f"semstash error: {e}")
    """


class ConfigurationError(SemStashError):
    """Invalid configuration.

    Raised when configuration values are invalid or missing.

    Example:
        ConfigurationError("dimension must be 256, 384, 1024, or 3072")
    """


class NotInitializedError(SemStashError):
    """Stash not initialized.

    Raised when operations are attempted before init.

    Example:
        NotInitializedError("Bucket not configured. Use SemStash('bucket-name')")
    """


class BucketNotFoundError(SemStashError):
    """S3 or S3 Vectors bucket not found.

    Raised when the expected bucket doesn't exist.

    Example:
        BucketNotFoundError("Bucket 'my-bucket' not found. Run 'semstash init' first.")
    """


class ContentNotFoundError(SemStashError):
    """Content not found in storage.

    Raised when trying to retrieve or delete non-existent content.

    Example:
        ContentNotFoundError("No content found with key 'photo.jpg'")
    """


class ContentExistsError(SemStashError):
    """Content already exists at the specified key.

    Raised when uploading to an existing key without force=True.

    Example:
        ContentExistsError(
            "Content already exists at 'photo.jpg'. "
            "Use force=True to overwrite."
        )
    """


class EmbeddingError(SemStashError):
    """Error generating embedding.

    Raised when the embedding API fails.

    Example:
        EmbeddingError("Failed to generate embedding: API timeout")
    """


class UnsupportedContentTypeError(SemStashError):
    """Content type not supported for embedding.

    Raised when trying to embed unsupported file types.

    Example:
        UnsupportedContentTypeError(
            "Cannot embed application/x-executable. "
            "Supported: text, image, audio, video"
        )
    """


class RegionNotSupportedError(SemStashError):
    """Region not supported for service.

    Raised when using an unsupported region.

    Example:
        RegionNotSupportedError(
            "Nova Embeddings only available in us-east-1. "
            "Current region: eu-west-1"
        )
    """


class DimensionError(SemStashError):
    """Invalid embedding dimension.

    Raised when dimension is not one of the supported values.

    Example:
        DimensionError("dimension must be 256, 384, 1024, or 3072, got 512")
    """


class StorageError(SemStashError):
    """General storage operation error.

    Raised for S3 or S3 Vectors operation failures.

    Example:
        StorageError("Failed to upload: Access Denied")
    """


class AlreadyExistsError(SemStashError):
    """Resource already exists.

    Raised when trying to create a bucket or stash that already exists.

    Example:
        AlreadyExistsError("Bucket 'my-bucket' already exists. Use open() to use existing bucket.")
    """
