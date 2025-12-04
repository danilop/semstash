"""Test helper functions for query result validation.

These helpers catch bugs like returnDistance/returnMetadata not being set,
which would result in default values (score=1.0, missing metadata).
"""

from typing import Any


def assert_valid_search_result(
    result: Any,
    *,
    expected_key: str | None = None,
    expected_content_type: str | None = None,
    min_score: float = 0.0,
    max_score: float = 1.0,
    require_url: bool = False,
    require_file_size: bool = False,
) -> None:
    """Assert that a SearchResult has valid data.

    Args:
        result: A SearchResult object to validate
        expected_key: If set, assert key matches this value
        expected_content_type: If set, assert content_type matches
        min_score: Minimum valid score (default 0.0)
        max_score: Maximum valid score (default 1.0, exclusive for detecting defaults)
        require_url: If True, assert url is not None
        require_file_size: If True, assert file_size is not None and > 0
    """
    # Key is always required
    assert result.key is not None, "SearchResult.key should not be None"
    if expected_key is not None:
        assert result.key == expected_key, f"Expected key {expected_key}, got {result.key}"

    # Score should be a real value, not the default 1.0
    assert result.score is not None, "SearchResult.score should not be None"
    assert result.score >= min_score, f"Score {result.score} should be >= {min_score}"
    assert result.score < max_score, (
        f"Score {result.score} should be < {max_score} "
        "(score=1.0 often indicates returnDistance was not set)"
    )

    # Content type validation
    if expected_content_type is not None:
        assert result.content_type == expected_content_type, (
            f"Expected content_type {expected_content_type}, got {result.content_type}"
        )

    # Optional validations
    if require_url:
        assert result.url is not None, "SearchResult.url should not be None"

    if require_file_size:
        assert result.file_size is not None, "SearchResult.file_size should not be None"
        assert result.file_size > 0, f"file_size should be > 0, got {result.file_size}"


def assert_valid_query_results(
    results: list[Any],
    *,
    min_count: int = 1,
    expected_keys: list[str] | None = None,
) -> None:
    """Assert that a list of query results is valid.

    Args:
        results: List of SearchResult objects
        min_count: Minimum number of results expected
        expected_keys: If set, assert all these keys are in results
    """
    assert len(results) >= min_count, f"Expected at least {min_count} results, got {len(results)}"

    for result in results:
        # Basic validation for each result
        assert_valid_search_result(result, max_score=1.0)

    if expected_keys is not None:
        result_keys = {r.key for r in results}
        for key in expected_keys:
            assert key in result_keys, f"Expected key {key} in results, got {result_keys}"
