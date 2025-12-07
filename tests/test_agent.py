"""Integration tests for the SemStash Agent using Strands SDK.

These tests verify that:
    - The agent correctly uses MCP tools (not direct semstash calls)
    - Content is actually stored in semstash via MCP
    - Different agent instances can access the same content

Run with: pytest --use-aws -m integration tests/test_agent.py
"""

from pathlib import Path

import pytest

from examples.strands_agent import semstash_agent
from semstash import SemStash


def get_response_text(result: object) -> str:
    """Extract text from agent response."""
    if hasattr(result, "message") and result.message:
        content = result.message.get("content", [])
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
    return str(result)


@pytest.mark.integration
@pytest.mark.usefixtures("agent_stash_cleanup")
class TestSemStashAgent:
    """Integration tests for the SemStash Agent."""

    def test_agent_upload_then_new_agent_retrieves(
        self, agent_bucket_name: str, sample_text_file: Path
    ) -> None:
        """Test that content uploaded by one agent can be retrieved by a new agent instance."""
        # First agent uploads
        with semstash_agent(agent_bucket_name) as agent1:
            upload_result = get_response_text(agent1(f"Upload the file at {sample_text_file}"))
            assert "upload" in upload_result.lower() or sample_text_file.name in upload_result

        # Verify content exists in semstash directly (proves MCP tool worked)
        stash = SemStash(bucket=agent_bucket_name, auto_init=False)
        browse = stash.browse("/")
        keys = [item.key for item in browse.items]
        assert sample_text_file.name in keys, f"File not found in semstash. Keys: {keys}"

        # Second agent (new instance) retrieves
        with semstash_agent(agent_bucket_name) as agent2:
            retrieve_result = get_response_text(
                agent2(f"Retrieve the content with key: {sample_text_file.name}")
            )
            assert (
                sample_text_file.name in retrieve_result
                or "url" in retrieve_result.lower()
                or "https://" in retrieve_result
            )

    def test_agent_upload_then_new_agent_searches(
        self, agent_bucket_name: str, sample_image_file: Path
    ) -> None:
        """Test that content uploaded by one agent can be searched by a new agent instance."""
        # First agent uploads image
        with semstash_agent(agent_bucket_name) as agent1:
            upload_result = get_response_text(agent1(f"Upload the file at {sample_image_file}"))
            assert "upload" in upload_result.lower() or sample_image_file.name in upload_result

        # Verify in semstash directly
        stash = SemStash(bucket=agent_bucket_name, auto_init=False)
        browse = stash.browse("/")
        keys = [item.key for item in browse.items]
        assert sample_image_file.name in keys, f"Image not found in semstash. Keys: {keys}"

        # Second agent searches
        with semstash_agent(agent_bucket_name) as agent2:
            search_result = get_response_text(agent2("Search for images"))
            assert len(search_result) > 0

    def test_agent_upload_verified_in_semstash(
        self, agent_bucket_name: str, sample_json_file: Path
    ) -> None:
        """Test that agent upload actually stores content in semstash."""
        stash = SemStash(bucket=agent_bucket_name, auto_init=False)
        initial_stats = stash.get_stats()
        initial_count = initial_stats.content_count

        # Agent uploads
        with semstash_agent(agent_bucket_name) as agent:
            agent(f"Upload the file at {sample_json_file}")

        # Verify count increased
        final_stats = stash.get_stats()
        assert final_stats.content_count > initial_count, (
            "Content count should increase after upload"
        )

        # Verify file exists
        browse = stash.browse("/")
        keys = [item.key for item in browse.items]
        assert sample_json_file.name in keys

    def test_agent_delete_verified_in_semstash(
        self, agent_bucket_name: str, sample_text_file: Path
    ) -> None:
        """Test that agent delete actually removes content from semstash."""
        stash = SemStash(bucket=agent_bucket_name, auto_init=False)

        # First agent uploads
        with semstash_agent(agent_bucket_name) as agent1:
            agent1(f"Upload the file at {sample_text_file}")

        # Verify exists
        browse = stash.browse("/")
        keys = [item.key for item in browse.items]
        assert sample_text_file.name in keys

        # Second agent deletes
        with semstash_agent(agent_bucket_name) as agent2:
            delete_result = get_response_text(
                agent2(f"Delete the content with key: {sample_text_file.name}")
            )
            assert "delete" in delete_result.lower() or sample_text_file.name in delete_result

        # Verify deleted from semstash
        browse = stash.browse("/")
        keys = [item.key for item in browse.items]
        assert sample_text_file.name not in keys, "File should be deleted from semstash"
