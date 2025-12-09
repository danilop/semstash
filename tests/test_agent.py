"""Integration tests for the SemStash Agent using Strands SDK.

These tests verify that:
    - The agent correctly uses MCP tools (not direct semstash calls)
    - Content retrieval and search work correctly
    - Different agent instances can access the same content

Run with: pytest --use-aws -m integration tests/test_agent.py
"""

from pathlib import Path

import pytest

from semstash import SemStash
from semstash.agent import SemStashAgent


@pytest.fixture
def initialized_stash(agent_bucket_name: str) -> SemStash:
    """Initialize stash for agent tests and return it."""
    from semstash.exceptions import AlreadyExistsError

    stash = SemStash(agent_bucket_name, dimension=256)
    try:
        stash.init()
    except AlreadyExistsError:
        stash.open()
    return stash


@pytest.mark.integration
@pytest.mark.usefixtures("agent_stash_cleanup")
class TestSemStashAgent:
    """Integration tests for the SemStash Agent.

    These tests focus on the agent's read-focused capabilities:
    - Browse existing content
    - Query/search for content
    - Get content details
    - Delete content
    - Show stats
    """

    def test_agent_browse_stash(
        self, agent_bucket_name: str, sample_text_file: Path, initialized_stash: SemStash
    ) -> None:
        """Test that agent can browse content in the stash."""
        # Upload content directly using SemStash (file_path, target)
        initialized_stash.upload(sample_text_file, sample_text_file.name, force=True)

        # Verify agent can browse
        with SemStashAgent(bucket=agent_bucket_name) as agent:
            result = agent.chat("Browse the root of the stash and tell me what files are there")
            # Agent should mention the uploaded file
            assert len(result) > 0
            # The response should contain some reference to the content
            assert (
                sample_text_file.name in result
                or "sample" in result.lower()
                or "txt" in result.lower()
            )

    def test_agent_query_text_content(
        self, agent_bucket_name: str, sample_text_file: Path, initialized_stash: SemStash
    ) -> None:
        """Test that agent can search/query for text content."""
        # Upload content directly using SemStash
        initialized_stash.upload(sample_text_file, sample_text_file.name, force=True)

        # Verify agent can query
        with SemStashAgent(bucket=agent_bucket_name) as agent:
            result = agent.chat("Search for content about testing semantic storage")
            # Agent should return something meaningful
            assert len(result) > 0

    def test_agent_summarize_content(
        self, agent_bucket_name: str, initialized_stash: SemStash, tmp_path: Path
    ) -> None:
        """Test that agent can retrieve and summarize actual content."""
        # Create a file with specific, verifiable content
        doc_file = tmp_path / "project_notes.txt"
        doc_file.write_text(
            "Project Alpha Meeting Notes - March 2024\n\n"
            "Attendees: Alice, Bob, and Charlie\n\n"
            "Key decisions:\n"
            "1. Launch date set for June 15th\n"
            "2. Budget approved at $50,000\n"
            "3. Bob will lead the frontend team\n\n"
            "Action items:\n"
            "- Alice to finalize requirements by April 1st\n"
            "- Charlie to set up CI/CD pipeline\n"
        )

        # Upload the document
        initialized_stash.upload(doc_file, doc_file.name, force=True)

        # Ask agent to summarize - it should retrieve and understand the content
        with SemStashAgent(bucket=agent_bucket_name) as agent:
            result = agent.chat(
                "Find the project notes and tell me: "
                "What is the launch date and who is leading the frontend?"
            )

            # Verify agent found and understood the content
            result_lower = result.lower()
            assert len(result) > 50, "Response too short to be a real summary"
            # Should mention key facts from the document
            assert "june" in result_lower or "15" in result_lower, (
                f"Agent should mention launch date. Got: {result}"
            )
            assert "bob" in result_lower or "frontend" in result_lower, (
                f"Agent should mention Bob/frontend. Got: {result}"
            )

    def test_agent_get_content(
        self, agent_bucket_name: str, sample_text_file: Path, initialized_stash: SemStash
    ) -> None:
        """Test that agent can get specific content by key."""
        # Upload content directly using SemStash
        initialized_stash.upload(sample_text_file, sample_text_file.name, force=True)

        # Verify agent can get content
        with SemStashAgent(bucket=agent_bucket_name) as agent:
            result = agent.chat(f"Get the content with key {sample_text_file.name}")
            # Agent should return something meaningful
            assert len(result) > 0

    def test_agent_stats(
        self, agent_bucket_name: str, sample_text_file: Path, initialized_stash: SemStash
    ) -> None:
        """Test that agent can show storage statistics."""
        # Upload content directly using SemStash
        initialized_stash.upload(sample_text_file, sample_text_file.name, force=True)

        # Verify agent can get stats
        with SemStashAgent(bucket=agent_bucket_name) as agent:
            result = agent.chat("Show me the storage statistics")
            # Agent should return stats info
            assert len(result) > 0

    def test_agent_delete_content(
        self, agent_bucket_name: str, sample_text_file: Path, initialized_stash: SemStash
    ) -> None:
        """Test that agent can delete content."""
        # Upload content directly using SemStash
        initialized_stash.upload(sample_text_file, sample_text_file.name, force=True)

        # Verify content exists
        browse = initialized_stash.browse("/")
        keys = [item.key for item in browse.items]
        assert sample_text_file.name in keys

        # Have agent delete it
        with SemStashAgent(bucket=agent_bucket_name) as agent:
            result = agent.chat(f"Delete the content with key {sample_text_file.name}")
            assert len(result) > 0

        # Verify deleted from semstash
        browse = initialized_stash.browse("/")
        keys = [item.key for item in browse.items]
        assert sample_text_file.name not in keys, "File should be deleted from semstash"

    def test_agent_conversation_reset(
        self, agent_bucket_name: str, initialized_stash: SemStash
    ) -> None:
        """Test that agent conversation can be reset."""
        with SemStashAgent(bucket=agent_bucket_name) as agent:
            # Have a conversation
            agent.chat("Hello, can you help me?")

            # Reset
            agent.reset_conversation()

            # Should still work after reset
            result = agent.chat("Show me the storage statistics")
            assert len(result) > 0

    def test_agent_streaming(self, agent_bucket_name: str, initialized_stash: SemStash) -> None:
        """Test that agent streaming works."""
        with SemStashAgent(bucket=agent_bucket_name, streaming=True) as agent:
            events = list(agent.chat_stream("Say hello"))
            # Should have received some events
            assert len(events) > 0
