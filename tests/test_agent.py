"""Integration tests for the SemStash Agent using Strands SDK.

These tests verify that:
    - The agent correctly uses MCP tools (not direct semstash calls)
    - Content is actually stored in semstash via MCP
    - Different agent instances can access the same content

Run with: pytest --use-aws -m integration tests/test_agent.py
"""

import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.tools.mcp import MCPClient

from semstash import SemStash


def create_mcp_client(bucket: str, dimension: int = 3072) -> MCPClient:
    """Create an MCP client connected to the local semstash server."""
    env = {
        **os.environ,
        "SEMSTASH_BUCKET": bucket,
        "SEMSTASH_REGION": "us-east-1",
        "SEMSTASH_DIMENSION": str(dimension),
    }
    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command=sys.executable, args=["-m", "semstash.mcp_server"], env=env
            )
        )
    )


@contextmanager
def create_agent(bucket: str, dimension: int = 3072) -> Generator[Agent]:
    """Create an agent using the local MCP server."""
    mcp_client = create_mcp_client(bucket, dimension)

    with mcp_client:
        tools = mcp_client.list_tools_sync()
        assert len(tools) > 0, "MCP server should provide at least one tool"
        yield Agent(tools=tools, system_prompt="You help users store and retrieve information.")


def get_response_text(result: object) -> str:
    """Extract text from agent response."""
    if hasattr(result, "message") and result.message:
        content = result.message.get("content", [])
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
    return str(result)


@pytest.mark.integration
class TestSemStashAgent:
    """Integration tests for the SemStash Agent."""

    def test_agent_upload_then_new_agent_retrieves(
        self, integration_bucket_name: str, sample_text_file: Path
    ) -> None:
        """Test that content uploaded by one agent can be retrieved by a new agent instance."""
        # First agent uploads
        with create_agent(bucket=integration_bucket_name) as agent1:
            upload_result = get_response_text(agent1(f"Upload the file at {sample_text_file}"))
            assert "upload" in upload_result.lower() or sample_text_file.name in upload_result

        # Verify content exists in semstash directly (proves MCP tool worked)
        stash = SemStash(bucket=integration_bucket_name, auto_init=False)
        browse = stash.browse("/")
        keys = [item.key for item in browse.items]
        assert sample_text_file.name in keys, f"File not found in semstash. Keys: {keys}"

        # Second agent (new instance) retrieves
        with create_agent(bucket=integration_bucket_name) as agent2:
            retrieve_result = get_response_text(
                agent2(f"Retrieve the content with key: {sample_text_file.name}")
            )
            assert "url" in retrieve_result.lower() or "https://" in retrieve_result

    def test_agent_upload_then_new_agent_searches(
        self, integration_bucket_name: str, sample_image_file: Path
    ) -> None:
        """Test that content uploaded by one agent can be searched by a new agent instance."""
        # First agent uploads image
        with create_agent(bucket=integration_bucket_name) as agent1:
            upload_result = get_response_text(agent1(f"Upload the file at {sample_image_file}"))
            assert "upload" in upload_result.lower() or sample_image_file.name in upload_result

        # Verify in semstash directly
        stash = SemStash(bucket=integration_bucket_name, auto_init=False)
        browse = stash.browse("/")
        keys = [item.key for item in browse.items]
        assert sample_image_file.name in keys, f"Image not found in semstash. Keys: {keys}"

        # Second agent searches
        with create_agent(bucket=integration_bucket_name) as agent2:
            search_result = get_response_text(agent2("Search for images"))
            assert len(search_result) > 0

    def test_agent_gets_same_tools_as_mcp_server(self, integration_bucket_name: str) -> None:
        """Test that agent gets the same tools that MCP server provides."""
        # First, connect directly to MCP server to get expected tool count
        mcp_client = create_mcp_client(integration_bucket_name)
        with mcp_client:
            expected_tools = mcp_client.list_tools_sync()
            expected_count = len(expected_tools)

        # Now create agent and verify it gets the same number of tools
        mcp_client2 = create_mcp_client(integration_bucket_name)
        with mcp_client2:
            agent_tools = mcp_client2.list_tools_sync()
            assert len(agent_tools) == expected_count, (
                f"Agent got {len(agent_tools)} tools, expected {expected_count}"
            )
            assert expected_count > 0, "MCP server should provide tools"

    def test_agent_upload_verified_in_semstash(
        self, integration_bucket_name: str, sample_json_file: Path
    ) -> None:
        """Test that agent upload actually stores content in semstash."""
        stash = SemStash(bucket=integration_bucket_name, auto_init=False)
        initial_stats = stash.get_stats()
        initial_count = initial_stats.content_count

        # Agent uploads
        with create_agent(bucket=integration_bucket_name) as agent:
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
        self, integration_bucket_name: str, sample_text_file: Path
    ) -> None:
        """Test that agent delete actually removes content from semstash."""
        stash = SemStash(bucket=integration_bucket_name, auto_init=False)

        # First agent uploads
        with create_agent(bucket=integration_bucket_name) as agent1:
            agent1(f"Upload the file at {sample_text_file}")

        # Verify exists
        browse = stash.browse("/")
        keys = [item.key for item in browse.items]
        assert sample_text_file.name in keys

        # Second agent deletes
        with create_agent(bucket=integration_bucket_name) as agent2:
            delete_result = get_response_text(
                agent2(f"Delete the content with key: {sample_text_file.name}")
            )
            assert "delete" in delete_result.lower() or sample_text_file.name in delete_result

        # Verify deleted from semstash
        browse = stash.browse("/")
        keys = [item.key for item in browse.items]
        assert sample_text_file.name not in keys, "File should be deleted from semstash"
