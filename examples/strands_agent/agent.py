"""SemStash Agent - AI agent for semantic storage using Strands SDK."""

import os
from collections.abc import Generator
from contextlib import contextmanager

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.tools.mcp import MCPClient


@contextmanager
def semstash_agent(
    bucket: str, region: str = "us-east-1", dimension: int = 3072
) -> Generator[Agent]:
    """Create an AI agent for semantic storage via semstash MCP server."""
    env = {
        **os.environ,
        "SEMSTASH_BUCKET": bucket,
        "SEMSTASH_REGION": region,
        "SEMSTASH_DIMENSION": str(dimension),
    }
    mcp_client = MCPClient(
        lambda: stdio_client(
            StdioServerParameters(command="uvx", args=["semstash", "mcp"], env=env)
        )
    )

    with mcp_client:
        tools = mcp_client.list_tools_sync()
        yield Agent(tools=tools, system_prompt="You help users store and retrieve information.")
