"""SemStash Agent - AI agent for semantic storage using Strands SDK."""

import sys
from collections.abc import Generator
from contextlib import contextmanager

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.tools.mcp import MCPClient


@contextmanager
def semstash_agent(bucket: str, use_local: bool = True) -> Generator[Agent]:
    """Create an AI agent for semantic storage via semstash MCP server.

    Args:
        bucket: The stash/bucket name to use.
        use_local: If True, use local Python module. If False, use uvx.
    """
    if use_local:
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "semstash.mcp_server", bucket]
        )
    else:
        params = StdioServerParameters(command="uvx", args=["semstash", "mcp", bucket])

    mcp_client = MCPClient(lambda: stdio_client(params))

    with mcp_client:
        tools = mcp_client.list_tools_sync()
        yield Agent(tools=tools, system_prompt="You help users store and retrieve information.")
