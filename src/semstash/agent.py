"""SemStash Agent - AI agent for semantic storage using Strands SDK.

Provides conversational access to semantic storage via:
- Python API (SemStashAgent context manager)
- CLI (semstash agent <stash>)
- Web interface

The agent focuses on interacting with existing content in the stash:
- Search/query for content
- Get summaries of documents
- Browse stored content
- Delete content when needed

All stash operations go through the SemStash MCP server.
"""

import contextlib
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

# Import community tools (type: ignore for missing stubs)
from strands_tools import (  # type: ignore[import-untyped]
    calculator,
    current_time,
    file_read,
    graph,
    http_request,
    sleep,
    swarm,
    use_agent,
)

# Default model: Amazon Nova 2 Lite via global inference profile
DEFAULT_MODEL_ID = "global.amazon.nova-2-lite-v1:0"

# System prompt defining agent capabilities
SYSTEM_PROMPT = """You are a helpful AI assistant with access to semantic storage (SemStash).

Your primary purpose is to help users interact with content already stored in their stash:
- **Search/Query**: Find relevant content using natural language queries
- **Browse**: Explore what's stored in the stash at different paths
- **Get/Retrieve**: Access specific content and provide summaries
- **Delete**: Remove content when requested
- **Stats**: Show storage statistics

You also have access to additional tools:
- Calculator for math operations
- Current time information
- File reading (for local files the user wants to reference)
- HTTP requests for web data
- Multi-agent tools (use_agent, swarm, graph) for complex tasks

Important guidelines:
1. All stash operations must use the SemStash MCP tools (query, browse, get, delete, stats, init)
2. When asked to summarize or explain content, first use 'get' with return_content=True
3. Use 'query' for semantic search - describe what you're looking for naturally
4. Use 'browse' to explore the stash structure before searching
5. Be helpful in explaining what content is available and how it relates to user questions

The stash is organized like a filesystem with paths (e.g., '/docs/', '/images/')."""


def _get_mcp_server_params(bucket: str, use_local: bool = True) -> StdioServerParameters:
    """Create MCP server parameters for semstash.

    Args:
        bucket: The stash/bucket name to use.
        use_local: If True, use local Python module. If False, use uvx.

    Returns:
        StdioServerParameters for connecting to the MCP server.
    """
    if use_local:
        return StdioServerParameters(
            command=sys.executable, args=["-m", "semstash.mcp_server", bucket]
        )
    return StdioServerParameters(command="uvx", args=["semstash", "mcp", bucket])


def _get_community_tools() -> list[Any]:
    """Get the list of community tools for the agent.

    Returns:
        List of community tools (read-only operations + utilities).
    """
    return [
        calculator,
        current_time,
        sleep,
        file_read,
        http_request,
        use_agent,
        swarm,
        graph,
    ]


class SemStashAgent:
    """AI agent for semantic storage via SemStash MCP server.

    This agent provides conversational access to a SemStash instance,
    focusing on content retrieval, search, and exploration.

    Usage:
        with SemStashAgent(bucket="my-stash") as agent:
            # Non-streaming chat
            result = agent.chat("What files are stored?")
            print(result)

            # Streaming chat with callback
            def on_token(text: str) -> None:
                print(text, end='', flush=True)
            agent.chat_stream("Summarize the readme", callback=on_token)

    Attributes:
        bucket: The stash/bucket name.
        model_id: The Bedrock model ID to use.
        streaming: Whether to enable streaming by default.
    """

    def __init__(
        self,
        bucket: str,
        model_id: str = DEFAULT_MODEL_ID,
        streaming: bool = True,
        use_local: bool = True,
        system_prompt: str | None = None,
        region: str | None = None,
    ) -> None:
        """Initialize the SemStash agent.

        Args:
            bucket: The stash/bucket name to use.
            model_id: Bedrock model ID (default: Nova 2 Lite global).
            streaming: Enable streaming responses.
            use_local: Use local Python module (True) or uvx (False).
            system_prompt: Custom system prompt (optional).
            region: AWS region for Bedrock (optional, uses default if not set).
        """
        self.bucket = bucket
        self.model_id = model_id
        self.streaming = streaming
        self.use_local = use_local
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.region = region

        self._mcp_client: MCPClient | None = None
        self._agent: Agent | None = None
        self._stream_events: list[Any] = []

    def __enter__(self) -> "SemStashAgent":
        """Enter context manager and initialize the agent."""
        # Set region if provided
        if self.region:
            os.environ["AWS_DEFAULT_REGION"] = self.region

        # Create MCP client
        params = _get_mcp_server_params(self.bucket, self.use_local)
        self._mcp_client = MCPClient(lambda: stdio_client(params))
        self._mcp_client.__enter__()

        # Get MCP tools
        mcp_tools = self._mcp_client.list_tools_sync()

        # Combine MCP tools with community tools
        all_tools = list(mcp_tools) + _get_community_tools()

        # Create the model
        model = BedrockModel(
            model_id=self.model_id,
            streaming=self.streaming,
        )

        # Create the agent with default conversation manager (sliding window)
        # Disable default callback handler so we can capture events
        self._agent = Agent(
            model=model,
            tools=all_tools,
            system_prompt=self.system_prompt,
            callback_handler=None,
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager and cleanup."""
        if self._mcp_client:
            # MCPClient expects BaseException, not optional types
            # Use contextlib.suppress to handle any cleanup errors gracefully
            with contextlib.suppress(Exception):
                self._mcp_client.__exit__(
                    exc_type or BaseException,  # type: ignore[arg-type]
                    exc_val or BaseException(),
                    exc_tb,
                )
            self._mcp_client = None
        self._agent = None

    def chat(self, message: str) -> str:
        """Send a message and get a response.

        Args:
            message: The user message.

        Returns:
            The agent's response text.

        Raises:
            RuntimeError: If agent is not initialized (not in context manager).
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized. Use within 'with' context.")

        result = self._agent(message)

        # Extract text from response
        if hasattr(result, "message") and result.message:
            content = result.message.get("content", [])
            return " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and "text" in block
            )
        return str(result)

    def chat_stream(self, message: str, callback: Any | None = None) -> Generator[dict[str, Any]]:
        """Send a message and stream the response.

        This method uses a custom callback handler to capture streaming events.
        Each event is yielded as a dict with 'type' and optional 'data' keys.

        Args:
            message: The user message.
            callback: Optional callback function(text: str) called for each text chunk.

        Yields:
            Stream events from the agent as dicts with 'type' and 'data' keys.

        Raises:
            RuntimeError: If agent is not initialized (not in context manager).
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized. Use within 'with' context.")

        # Collect events during streaming
        self._stream_events = []

        def stream_callback(**kwargs: Any) -> None:
            """Callback to capture stream events."""
            event: dict[str, Any] = {"type": "unknown"}
            if "data" in kwargs:
                event = {"type": "data", "data": kwargs["data"]}
                if callback:
                    callback(kwargs["data"])
            elif "delta" in kwargs:
                event = {"type": "delta", "data": kwargs["delta"]}
            else:
                event = {"type": "event", "data": kwargs}
            self._stream_events.append(event)

        # Temporarily set callback handler for streaming
        original_handler = self._agent.callback_handler
        self._agent.callback_handler = stream_callback

        try:
            # Invoke the agent (this triggers streaming via callback)
            self._agent(message)

            # Yield collected events
            yield from self._stream_events

        finally:
            # Restore original handler
            self._agent.callback_handler = original_handler
            self._stream_events = []

    def reset_conversation(self) -> None:
        """Clear the conversation history.

        Raises:
            RuntimeError: If agent is not initialized (not in context manager).
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized. Use within 'with' context.")

        # Access the messages list directly and clear it
        if hasattr(self._agent, "messages"):
            self._agent.messages.clear()


@contextmanager
def semstash_agent(
    bucket: str,
    model_id: str = DEFAULT_MODEL_ID,
    streaming: bool = True,
    use_local: bool = True,
    system_prompt: str | None = None,
    region: str | None = None,
) -> Generator[SemStashAgent]:
    """Context manager to create a SemStash agent.

    This is a convenience function that wraps SemStashAgent.

    Args:
        bucket: The stash/bucket name to use.
        model_id: Bedrock model ID (default: Nova 2 Lite global).
        streaming: Enable streaming responses.
        use_local: Use local Python module (True) or uvx (False).
        system_prompt: Custom system prompt (optional).
        region: AWS region for Bedrock (optional).

    Yields:
        SemStashAgent instance ready for chat.

    Example:
        with semstash_agent("my-stash") as agent:
            response = agent.chat("What's in my stash?")
            print(response)
    """
    with SemStashAgent(
        bucket=bucket,
        model_id=model_id,
        streaming=streaming,
        use_local=use_local,
        system_prompt=system_prompt,
        region=region,
    ) as agent:
        yield agent
