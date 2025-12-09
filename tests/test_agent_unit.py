"""Unit tests for the SemStash Agent module.

These tests use mocking to verify agent behavior without requiring AWS.
For integration tests with real AWS, see test_agent.py.

Run with: pytest tests/test_agent_unit.py
"""

from unittest.mock import MagicMock, patch

import pytest

from semstash.agent import (
    DEFAULT_MODEL_ID,
    SYSTEM_PROMPT,
    SemStashAgent,
    _get_community_tools,
    _get_mcp_server_params,
)


class TestMcpServerParams:
    """Tests for _get_mcp_server_params function."""

    def test_local_params(self) -> None:
        """Test that local params use sys.executable."""
        params = _get_mcp_server_params("test-bucket", use_local=True)
        assert params.command.endswith("python") or "python" in params.command
        assert "-m" in params.args
        assert "semstash.mcp_server" in params.args
        assert "test-bucket" in params.args

    def test_uvx_params(self) -> None:
        """Test that uvx params use uvx command."""
        params = _get_mcp_server_params("test-bucket", use_local=False)
        assert params.command == "uvx"
        assert "semstash" in params.args
        assert "mcp" in params.args
        assert "test-bucket" in params.args


class TestCommunityTools:
    """Tests for _get_community_tools function."""

    def test_returns_list(self) -> None:
        """Test that community tools returns a list."""
        tools = _get_community_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_no_write_tools(self) -> None:
        """Test that community tools doesn't include write tools."""
        tools = _get_community_tools()
        tool_names = [getattr(t, "__name__", str(t)) for t in tools]
        # Verify no file_write or editor tools (agent focuses on reading)
        assert "file_write" not in tool_names
        assert "editor" not in tool_names


class TestDefaultConstants:
    """Tests for module constants."""

    def test_default_model_id(self) -> None:
        """Test that default model ID is set."""
        assert DEFAULT_MODEL_ID == "us.amazon.nova-lite-v1:0"

    def test_system_prompt_content(self) -> None:
        """Test that system prompt contains key instructions."""
        assert "SemStash" in SYSTEM_PROMPT
        assert "Search/Query" in SYSTEM_PROMPT or "Search" in SYSTEM_PROMPT
        assert "Browse" in SYSTEM_PROMPT
        assert "Delete" in SYSTEM_PROMPT


class TestSemStashAgentInit:
    """Tests for SemStashAgent initialization."""

    def test_init_stores_attributes(self) -> None:
        """Test that __init__ stores all attributes."""
        agent = SemStashAgent(
            bucket="test-bucket",
            model_id="test-model",
            streaming=False,
            use_local=False,
            system_prompt="Test prompt",
            region="us-west-2",
        )
        assert agent.bucket == "test-bucket"
        assert agent.model_id == "test-model"
        assert agent.streaming is False
        assert agent.use_local is False
        assert agent.system_prompt == "Test prompt"
        assert agent.region == "us-west-2"
        assert agent._mcp_client is None
        assert agent._agent is None

    def test_init_with_defaults(self) -> None:
        """Test that __init__ uses default values."""
        agent = SemStashAgent(bucket="test-bucket")
        assert agent.bucket == "test-bucket"
        assert agent.model_id == DEFAULT_MODEL_ID
        assert agent.streaming is True
        assert agent.use_local is True
        assert agent.system_prompt == SYSTEM_PROMPT
        assert agent.region is None


class TestSemStashAgentNotInitialized:
    """Tests for SemStashAgent when not in context manager."""

    def test_chat_raises_without_context(self) -> None:
        """Test that chat raises RuntimeError when not initialized."""
        agent = SemStashAgent(bucket="test-bucket")
        with pytest.raises(RuntimeError, match="not initialized"):
            agent.chat("Hello")

    def test_chat_stream_raises_without_context(self) -> None:
        """Test that chat_stream raises RuntimeError when not initialized."""
        agent = SemStashAgent(bucket="test-bucket")
        with pytest.raises(RuntimeError, match="not initialized"):
            list(agent.chat_stream("Hello"))

    def test_reset_raises_without_context(self) -> None:
        """Test that reset_conversation raises RuntimeError when not initialized."""
        agent = SemStashAgent(bucket="test-bucket")
        with pytest.raises(RuntimeError, match="not initialized"):
            agent.reset_conversation()


class TestSemStashAgentContextManager:
    """Tests for SemStashAgent context manager behavior."""

    @patch("semstash.agent.MCPClient")
    @patch("semstash.agent.BedrockModel")
    @patch("semstash.agent.Agent")
    def test_enter_initializes_components(
        self,
        mock_agent_class: MagicMock,
        mock_model_class: MagicMock,
        mock_mcp_class: MagicMock,
    ) -> None:
        """Test that __enter__ initializes MCP client and agent."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.list_tools_sync.return_value = [MagicMock()]
        mock_mcp_class.return_value = mock_mcp_instance

        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        agent = SemStashAgent(bucket="test-bucket", model_id="test-model")

        # Enter context
        result = agent.__enter__()

        # Verify initialization
        assert result is agent
        assert agent._mcp_client is mock_mcp_instance
        assert agent._agent is mock_agent_instance
        mock_mcp_instance.__enter__.assert_called_once()
        mock_mcp_instance.list_tools_sync.assert_called_once()
        mock_model_class.assert_called_once_with(model_id="test-model", streaming=True)
        mock_agent_class.assert_called_once()

    @patch("semstash.agent.MCPClient")
    @patch("semstash.agent.BedrockModel")
    @patch("semstash.agent.Agent")
    def test_exit_cleans_up(
        self,
        mock_agent_class: MagicMock,
        mock_model_class: MagicMock,
        mock_mcp_class: MagicMock,
    ) -> None:
        """Test that __exit__ cleans up MCP client."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.list_tools_sync.return_value = []
        mock_mcp_class.return_value = mock_mcp_instance

        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        agent = SemStashAgent(bucket="test-bucket")
        agent.__enter__()
        agent.__exit__(None, None, None)

        # Verify cleanup
        assert agent._mcp_client is None
        assert agent._agent is None

    @patch("semstash.agent.MCPClient")
    @patch("semstash.agent.BedrockModel")
    @patch("semstash.agent.Agent")
    def test_region_sets_environment(
        self,
        mock_agent_class: MagicMock,
        mock_model_class: MagicMock,
        mock_mcp_class: MagicMock,
    ) -> None:
        """Test that region parameter sets AWS_DEFAULT_REGION."""
        import os

        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.list_tools_sync.return_value = []
        mock_mcp_class.return_value = mock_mcp_instance

        mock_model_class.return_value = MagicMock()
        mock_agent_class.return_value = MagicMock()

        original_region = os.environ.get("AWS_DEFAULT_REGION")

        try:
            agent = SemStashAgent(bucket="test-bucket", region="eu-west-1")
            agent.__enter__()
            assert os.environ.get("AWS_DEFAULT_REGION") == "eu-west-1"
            agent.__exit__(None, None, None)
        finally:
            # Restore original value
            if original_region is None:
                os.environ.pop("AWS_DEFAULT_REGION", None)
            else:
                os.environ["AWS_DEFAULT_REGION"] = original_region


class TestSemStashAgentChat:
    """Tests for SemStashAgent chat methods."""

    @patch("semstash.agent.MCPClient")
    @patch("semstash.agent.BedrockModel")
    @patch("semstash.agent.Agent")
    def test_chat_returns_text(
        self,
        mock_agent_class: MagicMock,
        mock_model_class: MagicMock,
        mock_mcp_class: MagicMock,
    ) -> None:
        """Test that chat returns text from agent response."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.list_tools_sync.return_value = []
        mock_mcp_class.return_value = mock_mcp_instance

        mock_model_class.return_value = MagicMock()

        mock_response = MagicMock()
        mock_response.message = {"content": [{"text": "Hello, world!"}]}

        mock_agent_instance = MagicMock()
        mock_agent_instance.return_value = mock_response
        mock_agent_class.return_value = mock_agent_instance

        with SemStashAgent(bucket="test-bucket") as agent:
            result = agent.chat("Hi")
            assert result == "Hello, world!"

    @patch("semstash.agent.MCPClient")
    @patch("semstash.agent.BedrockModel")
    @patch("semstash.agent.Agent")
    def test_chat_handles_multiple_text_blocks(
        self,
        mock_agent_class: MagicMock,
        mock_model_class: MagicMock,
        mock_mcp_class: MagicMock,
    ) -> None:
        """Test that chat joins multiple text blocks."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.list_tools_sync.return_value = []
        mock_mcp_class.return_value = mock_mcp_instance

        mock_model_class.return_value = MagicMock()

        mock_response = MagicMock()
        mock_response.message = {"content": [{"text": "Part 1"}, {"text": "Part 2"}]}

        mock_agent_instance = MagicMock()
        mock_agent_instance.return_value = mock_response
        mock_agent_class.return_value = mock_agent_instance

        with SemStashAgent(bucket="test-bucket") as agent:
            result = agent.chat("Hi")
            assert "Part 1" in result
            assert "Part 2" in result

    @patch("semstash.agent.MCPClient")
    @patch("semstash.agent.BedrockModel")
    @patch("semstash.agent.Agent")
    def test_chat_stream_yields_events(
        self,
        mock_agent_class: MagicMock,
        mock_model_class: MagicMock,
        mock_mcp_class: MagicMock,
    ) -> None:
        """Test that chat_stream yields events from callback handler."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.list_tools_sync.return_value = []
        mock_mcp_class.return_value = mock_mcp_instance

        mock_model_class.return_value = MagicMock()

        mock_agent_instance = MagicMock()
        mock_agent_instance.callback_handler = None

        # Simulate callback invocation during agent call
        def simulate_agent_call(message: str) -> MagicMock:
            # The callback handler will be set by chat_stream
            handler = mock_agent_instance.callback_handler
            if handler:
                handler(data="event1")
                handler(data="event2")
            return MagicMock()

        mock_agent_instance.side_effect = simulate_agent_call
        mock_agent_class.return_value = mock_agent_instance

        with SemStashAgent(bucket="test-bucket") as agent:
            events = list(agent.chat_stream("Hi"))
            assert len(events) == 2
            assert events[0]["type"] == "data"
            assert events[0]["data"] == "event1"
            assert events[1]["data"] == "event2"


class TestSemStashAgentResetConversation:
    """Tests for SemStashAgent reset_conversation method."""

    @patch("semstash.agent.MCPClient")
    @patch("semstash.agent.BedrockModel")
    @patch("semstash.agent.Agent")
    def test_reset_clears_messages(
        self,
        mock_agent_class: MagicMock,
        mock_model_class: MagicMock,
        mock_mcp_class: MagicMock,
    ) -> None:
        """Test that reset_conversation clears agent messages."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.list_tools_sync.return_value = []
        mock_mcp_class.return_value = mock_mcp_instance

        mock_model_class.return_value = MagicMock()

        mock_messages = ["msg1", "msg2"]  # Mutable list
        mock_agent_instance = MagicMock()
        mock_agent_instance.messages = mock_messages
        mock_agent_class.return_value = mock_agent_instance

        with SemStashAgent(bucket="test-bucket") as agent:
            agent.reset_conversation()
            assert len(mock_messages) == 0


class TestSemstashAgentContextFunction:
    """Tests for semstash_agent context manager function."""

    @patch("semstash.agent.MCPClient")
    @patch("semstash.agent.BedrockModel")
    @patch("semstash.agent.Agent")
    def test_context_function_yields_agent(
        self,
        mock_agent_class: MagicMock,
        mock_model_class: MagicMock,
        mock_mcp_class: MagicMock,
    ) -> None:
        """Test that semstash_agent context manager yields SemStashAgent."""
        from semstash.agent import semstash_agent

        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.list_tools_sync.return_value = []
        mock_mcp_class.return_value = mock_mcp_instance

        mock_model_class.return_value = MagicMock()
        mock_agent_class.return_value = MagicMock()

        with semstash_agent("test-bucket", model_id="test-model") as agent:
            assert isinstance(agent, SemStashAgent)
            assert agent.bucket == "test-bucket"
            assert agent.model_id == "test-model"
