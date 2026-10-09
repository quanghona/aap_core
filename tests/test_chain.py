"""Tests for aap_core.chain module."""

from typing import List, Tuple

from aap_core.chain import BaseCausalMultiTurnsChain, TypicalLLMChain
from aap_core.guardrail import PassGuardRail
from aap_core.prompt_augmenter import IdentityPromptAugmenter
from aap_core.types import AgentMessage, ChainMessage, ChainResponse, TokenUsage


class MockCausalChain(BaseCausalMultiTurnsChain):
    """Mock implementation of BaseCausalMultiTurnsChain for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._call_count = 0

    def _prepare_conversation(self, message: AgentMessage) -> List[ChainMessage]:
        return []

    def _generate_response(
        self, conversation: List[ChainMessage], **kwargs
    ) -> Tuple[List[ChainMessage], ChainResponse, bool, TokenUsage]:
        self._call_count += 1
        # Return no tool on first call to break the loop
        return (
            conversation,
            "response",
            False,
            TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    def _process_tools(
        self,
        conversation: List[ChainMessage],
        response: ChainResponse,
    ) -> List[ChainMessage]:
        return conversation

    def _append_responses(
        self, message: AgentMessage, conversation: List[ChainMessage]
    ) -> AgentMessage:
        message.responses.append(("chain", "final response"))
        return message


class TestBaseCausalMultiTurnsChain:
    """Tests for BaseCausalMultiTurnsChain."""

    def test_construction_with_defaults(self):
        """Test chain construction with default values."""
        chain = MockCausalChain()
        assert chain.include_history == 0
        assert chain.store_immediate_steps is False
        assert chain.max_turns == 50

    def test_construction_with_custom_values(self):
        """Test chain construction with custom values."""
        chain = MockCausalChain(include_history=3, max_turns=10)
        assert chain.include_history == 3
        assert chain.max_turns == 10

    def test_invoke_basic(self):
        """Test basic invoke returns message with responses."""
        chain = MockCausalChain()
        message = AgentMessage(query="test query")
        result = chain.invoke(message)
        assert result.execution_result == "success"
        assert len(result.responses) == 1
        assert result.responses[0] == ("chain", "final response")

    def test_invoke_initializes_token_usage(self):
        """Test that invoke initializes token_usage if None."""
        chain = MockCausalChain()
        message = AgentMessage(query="test query")
        assert message.token_usage is None
        result = chain.invoke(message)
        assert result.token_usage is not None
        assert "steps" in result.token_usage
        assert "total" in result.token_usage

    def test_invoke_preserves_existing_token_usage(self):
        """Test that invoke preserves existing token_usage structure."""
        chain = MockCausalChain()
        message = AgentMessage(
            query="test query",
            token_usage={
                "steps": [],
                "total": TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            },
        )
        result = chain.invoke(message)
        assert len(result.token_usage["steps"]) == 1

    def test_invoke_sets_origin(self):
        """Test that invoke sets the origin to chain name."""
        chain = MockCausalChain(name="test_chain")
        message = AgentMessage(query="test query")
        result = chain.invoke(message)
        assert result.origin == "test_chain"

    def test_final_response_as_context(self):
        """Test setting final response as context."""
        chain = MockCausalChain()
        chain.final_response_as_context("context_result")
        assert chain._last_response_as_context == "result"


class MockTypicalChain(TypicalLLMChain):
    """Mock implementation of TypicalLLMChain for testing."""

    def generate(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message.responses.append(("chain", "generated response"))
        message.execution_result = "success"
        return message


class TestTypicalLLMChain:
    """Tests for TypicalLLMChain."""

    def test_construction_with_defaults(self):
        """Test chain construction with default guardrails and augmenter."""
        chain = MockTypicalChain(name="test_chain")
        assert isinstance(chain.input_guardrail, PassGuardRail)
        assert isinstance(chain.output_guardrail, PassGuardRail)
        assert isinstance(chain.prompt_augmenter, IdentityPromptAugmenter)
        assert chain.tools == []

    def test_invoke_runs_pipeline(self):
        """Test that invoke runs the full pipeline: guardrail -> augmenter -> generate -> guardrail."""
        chain = MockTypicalChain(name="test_chain")
        message = AgentMessage(query="test query")
        result = chain.invoke(message)
        assert result.execution_result == "success"
        assert len(result.responses) == 1

    def test_invoke_with_custom_guardrails(self):
        """Test invoke with custom input/output guardrails."""
        custom_guardrail = PassGuardRail()
        chain = MockTypicalChain(
            name="test_chain",
            input_guardrail=custom_guardrail,
            output_guardrail=custom_guardrail,
        )
        message = AgentMessage(query="test query")
        result = chain.invoke(message)
        assert result.execution_result == "success"

    def test_invoke_with_tools(self):
        """Test invoke with tools defined."""

        def dummy_tool(msg: AgentMessage) -> AgentMessage:
            return msg

        chain = MockTypicalChain(name="test_chain", tools=[dummy_tool])
        message = AgentMessage(query="test query")
        result = chain.invoke(message)
        assert result.execution_result == "success"

    def test_invoke_preserves_message_query(self):
        """Test that the original query is preserved through the pipeline."""
        chain = MockTypicalChain(name="test_chain")
        message = AgentMessage(query="original query")
        result = chain.invoke(message)
        assert result.query == "original query"
