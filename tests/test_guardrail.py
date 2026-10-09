"""Tests for aap_core.guardrail module."""

from aap_core.guardrail import BaseGuardRail, PassGuardRail
from aap_core.types import AgentMessage


class TestPassGuardRail:
    """Tests for the PassGuardRail class."""

    def test_pass_guardrail_returns_message_unchanged(self):
        """Test that PassGuardRail returns the message unchanged."""
        guardrail = PassGuardRail()
        message = AgentMessage(query="test query")
        result = guardrail(message)
        assert result is message
        assert result.query == "test query"

    def test_pass_guardrail_with_kwargs(self):
        """Test PassGuardRail with extra kwargs."""
        guardrail = PassGuardRail()
        message = AgentMessage(query="test query")
        result = guardrail(message, some_kwarg="value")
        assert result is message

    def test_pass_guardrail_with_responses(self):
        """Test PassGuardRail preserves responses."""
        guardrail = PassGuardRail()
        message = AgentMessage(
            query="test",
            responses=[("agent1", "response1")],
            execution_result="success",
        )
        result = guardrail(message)
        assert result.responses == [("agent1", "response1")]
        assert result.execution_result == "success"

    def test_pass_guardrail_is_base_guardrail(self):
        """Test that PassGuardRail is an instance of BaseGuardRail."""
        guardrail = PassGuardRail()
        assert isinstance(guardrail, BaseGuardRail)
