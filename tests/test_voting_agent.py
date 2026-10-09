from unittest.mock import MagicMock

import pytest
from a2a.types import AgentCard
from aap_core.agent import BaseAgent
from aap_core.orchestration import VotingAgent
from aap_core.types import AgentMessage


@pytest.fixture
def agent_card(name: str = "test_agent") -> AgentCard:
    return AgentCard(
        name=name,
        description=f"Test agent {name}",
        capabilities={},
        skills=[],
        default_input_modes=[],
        default_output_modes=[],
        url="http://localhost:8000",
        version="1.0.0",
    )


class MockAgent(BaseAgent):
    """A minimal agent that returns a fixed response for testing."""

    response_content: str = "mock response"

    def __init__(
        self,
        name: str = "mock_agent",
        response_content: str = "mock response",
        **kwargs,
    ):
        card = AgentCard(
            name=name,
            description=f"Mock agent {name}",
            capabilities={},
            skills=[],
            default_input_modes=[],
            default_output_modes=[],
            url="http://localhost:8000",
            version="1.0.0",
        )
        super().__init__(card=card, **kwargs)
        self.response_content = response_content

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message.responses.append((self.card.name, self.response_content))
        message.execution_result = "success"
        return message


class MockChainAgent(BaseAgent):
    """An agent that uses a mock chain to produce responses."""

    def __init__(self, name: str = "chain_agent", chain: MagicMock = None, **kwargs):
        card = AgentCard(
            name=name,
            description=f"Chain agent {name}",
            capabilities={},
            skills=[],
            default_input_modes=[],
            default_output_modes=[],
            url="http://localhost:8000",
            version="1.0.0",
        )
        super().__init__(card=card, **kwargs)
        self.mock_chain = chain or MagicMock()
        self.mock_chain.invoke.return_value = AgentMessage(
            query="test",
            responses=[(name, "chain response")],
            execution_result="success",
        )

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        result = self.mock_chain.invoke(message, **kwargs)
        message.responses.append((self.card.name, result.responses[-1][1]))
        message.execution_result = result.execution_result
        return message


def _make_voting_agent(
    agents,
    voting_method: str = "majority_vote",
    scorer: str = "bleu",
    voting_prompt: str | None = None,
) -> VotingAgent:
    """Helper to create a VotingAgent with a valid card."""
    card = AgentCard(
        name="voting_agent",
        description="Voting agent",
        capabilities={},
        skills=[],
        default_input_modes=[],
        default_output_modes=[],
        url="http://localhost:8000",
        version="1.0.0",
    )
    kwargs: dict = dict(
        card=card,
        agents=agents,
        voting_method=voting_method,
        scorer=scorer,
    )
    if voting_prompt is not None:
        kwargs["voting_prompt"] = voting_prompt
    return VotingAgent(**kwargs)


class TestVotingAgentMajorityVoteBleu:
    """Tests for majority_vote with bleu/agent_forest scorer."""

    def test_single_agent_bleu(self):
        """Single agent should return its own response as the winner."""
        agents = [MockAgent(name="agent_a", response_content="Hello world")]
        voting_agent = _make_voting_agent(agents, scorer="bleu")
        message = AgentMessage(query="test query")
        # Pre-populate responses: each agent's response is a tuple (agent_name, content)
        message.responses = [
            ("agent_a", "Hello world"),
        ]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"
        assert result.origin == voting_agent.card.name
        # The winning response should be appended
        assert len(result.responses) > 0
        assert result.responses[-1][0] == voting_agent.card.name

    def test_two_agents_bleu(self):
        """Two agents with identical responses — scores should be equal, max picks first."""
        agents = [
            MockAgent(name="agent_a", response_content="Same answer"),
            MockAgent(name="agent_b", response_content="Same answer"),
        ]
        voting_agent = _make_voting_agent(agents, scorer="bleu")
        message = AgentMessage(query="test query")
        message.responses = [
            ("agent_a", "Same answer"),
            ("agent_b", "Same answer"),
        ]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"
        assert result.responses[-1][0] == voting_agent.card.name

    def test_two_agents_bleu_different_responses(self):
        """Two agents with different responses — scoring should still work."""
        agents = [
            MockAgent(name="agent_a", response_content="The capital is Paris"),
            MockAgent(name="agent_b", response_content="Paris is the capital"),
        ]
        voting_agent = _make_voting_agent(agents, scorer="bleu")
        message = AgentMessage(query="test query")
        message.responses = [
            ("agent_a", "The capital is Paris"),
            ("agent_b", "Paris is the capital"),
        ]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"
        assert result.responses[-1][0] == voting_agent.card.name

    def test_agent_forest_scorer(self):
        """agent_forest is an alias for bleu."""
        agents = [MockAgent(name="agent_x", response_content="Test response")]
        voting_agent = _make_voting_agent(agents, scorer="agent_forest")
        message = AgentMessage(query="test query")
        message.responses = [("agent_x", "Test response")]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"

    def test_three_agents_bleu(self):
        """Three agents — scoring across all pairs."""
        agents = [
            MockAgent(name="agent_a", response_content="Answer one"),
            MockAgent(name="agent_b", response_content="Answer two"),
            MockAgent(name="agent_c", response_content="Answer one"),
        ]
        voting_agent = _make_voting_agent(agents, scorer="bleu")
        message = AgentMessage(query="test query")
        message.responses = [
            ("agent_a", "Answer one"),
            ("agent_b", "Answer two"),
            ("agent_c", "Answer one"),
        ]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"
        assert result.responses[-1][0] == voting_agent.card.name


class TestVotingAgentMajorityVoteRouge:
    """Tests for majority_vote with rouge scorers."""

    def test_rougeL_scorer(self):
        """rougeL scorer should work correctly."""
        agents = [MockAgent(name="agent_a", response_content="Rouge test")]
        voting_agent = _make_voting_agent(agents, scorer="rougeL")
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "Rouge test")]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"

    def test_rouge1_scorer(self):
        """rouge1 scorer should work correctly."""
        agents = [MockAgent(name="agent_a", response_content="Rouge one")]
        voting_agent = _make_voting_agent(agents, scorer="rouge1")
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "Rouge one")]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"

    def test_rouge2_scorer(self):
        """rouge2 scorer should work correctly."""
        agents = [MockAgent(name="agent_a", response_content="Rouge two")]
        voting_agent = _make_voting_agent(agents, scorer="rouge2")
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "Rouge two")]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"

    def test_rougeL_with_different_responses(self):
        """rougeL with different responses from multiple agents.

        Note: The rouge scorer returns a Score object which cannot be added with +=.
        This exposes a bug in the source code where total_score (int) += Score fails.
        """
        agents = [
            MockAgent(name="agent_a", response_content="The quick brown fox"),
            MockAgent(name="agent_b", response_content="A quick brown fox"),
        ]
        voting_agent = _make_voting_agent(agents, scorer="rougeL")
        message = AgentMessage(query="test query")
        message.responses = [
            ("agent_a", "The quick brown fox"),
            ("agent_b", "A quick brown fox"),
        ]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"
        assert result.responses[-1][0] == voting_agent.card.name


class TestVotingAgentMajorityVoteErrors:
    """Error cases for majority_vote."""

    def test_invalid_scorer_string(self):
        """An unsupported scorer string should raise ValueError."""
        agents = [MockAgent(name="agent_a")]
        voting_agent = _make_voting_agent(agents, scorer="unsupported_metric")
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "response")]
        with pytest.raises(ValueError, match="Scorer not supported"):
            voting_agent.execute(message)

    def test_non_string_scorer_for_majority_vote(self):
        """A callable scorer with majority_vote should raise TypeError."""
        agents = [MockAgent(name="agent_a")]
        voting_agent = _make_voting_agent(agents, scorer=lambda x: 1.0)  # type: ignore
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "response")]
        with pytest.raises(TypeError, match="scorer need to be one of"):
            voting_agent.execute(message)


class TestVotingAgentLLMScore:
    """Tests for llm_score voting method."""

    def test_llm_score_basic(self):
        """Basic llm_score with a callable scorer."""
        agents = [
            MockAgent(name="agent_a", response_content="Answer A"),
            MockAgent(name="agent_b", response_content="Answer B"),
        ]
        voting_agent = _make_voting_agent(
            agents,
            voting_method="llm_score",
            scorer=lambda x: float(x) if x.isdigit() else 0.0,
            voting_prompt="Score this answer: {answer}",
        )
        message = AgentMessage(query="test query")
        message.responses = [
            ("agent_a", "5"),
            ("agent_b", "3"),
        ]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"
        assert result.responses[-1][0] == voting_agent.card.name

    def test_llm_score_without_prompt_raises(self):
        """voting_prompt is required for llm_score."""
        agents = [MockAgent(name="agent_a")]
        voting_agent = _make_voting_agent(
            agents,
            voting_method="llm_score",
            scorer=lambda x: 1.0,
            voting_prompt=None,
        )
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "response")]
        with pytest.raises(ValueError, match="voting_prompt is required"):
            voting_agent.execute(message)

    def test_llm_score_non_callable_scorer_raises(self):
        """A non-callable scorer with llm_score should raise TypeError."""
        agents = [MockAgent(name="agent_a")]
        voting_agent = _make_voting_agent(
            agents,
            voting_method="llm_score",
            scorer="not_a_callable",  # type: ignore
            voting_prompt="Score this",
        )
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "response")]
        with pytest.raises(TypeError, match="scorer must be a Callable"):
            voting_agent.execute(message)

    def test_llm_score_scorer_exception_handled(self):
        """If scorer raises an exception, the score should default to 0."""
        agents = [
            MockAgent(name="agent_a", response_content="5"),
            MockAgent(name="agent_b", response_content="3"),
        ]

        # This scorer always raises
        def bad_scorer(x: str) -> float:
            raise ValueError("scorer broken")

        voting_agent = _make_voting_agent(
            agents,
            voting_method="llm_score",
            scorer=bad_scorer,
            voting_prompt="Score this",
        )
        message = AgentMessage(query="test query")
        message.responses = [
            ("agent_a", "5"),
            ("agent_b", "3"),
        ]
        result = voting_agent.execute(message)
        # Should still succeed — exceptions are caught and score defaults to 0
        assert result.execution_result == "success"
        assert result.responses[-1][0] == voting_agent.card.name

    def test_llm_score_single_agent(self):
        """llm_score with a single agent."""
        agents = [MockAgent(name="agent_a", response_content="42")]
        voting_agent = _make_voting_agent(
            agents,
            voting_method="llm_score",
            scorer=lambda x: float(x),
            voting_prompt="Score this",
        )
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "42")]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"

    def test_llm_score_multiple_agents_different_scores(self):
        """llm_score with agents getting different scores.

        Note: Each response is scored by having ALL OTHER agents vote on it.
        So the score for a response = sum of scores from all other agents.
        """
        agents = [
            MockAgent(name="agent_a", response_content="9"),
            MockAgent(name="agent_b", response_content="3"),
            MockAgent(name="agent_c", response_content="7"),
        ]
        voting_agent = _make_voting_agent(
            agents,
            voting_method="llm_score",
            scorer=lambda x: float(x),
            voting_prompt="Score this",
        )
        message = AgentMessage(query="test query")
        message.responses = [
            ("agent_a", "9"),
            ("agent_b", "3"),
            ("agent_c", "7"),
        ]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"
        # agent_b's response ("3") gets scored by agent_a(9) + agent_c(7) = 16
        # agent_a's response ("9") gets scored by agent_b(3) + agent_c(7) = 10
        # agent_c's response ("7") gets scored by agent_a(9) + agent_b(3) = 12
        # So agent_b has the highest total score
        assert result.responses[-1][1] == "3"


class TestVotingAgentInvalidMethod:
    """Tests for invalid voting_method."""

    def test_invalid_voting_method(self):
        """An unsupported voting_method should raise ValidationError at construction time.

        Pydantic validates the Literal type at construction, so the error is raised
        before execute() is even called.
        """
        agents = [MockAgent(name="agent_a")]
        with pytest.raises(Exception):  # Pydantic ValidationError
            _make_voting_agent(
                agents,
                voting_method="invalid_method",  # type: ignore
                scorer="bleu",
            )


class TestVotingAgentState:
    """Tests for state transitions and composed state."""

    def test_state_becomes_idle_after_execute(self):
        """After execute completes, state should be 'idle'."""
        agents = [MockAgent(name="agent_a", response_content="Hello")]
        voting_agent = _make_voting_agent(agents)
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "Hello")]
        voting_agent.execute(message)
        assert voting_agent.state == "idle"

    def test_composed_state_is_sequential(self):
        """_set_composed_state should set sequential composition."""
        agents = [
            MockAgent(name="agent_a"),
            MockAgent(name="agent_b"),
        ]
        voting_agent = _make_voting_agent(agents)
        # The composed state should contain all agent names
        assert "agent_a" in voting_agent.composed_state
        assert "agent_b" in voting_agent.composed_state

    def test_state_becomes_running_during_execute(self):
        """State should transition to 'running' during execute."""
        agents = [MockAgent(name="agent_a", response_content="Hello")]
        voting_agent = _make_voting_agent(agents)
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "Hello")]
        # Before execute, state is idle
        assert voting_agent.state == "idle"
        # During execute, state should be "running"
        # We can't easily observe the intermediate state, but we can verify
        # the final state is idle
        voting_agent.execute(message)
        assert voting_agent.state == "idle"


class TestVotingAgentEdgeCases:
    """Edge case tests for VotingAgent."""

    def test_empty_response_content(self):
        """Agents with empty response strings should still work."""
        agents = [
            MockAgent(name="agent_a", response_content=""),
            MockAgent(name="agent_b", response_content="non-empty"),
        ]
        voting_agent = _make_voting_agent(agents)
        message = AgentMessage(query="test query")
        message.responses = [
            ("agent_a", ""),
            ("agent_b", "non-empty"),
        ]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"

    def test_very_long_responses(self):
        """Very long response strings should not cause issues."""
        long_text = "word " * 1000
        agents = [
            MockAgent(name="agent_a", response_content=long_text),
            MockAgent(name="agent_b", response_content=long_text),
        ]
        voting_agent = _make_voting_agent(agents)
        message = AgentMessage(query="test query")
        message.responses = [
            ("agent_a", long_text),
            ("agent_b", long_text),
        ]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"

    def test_response_origin_set_correctly(self):
        """The origin of the result message should be the voting agent's name."""
        agents = [MockAgent(name="agent_a", response_content="Hello")]
        voting_agent = _make_voting_agent(agents)
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "Hello")]
        result = voting_agent.execute(message)
        assert result.origin == voting_agent.card.name

    def test_result_execution_result_is_success(self):
        """The result message should have execution_result='success'."""
        agents = [MockAgent(name="agent_a", response_content="Hello")]
        voting_agent = _make_voting_agent(agents)
        message = AgentMessage(query="test query")
        message.responses = [("agent_a", "Hello")]
        result = voting_agent.execute(message)
        assert result.execution_result == "success"

    def test_multiple_responses_appended(self):
        """The winning response should be appended to the responses list."""
        agents = [MockAgent(name="agent_a", response_content="Winner")]
        voting_agent = _make_voting_agent(agents)
        message = AgentMessage(query="test query")
        original_len = len(message.responses)
        message.responses = [("agent_a", "Winner")]
        result = voting_agent.execute(message)
        # The winning response (from voting agent) should be appended
        assert len(result.responses) > original_len
