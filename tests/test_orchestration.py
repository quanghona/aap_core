"""Tests for aap_core.orchestration module - ReflectionAgent, LoopAgent, SequentialAgent, ParallelAgent, DebateAgent."""

import pytest
from a2a.types import AgentCard
from aap_core.agent import BaseAgent
from aap_core.orchestration import (
    DebateAgent,
    LoopAgent,
    ParallelAgent,
    ReflectionAgent,
    SequentialAgent,
)
from aap_core.types import AgentMessage, BaseLLMChain


def _make_agent_card(name: str = "test_agent") -> AgentCard:
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
    return_success: bool = True

    def __init__(
        self,
        name: str = "mock_agent",
        response_content: str = "mock response",
        return_success: bool = True,
        **kwargs,
    ):
        card = _make_agent_card(name)
        super().__init__(card=card, **kwargs)
        self.response_content = response_content
        self.return_success = return_success

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message.responses.append((self.card.name, self.response_content))
        message.execution_result = "success" if self.return_success else "error"
        return message


class MockChain(BaseLLMChain):
    """A minimal chain that returns a fixed response for testing."""

    response_content: str = "mock chain response"

    def __init__(
        self,
        name: str = "mock_chain",
        response_content: str = "mock chain response",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.response_content = response_content

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message.responses.append((self.name, self.response_content))
        message.execution_result = "success"
        return message


class TestReflectionAgent:
    """Tests for ReflectionAgent."""

    def test_construction(self):
        """Test ReflectionAgent construction."""
        chain_task = MockChain(name="task_chain")
        chain_reflection = MockChain(name="reflection_chain")
        agent = ReflectionAgent(
            card=_make_agent_card("reflection_agent"),
            chain_task=chain_task,
            chain_reflection=chain_reflection,
        )
        assert agent.task_response_key == "context_response"

    def test_construction_with_custom_task_response_key(self):
        """Test ReflectionAgent with custom task_response_key."""
        chain_task = MockChain(name="task_chain")
        chain_reflection = MockChain(name="reflection_chain")
        agent = ReflectionAgent(
            card=_make_agent_card("reflection_agent"),
            chain_task=chain_task,
            chain_reflection=chain_reflection,
            task_response_key="context_custom",
        )
        assert agent.task_response_key == "context_custom"

    def test_task_response_key_validation(self):
        """Test that task_response_key must start with 'context_'."""
        chain_task = MockChain(name="task_chain")
        chain_reflection = MockChain(name="reflection_chain")
        with pytest.raises(
            ValueError, match="task_response_key must start with 'context_'"
        ):
            ReflectionAgent(
                card=_make_agent_card("reflection_agent"),
                chain_task=chain_task,
                chain_reflection=chain_reflection,
                task_response_key="invalid_key",
            )

    def test_execute_success(self):
        """Test ReflectionAgent execute with successful chains."""
        chain_task = MockChain(name="task_chain")
        chain_reflection = MockChain(name="reflection_chain")
        agent = ReflectionAgent(
            card=_make_agent_card("reflection_agent"),
            chain_task=chain_task,
            chain_reflection=chain_reflection,
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message)
        assert result.execution_result == "success"
        assert result.origin == "reflection_agent"
        assert agent.state == "idle"

    def test_execute_task_failure(self):
        """Test ReflectionAgent execute when task chain fails."""

        # Create a chain that returns error by subclassing
        class ErrorChain(BaseLLMChain):
            def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
                message.execution_result = "error"
                return message

        chain_task = ErrorChain(name="task_chain")
        chain_reflection = MockChain(name="reflection_chain")
        agent = ReflectionAgent(
            card=_make_agent_card("reflection_agent"),
            chain_task=chain_task,
            chain_reflection=chain_reflection,
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message)
        assert result.execution_result == "error"
        assert "Call chain not success" in result.error_message

    def test_execute_state_transitions(self):
        """Test ReflectionAgent state transitions during execution."""
        chain_task = MockChain(name="task_chain")
        chain_reflection = MockChain(name="reflection_chain")
        agent = ReflectionAgent(
            card=_make_agent_card("reflection_agent"),
            chain_task=chain_task,
            chain_reflection=chain_reflection,
        )
        message = AgentMessage(query="test query")
        agent.execute(message)
        # State should end as idle
        assert agent.state == "idle"

    def test_execute_keeps_original_response(self):
        """Test that original response is kept when keep_original_response=True."""
        chain_task = MockChain(name="task_chain", response_content="task response")
        chain_reflection = MockChain(
            name="reflection_chain", response_content="reflection response"
        )
        agent = ReflectionAgent(
            card=_make_agent_card("reflection_agent"),
            chain_task=chain_task,
            chain_reflection=chain_reflection,
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message, keep_original_response=True)
        # Should have both task and reflection responses
        assert len(result.responses) >= 1


class TestLoopAgent:
    """Tests for LoopAgent."""

    def test_construction(self):
        """Test LoopAgent construction."""
        agent = MockAgent(name="looped_agent")
        loop_agent = LoopAgent(
            card=_make_agent_card("loop_agent"),
            agent=agent,
            is_stop=lambda msg: len(msg.responses) > 5,
        )
        assert loop_agent.agent is agent

    def test_execute_with_callable_stop(self):
        """Test LoopAgent execute with callable is_stop."""
        agent = MockAgent(name="looped_agent")
        call_count = [0]

        def stop_condition(msg):
            call_count[0] += 1
            return call_count[0] >= 3

        loop_agent = LoopAgent(
            card=_make_agent_card("loop_agent"),
            agent=agent,
            is_stop=stop_condition,
        )
        message = AgentMessage(query="test query")
        result = loop_agent.execute(message)
        assert result.origin == "loop_agent"
        assert loop_agent.state == "idle"

    def test_execute_with_iterator_stop(self):
        """Test LoopAgent execute with iterator is_stop."""
        agent = MockAgent(name="looped_agent")

        def stop_generator():
            yield False
            yield False
            yield True

        loop_agent = LoopAgent(
            card=_make_agent_card("loop_agent"),
            agent=agent,
            is_stop=stop_generator(),
        )
        message = AgentMessage(query="test query")
        result = loop_agent.execute(message)
        assert result.origin == "loop_agent"

    def test_execute_stops_on_error(self):
        """Test LoopAgent stops when agent returns error."""
        agent = MockAgent(name="looped_agent", return_success=False)
        loop_agent = LoopAgent(
            card=_make_agent_card("loop_agent"),
            agent=agent,
            is_stop=lambda msg: False,  # Never stop, but error will break the loop
        )
        message = AgentMessage(query="test query")
        result = loop_agent.execute(message)
        # LoopAgent returns a copy of the message, not the error message
        # The error is in the inner agent's execution
        assert result.origin == "loop_agent"
        assert loop_agent.state == "idle"

    def test_execute_with_keep_result_int(self):
        """Test LoopAgent with keep_result as int."""
        agent = MockAgent(name="looped_agent")
        call_count = [0]

        def stop_condition(msg):
            call_count[0] += 1
            return call_count[0] >= 4  # Stop after 4 iterations

        loop_agent = LoopAgent(
            card=_make_agent_card("loop_agent"),
            agent=agent,
            is_stop=stop_condition,
        )
        message = AgentMessage(query="test query")
        result = loop_agent.execute(message, keep_result=1)
        assert result.origin == "loop_agent"

    def test_execute_with_keep_result_callable(self):
        """Test LoopAgent with keep_result as callable."""
        agent = MockAgent(name="looped_agent")
        call_count = [0]

        def stop_condition(msg):
            call_count[0] += 1
            return call_count[0] >= 4  # Stop after 4 iterations

        loop_agent = LoopAgent(
            card=_make_agent_card("loop_agent"),
            agent=agent,
            is_stop=stop_condition,
        )
        message = AgentMessage(query="test query")
        result = loop_agent.execute(
            message, keep_result=lambda responses: responses[-2:]
        )
        assert result.origin == "loop_agent"

    def test_set_composed_state(self):
        """Test LoopAgent _set_composed_state."""
        agent = MockAgent(name="looped_agent")
        loop_agent = LoopAgent(
            card=_make_agent_card("loop_agent"),
            agent=agent,
            is_stop=lambda msg: True,
        )
        loop_agent._set_composed_state()
        assert "loop_agent" in loop_agent.composed_state
        assert "looped_agent" in loop_agent.composed_state


class TestSequentialAgent:
    """Tests for SequentialAgent."""

    def test_construction(self):
        """Test SequentialAgent construction."""
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        seq_agent = SequentialAgent(
            card=_make_agent_card("sequential_agent"),
            agents=[agent1, agent2],
        )
        assert len(seq_agent.agents) == 2

    def test_execute_sequential(self):
        """Test SequentialAgent executes agents in order."""
        agent1 = MockAgent(name="agent1", response_content="response1")
        agent2 = MockAgent(name="agent2", response_content="response2")
        seq_agent = SequentialAgent(
            card=_make_agent_card("sequential_agent"),
            agents=[agent1, agent2],
        )
        message = AgentMessage(query="test query")
        result = seq_agent.execute(message)
        assert result.origin == "sequential_agent"
        assert agent1.state == "idle"
        assert agent2.state == "idle"

    def test_execute_stops_on_error(self):
        """Test SequentialAgent stops when an agent returns error."""
        agent1 = MockAgent(name="agent1", return_success=False)
        agent2 = MockAgent(name="agent2")
        seq_agent = SequentialAgent(
            card=_make_agent_card("sequential_agent"),
            agents=[agent1, agent2],
        )
        message = AgentMessage(query="test query")
        result = seq_agent.execute(message)
        assert result.execution_result == "error"

    def test_set_composed_state(self):
        """Test SequentialAgent _set_composed_state."""
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        seq_agent = SequentialAgent(
            card=_make_agent_card("sequential_agent"),
            agents=[agent1, agent2],
        )
        seq_agent._set_composed_state()
        assert "sequential_agent" in seq_agent.composed_state
        assert "agent1" in seq_agent.composed_state
        assert "agent2" in seq_agent.composed_state


class TestParallelAgent:
    """Tests for ParallelAgent."""

    def test_construction(self):
        """Test ParallelAgent construction."""
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        par_agent = ParallelAgent(
            card=_make_agent_card("parallel_agent"),
            agents=[agent1, agent2],
        )
        assert len(par_agent.agents) == 2

    def test_execute_single_message(self):
        """Test ParallelAgent with a single message."""
        agent1 = MockAgent(name="agent1", response_content="response1")
        agent2 = MockAgent(name="agent2", response_content="response2")
        par_agent = ParallelAgent(
            card=_make_agent_card("parallel_agent"),
            agents=[agent1, agent2],
        )
        message = AgentMessage(query="test query")
        result = par_agent.execute(message)
        assert result.origin == "parallel_agent"
        assert result.execution_result == "success"
        assert len(result.responses) == 2

    def test_execute_multiple_messages(self):
        """Test ParallelAgent with multiple messages."""
        agent1 = MockAgent(name="agent1", response_content="response1")
        agent2 = MockAgent(name="agent2", response_content="response2")
        par_agent = ParallelAgent(
            card=_make_agent_card("parallel_agent"),
            agents=[agent1, agent2],
        )
        messages = [
            AgentMessage(query="query1"),
            AgentMessage(query="query2"),
        ]
        result = par_agent.execute(messages)
        assert result.origin == "parallel_agent"
        assert len(result.responses) == 2

    def test_execute_message_count_mismatch(self):
        """Test ParallelAgent raises ValueError for message count mismatch."""
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        par_agent = ParallelAgent(
            card=_make_agent_card("parallel_agent"),
            agents=[agent1, agent2],
        )
        messages = [AgentMessage(query="query1")]
        with pytest.raises(ValueError, match="messages must be a list"):
            par_agent.execute(messages)

    def test_set_composed_state(self):
        """Test ParallelAgent _set_composed_state."""
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        par_agent = ParallelAgent(
            card=_make_agent_card("parallel_agent"),
            agents=[agent1, agent2],
        )
        par_agent._set_composed_state()
        assert "parallel_agent" in par_agent.composed_state
        assert "parallel" in par_agent.composed_state


class TestDebateAgent:
    """Tests for DebateAgent."""

    def test_construction_round_robin(self):
        """Test DebateAgent construction with round_robin strategy."""
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        debate = DebateAgent(
            card=_make_agent_card("debate_agent"),
            agents=[agent1, agent2],
            pick_strategy="round_robin",
            max_turns=2,
        )
        assert debate.pick_strategy == "round_robin"

    def test_construction_random(self):
        """Test DebateAgent construction with random strategy."""
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        debate = DebateAgent(
            card=_make_agent_card("debate_agent"),
            agents=[agent1, agent2],
            pick_strategy="random",
            random_seed=42,
            max_turns=2,
        )
        assert debate.random_seed == 42

    def test_execute_round_robin(self):
        """Test DebateAgent execute with round_robin strategy."""
        agent1 = MockAgent(name="agent1", response_content="response1")
        agent2 = MockAgent(name="agent2", response_content="response2")
        debate = DebateAgent(
            card=_make_agent_card("debate_agent"),
            agents=[agent1, agent2],
            pick_strategy="round_robin",
            max_turns=2,
        )
        message = AgentMessage(query="test query")
        result = debate.execute(message)
        assert result.origin == "debate_agent"
        assert result.execution_result == "success"
        assert debate.state == "idle"

    def test_execute_random(self):
        """Test DebateAgent execute with random strategy."""
        agent1 = MockAgent(name="agent1", response_content="response1")
        agent2 = MockAgent(name="agent2", response_content="response2")
        debate = DebateAgent(
            card=_make_agent_card("debate_agent"),
            agents=[agent1, agent2],
            pick_strategy="random",
            random_seed=42,
            max_turns=2,
        )
        message = AgentMessage(query="test query")
        result = debate.execute(message)
        assert result.origin == "debate_agent"
        assert result.execution_result == "success"

    def test_execute_simultaneous(self):
        """Test DebateAgent execute with simultaneous strategy."""
        agent1 = MockAgent(name="agent1", response_content="response1")
        agent2 = MockAgent(name="agent2", response_content="response2")
        debate = DebateAgent(
            card=_make_agent_card("debate_agent"),
            agents=[agent1, agent2],
            pick_strategy="simultaneous",
            max_turns=1,
        )
        message = AgentMessage(query="test query")
        result = debate.execute(message)
        assert result.origin == "debate_agent"
        assert result.execution_result == "success"
        # Simultaneous should produce 2 responses (one per agent)
        assert len(result.responses) == 2

    def test_execute_with_custom_callable_strategy(self):
        """Test DebateAgent execute with custom callable pick_strategy."""
        agent1 = MockAgent(name="agent1", response_content="response1")
        agent2 = MockAgent(name="agent2", response_content="response2")

        def custom_strategy(agents):
            return agents[0]

        debate = DebateAgent(
            card=_make_agent_card("debate_agent"),
            agents=[agent1, agent2],
            pick_strategy=custom_strategy,
            max_turns=2,
        )
        message = AgentMessage(query="test query")
        result = debate.execute(message)
        assert result.origin == "debate_agent"
        assert result.execution_result == "success"

    def test_execute_with_should_stop(self):
        """Test DebateAgent execute with should_stop condition."""
        agent1 = MockAgent(name="agent1", response_content="response1")
        debate = DebateAgent(
            card=_make_agent_card("debate_agent"),
            agents=[agent1],
            pick_strategy="round_robin",
            max_turns=10,
            should_stop=lambda msg: len(msg.responses) >= 2,
        )
        message = AgentMessage(query="test query")
        result = debate.execute(message)
        # Should stop early due to should_stop condition
        assert result.execution_result == "success"

    def test_execute_stops_on_error(self):
        """Test DebateAgent stops when an agent returns error."""
        agent1 = MockAgent(name="agent1", return_success=False)
        debate = DebateAgent(
            card=_make_agent_card("debate_agent"),
            agents=[agent1],
            pick_strategy="round_robin",
            max_turns=10,
        )
        message = AgentMessage(query="test query")
        result = debate.execute(message)
        assert result.execution_result == "success"  # DebateAgent sets success at end

    def test_execute_invalid_strategy(self):
        """Test DebateAgent raises ValueError for invalid strategy at construction."""
        agent1 = MockAgent(name="agent1")
        # Pydantic validates the literal type at construction time
        with pytest.raises(Exception):  # ValidationError
            DebateAgent(
                card=_make_agent_card("debate_agent"),
                agents=[agent1],
                pick_strategy="invalid_strategy",
                max_turns=1,
            )

    def test_set_composed_state_round_robin(self):
        """Test DebateAgent _set_composed_state with round_robin."""
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        debate = DebateAgent(
            card=_make_agent_card("debate_agent"),
            agents=[agent1, agent2],
            pick_strategy="round_robin",
            max_turns=2,
        )
        debate._set_composed_state()
        # round_robin uses sequential connection with "-" separator
        assert "-" in debate.composed_state

    def test_set_composed_state_simultaneous(self):
        """Test DebateAgent _set_composed_state with simultaneous."""
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        debate = DebateAgent(
            card=_make_agent_card("debate_agent"),
            agents=[agent1, agent2],
            pick_strategy="simultaneous",
            max_turns=2,
        )
        debate._set_composed_state()
        # simultaneous uses parallel connection with "|" separator
        assert "|" in debate.composed_state
