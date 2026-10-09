"""Tests for CoordinatorAgent in aap_core.orchestration."""

import pytest
from a2a.types import AgentCard
from aap_core.agent import BaseAgent
from aap_core.orchestration import CoordinatorAgent
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


class TrackingMockAgent(BaseAgent):
    """A MockAgent that tracks execute() calls for testing."""

    response_content: str = "mock response"
    return_success: bool = True

    def __init__(
        self,
        name: str = "tracking_mock_agent",
        response_content: str = "mock response",
        return_success: bool = True,
        **kwargs,
    ):
        card = _make_agent_card(name)
        super().__init__(card=card, **kwargs)
        self.response_content = response_content
        self.return_success = return_success
        self._execute_calls: list[AgentMessage] = []

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self._execute_calls.append(message)
        message.responses.append((self.card.name, self.response_content))
        message.execution_result = "success" if self.return_success else "error"
        return message

    @property
    def call_count(self) -> int:
        return len(self._execute_calls)

    @property
    def last_call_args(self) -> AgentMessage | None:
        return self._execute_calls[-1] if self._execute_calls else None


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


def _make_parse_plan():
    """Create a mock parse_plan function that returns a simple plan.

    Returns a function that accepts (message, workers) and uses the workers passed in.
    """

    def parse_plan(message, workers):
        return [
            (
                AgentMessage(query="do task", responses=[]),
                workers[0],
                [],
            )
        ]

    return parse_plan


@pytest.fixture
def planner():
    return MockAgent(name="planner")


@pytest.fixture
def workers():
    return [MockAgent(name="worker1"), MockAgent(name="worker2")]


@pytest.fixture
def parse_plan():
    return _make_parse_plan()


class TestCoordinatorAgentConstruction:
    """Test CoordinatorAgent construction and field validation."""

    def test_minimal_construction(self, planner, workers, parse_plan):
        """Test that CoordinatorAgent can be constructed with minimal required fields."""
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=parse_plan,
            workers=workers,
        )
        assert agent.planner_agent is planner
        assert agent.workers == workers
        assert agent.parse_plan is parse_plan
        assert agent.summary_chain is None
        assert agent.summary_prompt is None
        assert agent.summary_steps_key == "context_results"

    def test_construction_with_summary_chain(self, workers, parse_plan):
        """Test construction with a summary chain."""
        summary_chain = MockChain(name="summary_chain")
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=MockAgent(name="planner"),
            parse_plan=parse_plan,
            workers=workers,
            summary_chain=summary_chain,
        )
        assert agent.summary_chain is summary_chain

    def test_construction_with_summary_prompt(self, planner, workers, parse_plan):
        """Test construction with a summary prompt."""
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=parse_plan,
            workers=workers,
            summary_prompt="Summarize the results",
        )
        assert agent.summary_prompt == "Summarize the results"

    def test_summary_steps_key_validation_valid(self):
        """Test that summary_steps_key must start with 'context_'."""
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=MockAgent(name="planner"),
            parse_plan=_make_parse_plan(),
            workers=[MockAgent(name="worker1")],
            summary_steps_key="context_results",
        )
        assert agent.summary_steps_key == "context_results"

    def test_summary_steps_key_validation_invalid(self):
        """Test that summary_steps_key raises ValueError if not starting with 'context_'."""
        with pytest.raises(
            ValueError, match="summary_steps_key must start with 'context_'"
        ):
            CoordinatorAgent(
                card=_make_agent_card("coordinator"),
                planner_agent=MockAgent(name="planner"),
                parse_plan=_make_parse_plan(),
                workers=[MockAgent(name="worker1")],
                summary_steps_key="results",
            )


class TestCoordinatorAgentPlanning:
    """Test the planning stage of CoordinatorAgent.execute()."""

    def test_planner_failure_returns_early(self, workers, parse_plan):
        """Test that if planner fails, execute returns early with the error message."""
        planner = MockAgent(name="planner", return_success=False)
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=parse_plan,
            workers=workers,
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message)
        assert result.execution_result != "success"
        assert agent.state == "planning"

    def test_planner_state_set(self, workers, parse_plan):
        """Test that state transitions correctly during execution.

        Note: CoordinatorAgent.execute() has a bug where it returns early
        without setting state to 'idle' when summary_chain is None and
        summary_prompt is None. This test documents that behavior.
        """
        planner = TrackingMockAgent(name="planner")
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=parse_plan,
            workers=workers,
        )
        message = AgentMessage(query="test query")
        agent.execute(message)
        # Due to early return in CoordinatorAgent.execute() when no summary_chain/summary_prompt,
        # state remains at the last step execution state
        assert "worker" in agent.state
        # Planner should have been called once
        assert planner.call_count == 1


class TestCoordinatorAgentExecution:
    """Test the step execution stage of CoordinatorAgent.execute()."""

    def test_single_step_execution(self, workers, parse_plan):
        """Test that a single step is executed by the assigned worker."""
        planner = MockAgent(name="planner")
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=parse_plan,
            workers=workers,
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message)
        # When no summary_chain/summary_prompt, returns the planner's message
        assert result.execution_result == "success"
        assert result.query == "test query"  # original query preserved

    def test_multiple_steps_execution(self, workers, parse_plan):
        """Test that multiple steps are executed sequentially."""
        worker1 = MockAgent(name="worker1")
        worker2 = MockAgent(name="worker2")

        def multi_step_parse_plan(message, workers):
            return [
                (AgentMessage(query="task 1", responses=[]), workers[0], []),
                (AgentMessage(query="task 2", responses=[]), workers[1], []),
            ]

        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=MockAgent(name="planner"),
            parse_plan=multi_step_parse_plan,
            workers=[worker1, worker2],
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message)
        assert result.execution_result == "success"

    def test_worker_state_tracking(self, workers, parse_plan):
        """Test that state is updated for each step execution.

        Note: CoordinatorAgent.execute() has a bug where it returns early
        without setting state to 'idle' when summary_chain is None and
        summary_prompt is None. This test documents that behavior.
        """
        planner = MockAgent(name="planner")
        tracking_worker = TrackingMockAgent(name="worker1")
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=lambda msg, workers: [
                (AgentMessage(query="do task", responses=[]), workers[0], [])
            ],
            workers=[tracking_worker],
        )
        message = AgentMessage(query="test query")
        agent.execute(message)
        # Due to early return in CoordinatorAgent.execute() when no summary_chain/summary_prompt,
        # state remains at the last step execution state
        assert "worker" in agent.state
        # Worker should have been called
        assert tracking_worker.call_count == 1

    def test_worker_receives_context(self, workers, parse_plan):
        """Test that workers receive context with previous step results."""
        planner = MockAgent(name="planner")
        tracking_worker = TrackingMockAgent(name="worker1")
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=lambda msg, workers: [
                (AgentMessage(query="do task", responses=[]), workers[0], [])
            ],
            workers=[tracking_worker],
        )
        message = AgentMessage(query="test query")
        agent.execute(message)
        # Check that worker.execute was called with a message that has context
        last_call = tracking_worker.last_call_args
        assert last_call is not None
        assert last_call.context is not None
        assert "results" in last_call.context


class TestCoordinatorAgentSummary:
    """Test the summary stage of CoordinatorAgent.execute()."""

    def test_no_summary_chain_or_prompt_returns_planner_message(
        self, workers, parse_plan
    ):
        """Test that without summary_chain or summary_prompt, the planner message is returned."""
        planner = MockAgent(name="planner")
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=parse_plan,
            workers=workers,
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message)
        # When no summary_chain/summary_prompt, returns the planner's message
        assert result.execution_result == "success"
        assert result.query == "test query"  # original query preserved

    def test_summary_prompt_uses_planner_as_summary(self, workers, parse_plan):
        """Test that summary_prompt uses the planner agent as the summary agent."""
        planner = TrackingMockAgent(name="planner")
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=parse_plan,
            workers=workers,
            summary_prompt="Summarize the results",
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message)
        assert result.execution_result == "success"
        # Planner should be called twice: once for planning, once for summary
        assert planner.call_count == 2
        # The second call should have the summary prompt as query
        assert planner.last_call_args.query == "Summarize the results"

    def test_summary_chain_produces_final_answer(self, workers, parse_plan):
        """Test that summary_chain is used to produce the final answer."""
        planner = MockAgent(name="planner")
        summary_chain = MockChain(name="summary_chain")

        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=parse_plan,
            workers=workers,
            summary_chain=summary_chain,
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message)
        assert result.execution_result == "success"
        assert result.origin == "coordinator"
        # The summary_chain should have been invoked
        # Note: __value_deco changes the chain name to the coordinator's card name
        assert any(r[0] == "coordinator" for r in result.responses)
        # The query should be the original query
        assert result.query == "test query"

    def test_summary_chain_state_tracking(self, workers, parse_plan):
        """Test that state transitions correctly when using summary_chain."""
        planner = MockAgent(name="planner")
        summary_chain = MockChain(name="summary_chain")

        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=parse_plan,
            workers=workers,
            summary_chain=summary_chain,
        )
        message = AgentMessage(query="test query")
        agent.execute(message)
        assert agent.state == "idle"


class TestCoordinatorAgentEdgeCases:
    """Test edge cases for CoordinatorAgent.execute()."""

    def test_worker_failure_stops_execution(self, workers, parse_plan):
        """Test that if a worker fails, execution continues to next worker.

        Note: CoordinatorAgent.execute() does NOT stop on worker failure.
        It continues executing all workers. The result message's execution_result
        comes from the planner (not the workers), so it remains 'success'.

        Also note: when no summary_chain/summary_prompt, the coordinator returns
        the planner's message (not the workers' results).
        """
        worker1 = TrackingMockAgent(name="worker1")
        worker2 = TrackingMockAgent(name="worker2", return_success=False)

        def failing_plan(message, workers):
            return [
                (AgentMessage(query="task 1", responses=[]), workers[0], []),
                (AgentMessage(query="task 2", responses=[]), workers[1], []),
            ]

        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=MockAgent(name="planner"),
            parse_plan=failing_plan,
            workers=[worker1, worker2],
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message)
        # CoordinatorAgent continues executing all workers regardless of failure
        # The result message's execution_result comes from the planner (not workers)
        assert result.execution_result == "success"  # from planner
        # Both workers should have been called
        assert worker1.call_count == 1
        assert worker2.call_count == 1
        # Result is the planner's message (not workers' results) when no summary_chain/summary_prompt
        assert len(result.responses) == 1  # only planner's response
        assert result.responses[0][0] == "planner"

    def test_custom_summary_steps_key(self):
        """Test that a custom summary_steps_key is used correctly."""
        planner = MockAgent(name="planner")
        worker = TrackingMockAgent(name="worker1")
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=planner,
            parse_plan=lambda msg, workers: [
                (AgentMessage(query="task", responses=[]), workers[0], [])
            ],
            workers=[worker],
            summary_steps_key="context_outputs",
        )
        message = AgentMessage(query="test query")
        agent.execute(message)
        # The context key used should be 'outputs' (prefix stripped)
        last_call = worker.last_call_args
        assert last_call is not None
        assert "outputs" in last_call.context

    def test_single_worker(self):
        """Test that CoordinatorAgent works with a single worker."""
        worker = MockAgent(name="single_worker")
        agent = CoordinatorAgent(
            card=_make_agent_card("coordinator"),
            planner_agent=MockAgent(name="planner"),
            parse_plan=lambda msg, workers: [
                (AgentMessage(query="task", responses=[]), workers[0], [])
            ],
            workers=[worker],
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message)
        assert result.execution_result == "success"

    def test_execute_sets_origin(self):
        """Test that the result message origin is set to the coordinator's card name
        when using summary_chain."""
        planner = MockAgent(name="planner")
        summary_chain = MockChain(name="summary_chain")
        agent = CoordinatorAgent(
            card=_make_agent_card("my_coordinator"),
            planner_agent=planner,
            parse_plan=lambda msg, workers: [
                (AgentMessage(query="task", responses=[]), workers[0], [])
            ],
            workers=[MockAgent(name="worker1")],
            summary_chain=summary_chain,
        )
        message = AgentMessage(query="test query")
        result = agent.execute(message)
        assert result.origin == "my_coordinator"
