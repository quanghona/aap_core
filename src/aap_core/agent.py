import abc
from collections.abc import Callable, Sequence
from typing import Any, Dict, Literal

from a2a.types import AgentCard
from pydantic import BaseModel, Field, PrivateAttr

from .types import AgentMessage, BaseLLMChain


class BaseAgent(abc.ABC, BaseModel):
    """An interface for agent implementations."""

    _state: str = PrivateAttr("idle")
    _composed_state: str = PrivateAttr("idle")

    card: AgentCard = Field(
        ...,
        description="""
        A self-describing manifest for an agent.
        It provides essential metadata including the agent's identity, capabilities, skills, supported communication methods, and security requirements.""",
    )
    state_change_callback: Callable[[str], None] | None = Field(
        default=None,
        description="The callback function to update the state of the agent.",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_composed_state()
        for _, value in self:
            self.__value_deco(value)

    @abc.abstractmethod
    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        """The execution logic of the agent.
        Depends on the detail implementation by subclass, agent can calls tools or RAG or MCP during execution.
        And agent may or may not utilize LLM, depends on the implementation. A lot of agents are pure orchestration without LLM in it.
        During execution, the implementation should update the state of the agent
        and/or push notification to target to update the progress.
        There are 3 types of agent:
        - Remote agent: agent that is running on other server.
        Example: For Google's A2A, we will act as a client agent and communicate with agent using A2A protocol and getback result
        - Local agent: self-defined agent that is running on a this machine.
        - Orchestration agent: itself is a agent, which can perform what normal agent can do, and orchestrate other agents to complete the provided task.

        Args:
            message (AgentMessage): The message to execute the agent with.
            **kwargs: Additional keyword arguments to pass to the LLM generation method.

        Returns:
            AgentMessage: The response from the agent.
        """
        return message

    async def aexecute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        """The asynchronous version of method execute."""
        return self.execute(message, **kwargs)

    @property
    def state(self) -> str:
        """Get the current state of the agent."""
        return self._state

    @property
    def composed_state(self) -> str:
        """Get the current whole state of the agent hierarchy."""
        return self._composed_state

    def _set_composed_state(self) -> None:
        # A subchild class that have child agent in their implementation may need to override this
        self._composed_state = BaseAgent.build_composed_state(self, [], "sequential")

    def _sync_state(self, state: str = "") -> None:
        self._set_composed_state()
        if self.state_change_callback:
            self.state_change_callback(self._composed_state)

    @state.setter
    def state(self, value: str) -> None:
        self._state = value
        self._sync_state()

    def __value_deco(self, value: Any):
        if isinstance(value, str):
            pass
        elif isinstance(value, BaseAgent):
            value.state_change_callback = self._sync_state
        elif isinstance(
            value, BaseLLMChain
        ):  # force name of the chain is the same as the agent
            value.name = self.card.name
        elif isinstance(value, Sequence):
            for agent in value:
                if isinstance(agent, BaseAgent):
                    agent.state_change_callback = self._sync_state
                elif isinstance(value, BaseLLMChain):
                    value.name = self.card.name
        elif isinstance(value, Dict):
            for agent in value.values():
                if isinstance(agent, BaseAgent):
                    agent.state_change_callback = self._sync_state
                elif isinstance(value, BaseLLMChain):
                    value.name = self.card.name

    def __setattr__(self, name: str, value: Any):
        self.__value_deco(value)
        super().__setattr__(name, value)

    @classmethod
    def build_composed_state(
        cls,
        parent_agent: "BaseAgent",
        child_agents: Sequence["BaseAgent"],
        connect_type: Literal["sequential", "parallel"],
    ) -> str:
        """
        Build a composed state string for the given agent and its children.
        The composition rule is following:
        - The order of the agent hierarchy is topdown.
        - A state of a agent is with format: <agent_name>:<agent_state>
        - The level separator is "/". For example: <parent_agent>:<parent_state>/<child_agent1>:<child_state1>
        - If a child is made up of multiple agents, there are 2 type of connections which currently support are "sequential" and "parallel".
            + "sequential": the children are connected with "-" separator and encapsulate with bracket, for example: <parent_agent>:<parent_state>/((<child_agent1>:<child_state1>)-(<child_agent2>:<child_state2>))
            + "parallel": the children are connected with "|" separator and encapsulate with bracket, for example: <parent_agent>:<parent_state>/((<child_agent1>:<child_state1>)|(<child_agent2>:<child_state2>))
        - Lower level of hierarchy will be constructed the same way to form a complete tree of agent state.

        Args:
            parent_agent: The parent agent.
            child_agents: A list of child agents. This list is a direct child only. The function will use the composed_state of the direct child to form the tree. The composed state of the child is assumed to be already built by the child before.
            connect_type: The type of connection between the parent and children, either "sequential" or "parallel".

        Returns:
            str: The composed state string.
        """
        self_state = f"{parent_agent.card.name}:{parent_agent.state}"
        child_states = [child_agent.composed_state for child_agent in child_agents]
        if len(child_states) == 0:
            return self_state
        elif len(child_states) == 1:
            return f"{self_state}/{child_states[0]}"
        else:
            childs = [f"({state})" for state in child_states]
            if connect_type == "sequential":
                return f"{self_state}/({'-'.join(childs)})"
            elif connect_type == "parallel":
                return f"{self_state}/({'|'.join(childs)})"
            else:
                raise ValueError(f"Invalid connect_type: {connect_type}")
