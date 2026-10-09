from .types import AgentMessage, BaseChain, BaseLLMChain
from pydantic import Field


class ChainOfThought(BaseChain):
    def with_self_consistency(self, max_branches: int = 1):
        pass


class TreeOfThought(BaseChain):
    pass


class GraphOfThought(BaseChain):
    pass


class SelfConsistencyChainOfThought(BaseChain):
    """CoT-SC. This orchestration technically is a combination of a LoopAgent and VotingAgent"""

    chain: BaseLLMChain = Field(
        ...,
        description="LLM chain which have CoT capability. the simplest way is to add a sentence in the prompt force response reason step by step",
    )
    temperature: float = Field(
        1.5,
        description="temperature for the LLM response. This should have a reasonably high enough to generate multiple reasoning paths.",
    )
    max_turns: int = Field(
        5,
        description="The maximum number of branch to expand",
        ge=1,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loop_agent = LoopAgent()
        self._voting_agent = VotingAgent()

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self.chain(message, **kwargs)
