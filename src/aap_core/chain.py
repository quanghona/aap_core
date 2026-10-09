import abc
from collections.abc import Sequence
from typing import Callable, Generic, List, Tuple

from pydantic import Field, PrivateAttr

from .guardrail import BaseGuardRail, PassGuardRail
from .prompt_augmenter import (
    BasePromptAugmenter,
    IdentityPromptAugmenter,
)
from .types import AgentMessage, BaseLLMChain, ChainMessage, ChainResponse, TokenUsage


class BaseCausalMultiTurnsChain(BaseLLMChain, Generic[ChainMessage, ChainResponse]):
    """
    Base class for multi-turns chains.
    """

    _last_response_as_context: str | None = PrivateAttr(default=None)
    include_history: int = Field(
        default=0, description="number of history message turns to include"
    )
    store_immediate_steps: bool = Field(
        default=False, description="store intermediate steps output"
    )

    max_turns: int = Field(
        default=50, description="maximum number of turns before stopping"
    )

    def final_response_as_context(self, key: str) -> None:
        self._last_response_as_context = key.replace("context_", "")

    @abc.abstractmethod
    def _prepare_conversation(self, message: AgentMessage) -> List[ChainMessage]:
        pass

    @abc.abstractmethod
    def _generate_response(
        self, conversation: List[ChainMessage], **kwargs
    ) -> Tuple[List[ChainMessage], ChainResponse, bool, TokenUsage]:
        pass

    @abc.abstractmethod
    def _process_tools(
        self,
        conversation: List[ChainMessage],
        response: ChainResponse,
    ) -> List[ChainMessage]:
        pass

    @abc.abstractmethod
    def _append_responses(
        self, message: AgentMessage, conversation: List[ChainMessage]
    ) -> AgentMessage:
        pass

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if message.token_usage is None:
            message.token_usage = {
                "steps": [],
                "total": TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            }
        if "steps" not in message.token_usage:
            message.token_usage["steps"] = []
        if "total" not in message.token_usage:
            message.token_usage["total"] = TokenUsage(
                input_tokens=0, output_tokens=0, total_tokens=0
            )

        def add_token(usage: TokenUsage):
            message.token_usage["steps"].append(usage)  # type: ignore
            message.token_usage["total"]["input_tokens"] += usage["input_tokens"]  # type: ignore
            message.token_usage["total"]["output_tokens"] += usage["output_tokens"]  # type: ignore
            message.token_usage["total"]["total_tokens"] += usage["total_tokens"]  # type: ignore

        # Template method — ReAct loop: Thought → Act → Thought → ... → Final
        conversation = self._prepare_conversation(message)
        turn = 0
        while turn < self.max_turns:
            conversation, response, has_tool, usage = self._generate_response(
                conversation, **kwargs
            )
            add_token(usage)

            if not has_tool:
                break

            conversation = self._process_tools(conversation, response)
            turn += 1

        message = self._append_responses(message, conversation)

        if self._last_response_as_context is not None:
            if message.context is None:
                message.context = {}
            message.context[self._last_response_as_context] = message.responses.pop()[1]

        message.origin = self.name
        message.execution_result = "success"
        return message


class TypicalLLMChain(BaseLLMChain):
    """
    A LLM chain that consists of all components in a typical response generation.
    The chain has 4 steps:
    - input guardrail: validate the input message
    - prompt enhancement: add more context to the prompt or rewrite / refine the prompt
    - generate: generate the response. This steps should also handle the tool calling procedure and finalize the response
    - output guardrail: validate the output message"""

    input_guardrail: BaseGuardRail = Field(
        default=PassGuardRail(),
        description="validate the input message",
    )
    prompt_augmenter: BasePromptAugmenter = Field(
        default=IdentityPromptAugmenter(),
        description="add more context to the prompt or rewrite / refine the prompt",
    )
    tools: Sequence[Callable] = Field(
        default=[],
        description="tools to call for the LLM",
    )
    output_guardrail: BaseGuardRail = Field(
        default=PassGuardRail(),
        description="validate the output message",
    )

    @abc.abstractmethod
    def generate(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # call tool if any and finalize response
        pass

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message = self.input_guardrail(message)
        message = self.prompt_augmenter(message)
        message = self.generate(message, **kwargs)
        message = self.output_guardrail(message)
        return message
