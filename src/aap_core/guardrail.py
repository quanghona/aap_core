import abc

from .types import AgentMessage, BaseChain


class BaseGuardRail(BaseChain):
    @abc.abstractmethod
    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        pass


class PassGuardRail(BaseGuardRail):
    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return message
