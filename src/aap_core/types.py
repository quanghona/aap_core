import abc
import json
import logging
from typing import Any, Dict, List, Literal, Tuple, TypeVar
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from .utils import resolve_media

logger = logging.getLogger(__name__)


ChainMessage = TypeVar("ChainMessage")
ChainResponse = TypeVar("ChainResponse")
AgentResponse = Tuple[str, str]
ContentType = Literal["text", "image", "audio", "video", "document"]


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class MediaReference(TypedDict):
    """A resolved media reference, ready to be mapped to a framework-native content part.

    Attributes:
        content_type: The modality of this reference, from ContentType. Refs are
            self-describing so chains can map each to the correct native part type.
        kind: How the media is carried. Either a remote URL or base64-encoded bytes
            (local paths are read and converted to base64 by resolve_media).
        value: The URL string or the base64 payload (without the data: prefix).
        mime_type: Detected or guessed MIME type, e.g. "image/png".
    """

    content_type: ContentType
    kind: Literal["url", "base64"]
    value: str
    mime_type: str


class AgentMessage(BaseModel):
    query: str = Field(
        ...,
        description="""The user query consumed by agent's LLM.
        In this message class, we don't store the system prompt and user template.
        Because each agent have their own system prompt and user template, which is not shared between agents.""",
    )
    query_media: List[Tuple[ContentType, str]] | None = Field(
        default=None, description="The media content associated with the query."
    )
    origin: str | None = Field(
        default=None, description="The agent that send this message"
    )
    responses: List[AgentResponse] = Field(
        default=[],
        description="""
        If an agent generate multiple responses, either by same or different subagents, all of them will be stored here.
        In each response tuple, the first one should be agent name or index, and second is the response""",
    )
    context: dict | None = Field(
        default=None,
        description="""
        Agent additional material, which probably is the output of other agent or user entered.
        There are various types of context produced by user and other agents.
        The prompt that comsume this context need to explicitly know the format of this context.""",
    )
    execution_result: Literal["success", "error"] | None = Field(
        default=None,
        description="The execution result of the agent. Can be success or error",
    )
    error_message: str | None = Field(
        default=None, description="The error message if the execution result is error."
    )
    media: List[Tuple[ContentType, str]] | None = Field(
        default=None,
        description="The additional media content. Can be image, video, audio, etc.",
    )
    token_usage: Dict[str, TokenUsage | List[TokenUsage]] | None = Field(
        default=None,
        description="Token usage of the LLM(s) one agent or cross agents.",
    )

    def flatten_dict(self, d: dict, parent_key: str = "", sep="_") -> Dict[str, Any]:
        """
        Flattens a nested dictionary into a single-level dictionary.

        Args:
            d (dict): The dictionary to flatten.
            parent_key (str): The prefix for keys in the flattened dictionary.
            sep (str): The separator used to join parent and child keys.

        Returns:
            dict: The flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the AgentMessage object into a dictionary format.

        Returns a dictionary that contains all the information in the AgentMessage object.
        If the context is not None, it will be flattened and added to the returned dictionary.
        All the data in context will have prefix of 'context_' by default.
        For example, if the context is {'a': 1, 'b': 2}, the returned dictionary will contains {'context_a': 1, 'context_b': 2}

        Returns:
            dict: A dictionary that contains all the information in the AgentMessage object.
        """
        msg_json = self.model_dump(exclude_none=True, exclude={"context"})
        if self.context:
            context_dict = self.flatten_dict(self.context, parent_key="context")
            for k, v in context_dict.items():
                msg_json[k] = v
        return msg_json

    def format_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for prompt-template interpolation.

        Same as to_dict() but without the media fields (query_media and media).
        Media must never be interpolated into a text template as a Python repr;
        it is delivered to the model as native content parts instead.
        Chains should call this method, not to_dict(), when formatting user prompt templates.

        Returns:
            dict: Template-safe kwargs containing query, responses and flattened context.
        """
        kwargs = self.to_dict()
        kwargs.pop("query_media", None)
        kwargs.pop("media", None)
        return kwargs

    def dump_json(self) -> str:
        """
        Converts the AgentMessage object into a JSON string.

        Returns:
            str: A JSON string that contains all the information in the AgentMessage object.
        """
        msg_dict = self.to_dict()
        return json.dumps(msg_dict)


class BaseChain(abc.ABC, BaseModel):
    @abc.abstractmethod
    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        pass


class BaseLLMChain(BaseChain):
    """
    Base class for LLM chains.

    Implementation note:
    - We only need to override the invoke method for the chain logic.
    The __call__ method is left untouched under normal circumstances.
    - For token usage, the token count is store in the message.token_usage object.
    Depends on the child class implementation, the usage may contains number of token
    for individual steps or total token in a whole pipeline. Therefore, the key of
    the token_usage dict is managed by the child class.
    You can refer an token count object in each framework:
        + langchain: [AIMessage.usage_metadata](https://reference.langchain.com/python/langchain/messages/?_gl=1*w2hpy3*_gcl_au*ODg0NzQ4OTg1LjE3Njk0Nzc1MzM.*_ga*MTgzMjI1MTQwOS4xNzU5MjMyMjIy*_ga_47WX3HKKY2*czE3Njk1ODgxODkkbzE5OSRnMCR0MTc2OTU4ODE4OSRqNjAkbDAkaDA.#langchain.messages.AIMessage.usage_metadata) in the response message object
        and [UsageMetadata](https://reference.langchain.com/python/langchain/messages/?_gl=1*w2hpy3*_gcl_au*ODg0NzQ4OTg1LjE3Njk0Nzc1MzM.*_ga*MTgzMjI1MTQwOS4xNzU5MjMyMjIy*_ga_47WX3HKKY2*czE3Njk1ODgxODkkbzE5OSRnMCR0MTc2OTU4ODE4OSRqNjAkbDAkaDA.#langchain.messages.UsageMetadata) class
        + llama-index: There is a function to get the token in the llamaindex library is [token_from_response](https://developers.llamaindex.ai/python/framework-api-reference/callbacks/token_counter/#llama_index.core.callbacks.token_counting.get_tokens_from_response)
        + dspy: You need to set the configure enable `track_usage` either by `with dspy.context(..., track_usage=True) or `dspy.configure`.
        Then in the response object, there is a method `.get_lm_usage` to get the token count object.
        Otherwise the token object is empty.
        + transformers: the pure output of the model is a tensor of token. We just simple get the length (its shape) of the token to obtain the result.

    For convenience, we already provided a simple utility `token_from_response` in all integration packages to convert the token count object to a unified TokenUsage object.
    """

    name: str = Field(
        "chain",
        description="The name of the chain. Should be same at agent who hold this chain for easy to operate.",
    )
    media_support: Dict[ContentType, Literal["auto", "enabled", "disabled"]] = Field(
        default_factory=dict,
        description="""Per-modality policy controlling whether media of each ContentType
        from AgentMessage.query_media is delivered to the model as native content parts.
        Unlisted modalities default to "auto". Models support subsets of modalities,
        so the policy and capability detection are per-modality, never global.
        - "auto": rely on detect_capabilities(). Media of that modality is
          force-disabled when the model is known not to accept it; undetectable
          is assumed capable (fail loud).
        - "enabled": always attach parts of that modality, overriding detection.
        - "disabled": never attach parts of that modality.

        Currently only "image" refs are mapped to native parts by the integration
        packages; the other ContentType keys are wired at the base layer so adding
        a modality later is a package-side mapping plus sniffing-table change, not
        a redesign of this gate.""",
    )

    @abc.abstractmethod
    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        pass

    async def ainvoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self.invoke(message, **kwargs)

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self.invoke(message, **kwargs)

    def detect_capabilities(self) -> Dict[ContentType, bool]:
        """Report which modalities the underlying model is known to accept.

        Hook for subclasses in integration packages to query metadata that is
        authoritative for the exact model revision - langchain model profiles,
        the Ollama server's capabilities list, Hugging Face model-card task
        tags, transformers processor components. Model-family name heuristics
        are NOT a source: variants of one family differ per modality, so names
        misreport exactly the cases that matter. Models support subsets of
        modalities, so the result is per-modality.

        Returns:
            Dict[ContentType, bool]: Known capabilities per modality. Modalities
            absent from the dict are undecidable and treated as capable
            (fail-loud default).
        """
        return {}

    def _modality_ok(self, content_type: ContentType) -> bool:
        """Gate deciding whether parts of one modality may be attached."""
        policy = self.media_support.get(content_type, "auto")
        if policy == "enabled":
            return True
        if policy == "disabled":
            return False
        return self.detect_capabilities().get(content_type) is not False

    def _media_parts(
        self, message: AgentMessage, content_type: ContentType
    ) -> List[MediaReference]:
        """Resolve media of one modality carried by a message into framework-ready refs.

        This is the single funnel every chain must use to obtain media for a
        message. It returns an empty list when the message carries no media of
        that modality or when the modality gate (_modality_ok) is closed, in
        which case any carried items are dropped and a warning is logged.

        Args:
            message: The agent message whose query_media may carry media.
            content_type: The modality to extract, e.g. "image".

        Returns:
            List[MediaRef]: Resolved references of that modality, in original order.
        """
        items = [
            (media_type, value)
            for media_type, value in message.query_media or []
            if media_type == content_type
        ]
        if not items:
            return []
        if not self._modality_ok(content_type):
            logger.warning(
                f"Chain '{self.name}' model does not support {content_type} input; "
                f"dropping {len(items)} {content_type}(s) from query_media."
            )
            return []
        return resolve_media(items)

    def _image_parts(self, message: AgentMessage) -> List[MediaReference]:
        """Resolve the image media of a message.

        Sugar for _media_parts(message, "image"). Image is the first modality
        wired through the integration packages; audio/video/document will get
        their own sugar (or call _media_parts directly) when implemented.
        """
        return self._media_parts(message, "image")
