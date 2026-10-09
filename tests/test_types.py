"""Tests for aap_core.types module - AgentMessage, BaseChain, BaseLLMChain, TokenUsage."""

import json
import logging

import pytest
from aap_core.types import (
    AgentMessage,
    BaseChain,
    BaseLLMChain,
    TokenUsage,
)
from pydantic import PrivateAttr


class TestTokenUsage:
    """Tests for TokenUsage TypedDict."""

    def test_token_usage_creation(self):
        """Test creating a TokenUsage dict."""
        usage: TokenUsage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_token_usage_zero_values(self):
        """Test TokenUsage with zero values."""
        usage: TokenUsage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        assert usage["total_tokens"] == 0

    def test_token_usage_large_values(self):
        """Test TokenUsage with large values."""
        usage: TokenUsage = {
            "input_tokens": 1000000,
            "output_tokens": 500000,
            "total_tokens": 1500000,
        }
        assert usage["total_tokens"] == 1500000


class TestAgentMessage:
    """Tests for AgentMessage model."""

    def test_construction_required_fields(self):
        """Test AgentMessage construction with required fields."""
        msg = AgentMessage(query="test query")
        assert msg.query == "test query"
        assert msg.responses == []
        assert msg.context is None
        assert msg.execution_result is None

    def test_construction_with_all_fields(self):
        """Test AgentMessage construction with all fields."""
        msg = AgentMessage(
            query="test query",
            query_media=[("text", "hello")],
            origin="test_agent",
            responses=[("agent1", "response1")],
            context={"key": "value"},
            execution_result="success",
            error_message=None,
            media=[("image", "base64data")],
            token_usage={
                "total": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            },
        )
        assert msg.query == "test query"
        assert msg.origin == "test_agent"
        assert msg.execution_result == "success"
        assert len(msg.responses) == 1

    def test_flatten_dict_simple(self):
        """Test flatten_dict with simple nested dict."""
        msg = AgentMessage(query="test")
        nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        result = msg.flatten_dict(nested)
        assert result == {"a": 1, "b_c": 2, "b_d_e": 3}

    def test_flatten_dict_empty(self):
        """Test flatten_dict with empty dict."""
        msg = AgentMessage(query="test")
        result = msg.flatten_dict({})
        assert result == {}

    def test_flatten_dict_with_custom_separator(self):
        """Test flatten_dict with custom separator."""
        msg = AgentMessage(query="test")
        nested = {"a": {"b": 1}}
        result = msg.flatten_dict(nested, sep="-")
        assert result == {"a-b": 1}

    def test_flatten_dict_no_parent_key(self):
        """Test flatten_dict without parent_key."""
        msg = AgentMessage(query="test")
        nested = {"a": 1, "b": 2}
        result = msg.flatten_dict(nested)
        assert result == {"a": 1, "b": 2}

    def test_to_dict_without_context(self):
        """Test to_dict without context."""
        msg = AgentMessage(query="test query", origin="test_agent")
        result = msg.to_dict()
        assert result["query"] == "test query"
        assert result["origin"] == "test_agent"
        assert "context" not in result

    def test_to_dict_with_context(self):
        """Test to_dict with context."""
        msg = AgentMessage(
            query="test query",
            context={"a": 1, "b": {"c": 2}},
        )
        result = msg.to_dict()
        assert result["query"] == "test query"
        assert result["context_a"] == 1
        assert result["context_b_c"] == 2

    def test_to_dict_excludes_none(self):
        """Test to_dict excludes None values."""
        msg = AgentMessage(query="test query", execution_result=None)
        result = msg.to_dict()
        assert "execution_result" not in result

    def test_dump_json(self):
        """Test dump_json produces valid JSON string."""
        msg = AgentMessage(query="test query", origin="test_agent")
        json_str = msg.dump_json()
        parsed = json.loads(json_str)
        assert parsed["query"] == "test query"
        assert parsed["origin"] == "test_agent"

    def test_dump_json_with_context(self):
        """Test dump_json with context."""
        msg = AgentMessage(
            query="test query",
            context={"key": "value"},
        )
        json_str = msg.dump_json()
        parsed = json.loads(json_str)
        assert parsed["context_key"] == "value"

    def test_model_copy(self):
        """Test AgentMessage model_copy."""
        msg = AgentMessage(query="test query", responses=[("a", "b")])
        copied = msg.model_copy(deep=True)
        assert copied.query == msg.query
        assert copied.responses == msg.responses
        assert copied.responses is not msg.responses  # deep copy


class TestFormatKwargs:
    """Tests for AgentMessage.format_kwargs - template-safe serialization."""

    def test_excludes_query_media(self):
        """Test that query_media never reaches prompt templates (leak guard)."""
        msg = AgentMessage(query="q", query_media=[("image", "https://x/a.png")])
        assert "query_media" not in msg.format_kwargs()

    def test_excludes_media(self):
        """Test that output media never reaches prompt templates."""
        msg = AgentMessage(query="q", media=[("image", "base64data")])
        assert "media" not in msg.format_kwargs()

    def test_keeps_query_and_context(self):
        """Test that query and flattened context are still present."""
        msg = AgentMessage(query="q", context={"a": 1, "b": {"c": 2}})
        kwargs = msg.format_kwargs()
        assert kwargs["query"] == "q"
        assert kwargs["context_a"] == 1
        assert kwargs["context_b_c"] == 2

    def test_to_dict_still_serializes_media(self):
        """Test that to_dict (the serialization path) keeps media fields."""
        msg = AgentMessage(
            query="q", query_media=[("image", "u")], media=[("image", "m")]
        )
        d = msg.to_dict()
        assert d["query_media"] == [("image", "u")]
        assert d["media"] == [("image", "m")]

    def test_format_kwargs_template_roundtrip(self):
        """Test that formatting a template with format_kwargs cannot leak reprs."""
        msg = AgentMessage(query="what is this?", query_media=[("image", "/tmp/a.png")])
        rendered = "{query}".format(**msg.format_kwargs())
        assert rendered == "what is this?"


class TestBaseChain:
    """Tests for BaseChain abstract class."""

    def test_cannot_instantiate_abstract(self):
        """Test that BaseChain cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseChain()


class MockLLMChain(BaseLLMChain):
    """Mock implementation of BaseLLMChain for testing."""

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message.responses.append(("mock_chain", "mock response"))
        message.execution_result = "success"
        return message


class TestBaseLLMChain:
    """Tests for BaseLLMChain abstract class."""

    def test_cannot_instantiate_abstract(self):
        """Test that BaseLLMChain cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMChain()

    def test_construction_with_name(self):
        """Test BaseLLMChain construction with custom name."""
        chain = MockLLMChain(name="custom_chain")
        assert chain.name == "custom_chain"

    def test_construction_default_name(self):
        """Test BaseLLMChain construction with default name."""
        chain = MockLLMChain()
        assert chain.name == "chain"

    def test_invoke(self):
        """Test BaseLLMChain invoke method."""
        chain = MockLLMChain(name="test_chain")
        msg = AgentMessage(query="test")
        result = chain.invoke(msg)
        assert result.execution_result == "success"
        assert len(result.responses) == 1

    def test_ainvoke(self):
        """Test BaseLLMChain ainvoke method (sync version calls invoke)."""
        chain = MockLLMChain(name="test_chain")
        msg = AgentMessage(query="test")
        # ainvoke is async, so we call invoke directly for sync testing
        result = chain.invoke(msg)
        assert result.execution_result == "success"

    def test_call(self):
        """Test BaseLLMChain __call__ method."""
        chain = MockLLMChain(name="test_chain")
        msg = AgentMessage(query="test")
        result = chain(msg)
        assert result.execution_result == "success"

    def test_call_with_kwargs(self):
        """Test BaseLLMChain __call__ with kwargs."""
        chain = MockLLMChain(name="test_chain")
        msg = AgentMessage(query="test")
        result = chain(msg, extra_kwarg="value")
        assert result.execution_result == "success"


class CapabilityMockChain(MockLLMChain):
    """BaseLLMChain stub with a controllable detect_capabilities result."""

    _capabilities = PrivateAttr(default_factory=dict)

    def __init__(self, capabilities=None, **kwargs):
        super().__init__(**kwargs)
        self._capabilities = capabilities or {}

    def detect_capabilities(self):
        return self._capabilities


class TestMediaGate:
    """Tests for BaseLLMChain media_support / _modality_ok / _media_parts."""

    def test_default_media_support_is_empty(self):
        """Test unlisted modalities default to auto-detection."""
        assert MockLLMChain().media_support == {}

    def test_detect_capabilities_default_is_empty(self):
        """Test the base hook reports everything undecidable by default."""
        assert MockLLMChain().detect_capabilities() == {}

    def test_auto_undetectable_assumes_capable(self):
        """Test fail-loud policy: unknown modality keeps media."""
        assert CapabilityMockChain()._modality_ok("image") is True

    @pytest.mark.parametrize(
        "capabilities,expected",
        [({"image": True}, True), ({"image": False}, False), ({}, True)],
    )
    def test_auto_matrix(self, capabilities, expected):
        """Test auto mode maps per-modality detection results to the gate."""
        assert (
            CapabilityMockChain(capabilities=capabilities)._modality_ok("image")
            is expected
        )

    def test_enabled_overrides_negative_detection(self):
        """Test enabled forces a modality's gate open even when detection says no."""
        chain = CapabilityMockChain(
            capabilities={"image": False}, media_support={"image": "enabled"}
        )
        assert chain._modality_ok("image") is True

    def test_disabled_overrides_positive_detection(self):
        """Test disabled forces a modality's gate closed even when detection says yes."""
        chain = CapabilityMockChain(
            capabilities={"image": True}, media_support={"image": "disabled"}
        )
        assert chain._modality_ok("image") is False

    def test_gates_are_independent_per_modality(self):
        """Test a model supporting a subset: one modality off must not affect another."""
        chain = CapabilityMockChain(
            capabilities={"image": True, "audio": False},
            media_support={"video": "disabled"},
        )
        assert chain._modality_ok("image") is True
        assert chain._modality_ok("audio") is False  # detected incapable
        assert chain._modality_ok("video") is False  # policy disabled
        assert chain._modality_ok("document") is True  # unknown -> auto

    def test_media_parts_resolves_url_images(self):
        """Test images pass through the open gate as MediaRefs."""
        msg = AgentMessage(query="q", query_media=[("image", "https://x/a.png")])
        parts = CapabilityMockChain(capabilities={"image": True})._media_parts(
            msg, "image"
        )
        assert parts == [
            {
                "content_type": "image",
                "kind": "url",
                "value": "https://x/a.png",
                "mime_type": "image/png",
            }
        ]

    def test_media_parts_empty_without_media(self):
        """Test text-only messages yield no parts for any modality."""
        assert (
            CapabilityMockChain()._media_parts(AgentMessage(query="q"), "image") == []
        )

    def test_media_parts_filters_other_modalities(self):
        """Test only the requested modality is extracted from query_media."""
        msg = AgentMessage(
            query="q", query_media=[("text", "hi"), ("audio", "https://x/a.mp3")]
        )
        assert CapabilityMockChain()._media_parts(msg, "image") == []
        audio = CapabilityMockChain()._media_parts(msg, "audio")
        assert audio[0]["content_type"] == "audio"
        assert audio[0]["mime_type"] == "audio/mpeg"

    def test_media_parts_drops_media_when_gate_closed(self, caplog):
        """Test media are dropped with a modality-specific warning when incapable."""
        msg = AgentMessage(query="q", query_media=[("image", "https://x/a.png")])
        chain = CapabilityMockChain(capabilities={"image": False})
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            assert chain._media_parts(msg, "image") == []
        assert "does not support image input" in caplog.text

    def test_media_parts_local_path_resolved_to_base64(self, tmp_path):
        """Test local image paths are resolved through the gate."""
        f = tmp_path / "a.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        msg = AgentMessage(query="q", query_media=[("image", str(f))])
        parts = CapabilityMockChain()._media_parts(msg, "image")
        assert parts[0]["kind"] == "base64"
        assert parts[0]["mime_type"] == "image/png"

    def test_image_parts_delegates_to_media_parts(self):
        """Test the image sugar matches the generic funnel exactly."""
        msg = AgentMessage(
            query="q",
            query_media=[("image", "https://x/a.png"), ("audio", "https://x/b.mp3")],
        )
        chain = CapabilityMockChain(capabilities={"image": True})
        assert chain._image_parts(msg) == chain._media_parts(msg, "image")
