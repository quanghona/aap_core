"""Tests for aap_core.utils module."""

import base64

import pytest
from aap_core.utils import remove_thinking, resolve_media


class TestRemoveThinking:
    """Tests for the remove_thinking utility function."""

    def test_remove_thinking_tags(self):
        """Test removing thinking tags from a response."""
        response = "<think>Let me think about this.</think>Hello world"
        result = remove_thinking(response)
        assert result == "Hello world"

    def test_remove_thinking_no_tags(self):
        """Test response without thinking tags is unchanged."""
        response = "Hello world"
        result = remove_thinking(response)
        assert result == "Hello world"

    def test_remove_thinking_missing_opening_tag_qwen(self):
        """Test Qwen model case: has </think> but no <think> opening."""
        response = "<think>Let me think.</think>Hello world"
        result = remove_thinking(response)
        assert "<think>" not in result
        assert "Hello world" in result

    def test_remove_thinking_empty_response(self):
        """Test with empty string."""
        result = remove_thinking("")
        assert result == ""

    def test_remove_thinking_multiline_thinking(self):
        """Test removing thinking with multiline content."""
        response = "<think>Line 1\nLine 2\nLine 3</think>Final answer"
        result = remove_thinking(response)
        assert "Line 1" not in result
        assert "Line 2" not in result
        assert "Final answer" in result

    def test_remove_thinking_only_closing_tag(self):
        """Test with only closing tag present."""
        response = "</think>Hello"
        result = remove_thinking(response)
        assert result == "Hello"

    def test_remove_thinking_only_opening_tag(self):
        """Test with only opening tag present (no closing)."""
        response = "<think>Hello"
        result = remove_thinking(response)
        # Should not crash, returns as-is since no closing tag
        assert "<think>" in result

    def test_remove_thinking_multiple_thinking_blocks(self):
        """Test with multiple thinking blocks (only first is removed by DOTALL)."""
        response = "<think>First</think>Middle<think>Second</think>End"
        result = remove_thinking(response)
        # DOTALL removes from first <think> to last </think>
        assert "First" not in result
        assert "Second" not in result
        assert "End" in result


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


class TestResolveMedia:
    """Tests for the resolve_media utility function."""

    def test_resolve_none_and_empty(self):
        """Test that None and empty inputs produce an empty list."""
        assert resolve_media(None) == []
        assert resolve_media([]) == []

    def test_resolve_http_url(self):
        """Test that http(s) URLs pass through with kind=url."""
        result = resolve_media([("image", "https://example.com/cat.png")])
        assert result == [
            {
                "content_type": "image",
                "kind": "url",
                "value": "https://example.com/cat.png",
                "mime_type": "image/png",
            }
        ]

    def test_resolve_url_strips_query_string_for_mime_guess(self):
        """Test that URL query strings do not break MIME guessing."""
        result = resolve_media([("image", "http://example.com/cat.jpeg?v=2")])
        assert result[0]["mime_type"] == "image/jpeg"

    def test_resolve_url_unknown_extension_fallback(self):
        """Test that an unguessable URL falls back to application/octet-stream."""
        result = resolve_media([("image", "https://example.com/media/abc123")])
        assert result[0]["mime_type"] == "application/octet-stream"

    def test_resolve_data_uri(self):
        """Test that data: URIs are split into base64 kind with header MIME."""
        result = resolve_media([("image", "data:image/png;base64,AAAA")])
        assert result == [
            {
                "content_type": "image",
                "kind": "base64",
                "value": "AAAA",
                "mime_type": "image/png",
            }
        ]

    def test_resolve_local_path_sniffs_magic_bytes(self, tmp_path):
        """Test that local files are read and sniffed by magic bytes."""
        f = tmp_path / "mystery"  # no extension; only magic bytes identify it
        f.write_bytes(PNG_MAGIC + b"\x00" * 16)
        result = resolve_media([("image", str(f))])
        assert result[0]["kind"] == "base64"
        assert result[0]["mime_type"] == "image/png"
        assert base64.b64decode(result[0]["value"]) == f.read_bytes()

    def test_resolve_local_path_jpeg(self, tmp_path):
        """Test JPEG magic-byte detection."""
        f = tmp_path / "photo.dat"
        f.write_bytes(JPEG_MAGIC + b"\x00" * 32)
        result = resolve_media([("image", str(f))])
        assert result[0]["mime_type"] == "image/jpeg"

    def test_resolve_local_path_webp(self, tmp_path):
        """Test WebP detection via RIFF....WEBP container."""
        f = tmp_path / "anim.webp"
        f.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
        result = resolve_media([("image", str(f))])
        assert result[0]["mime_type"] == "image/webp"

    def test_resolve_local_path_falls_back_to_extension(self, tmp_path):
        """Test that unknown magic bytes fall back to the file extension."""
        f = tmp_path / "notes.txt"
        f.write_bytes(b"hello world")
        result = resolve_media([("document", str(f))])
        assert result[0]["mime_type"] == "text/plain"

    def test_resolve_file_uri_prefix(self, tmp_path):
        """Test that file:// URIs are treated as local paths."""
        f = tmp_path / "img.png"
        f.write_bytes(PNG_MAGIC)
        result = resolve_media([("image", f.as_uri())])
        assert result[0]["mime_type"] == "image/png"

    def test_resolve_missing_path_raises(self, tmp_path):
        """Test that a nonexistent local path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            resolve_media([("image", str(tmp_path / "nope.png"))])

    def test_resolve_preserves_order(self, tmp_path):
        """Test that resolution keeps the original media order."""
        f = tmp_path / "a.png"
        f.write_bytes(PNG_MAGIC)
        result = resolve_media(
            [("image", "https://example.com/b.png"), ("image", str(f))]
        )
        assert [r["kind"] for r in result] == ["url", "base64"]

    def test_resolve_multiple_media(self, tmp_path):
        """Test mixed input kinds in one call."""
        f = tmp_path / "c.png"
        f.write_bytes(PNG_MAGIC)
        result = resolve_media(
            [
                ("image", str(f)),
                ("image", "data:image/gif;base64,R0lGOD"),
                ("image", "http://example.com/d.bmp"),
            ]
        )
        assert len(result) == 3
        assert result[1] == {
            "content_type": "image",
            "kind": "base64",
            "value": "R0lGOD",
            "mime_type": "image/gif",
        }
        assert result[2]["kind"] == "url"

    def test_resolve_tags_content_type(self):
        """Test that every ref carries its own modality, ready for native-part mapping."""
        result = resolve_media(
            [
                ("image", "https://example.com/a.png"),
                ("audio", "https://example.com/b.mp3"),
                ("video", "data:video/mp4;base64,AAAA"),
                ("document", "https://example.com/c.pdf"),
            ]
        )
        assert [r["content_type"] for r in result] == [
            "image",
            "audio",
            "video",
            "document",
        ]
        assert result[1]["mime_type"] == "audio/mpeg"
        assert result[2] == {
            "content_type": "video",
            "kind": "base64",
            "value": "AAAA",
            "mime_type": "video/mp4",
        }
