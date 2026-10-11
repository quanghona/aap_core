"""Tests for aap_core.utils module."""

import base64

import pytest
from aap_core.utils import (
    SUPPORTED_MIME_TYPES,
    _EXTENSION_MIME_TYPES,
    _MAGIC_SIGNATURES,
    is_supported_mime,
    normalize_mime_type,
    remove_thinking,
    resolve_media,
)


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


def _ogg(payload: bytes) -> bytes:
    """Build an Ogg page whose first packet starts at offset 28.

    The layout is the point of the helper: a 27-byte page header (capture pattern,
    version, type, granule, serial, sequence, CRC, segment count) plus a
    one-entry segment table, which is what puts the codec identification string at
    offset 28 where the probe reads it. A fixture that merely happened to place
    the tag at whatever offset the code indexes would prove nothing.
    """
    header = b"OggS" + b"\x00" * 22 + b"\x01"  # 27 bytes, one segment
    return header + b"\x00" + payload


AUDIO_SAMPLES = {
    "audio/wav": b"RIFF\x00\x00\x00\x00WAVEfmt ",
    "audio/mpeg": b"\xff\xfb\x90\x64" + b"\x00" * 20,
    "audio/mp4": b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00",
    "audio/aac": b"\xff\xf1P\x80" + b"\x00" * 20,
    "audio/flac": b'fLaC\x00\x00\x00"' + b"\x00" * 20,
    "audio/ogg": _ogg(b"\x01vorbis") + b"\x00" * 20,
    "audio/opus": _ogg(b"OpusHead\x01\x02") + b"\x00" * 20,
    "audio/aiff": b'FORM\x00\x00\x00"AIFF' + b"\x00" * 8,
    "audio/amr": b"#!AMR\n" + b"\x00" * 8,
    "audio/webm": b"\x1a\x45\xdf\xa3" + b"\x00" * 13 + b"webm",
    "audio/x-ms-wma": b"\x30\x26\xb2\x75" + b"\x00" * 20,
}

# Formats the pruned tables no longer carry an entry for. They still reach
# resolve_media through stdlib mimetypes or a container probe, which is why the
# unsupported-format warning has to key off SUPPORTED_MIME_TYPES rather than off
# the tables.
UNSUPPORTED_AUDIO = ["audio/basic", "audio/speex", "audio/amr-wb", "audio/x-caf"]


class TestAudioResolution:
    """Tests for audio format resolution through resolve_media."""

    @pytest.mark.parametrize("expected", sorted(AUDIO_SAMPLES))
    def test_audio_container_detected_from_bytes(self, tmp_path, expected):
        """Test that each deliverable audio container is identified from its bytes."""
        f = tmp_path / "no-extension"
        f.write_bytes(AUDIO_SAMPLES[expected])
        result = resolve_media([("audio", str(f))])
        assert result[0]["mime_type"] == expected
        assert result[0]["kind"] == "base64"

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            (".wav", "audio/wav"),  # stdlib on this host: audio/x-wav
            (".amr", "audio/amr"),  # stdlib: audio/AMR, invalid casing
            (".aiff", "audio/aiff"),  # stdlib: audio/x-aiff
            (".opus", "audio/opus"),  # stdlib: audio/ogg, which drops the codec
            (".m4a", "audio/mp4"),
            (".mp3", "audio/mpeg"),
            (".wma", "audio/x-ms-wma"),
        ],
    )
    def test_audio_extension_beats_host_mime_database(self, suffix, expected):
        """Test that extensions resolve identically whatever the host MIME database says.

        These are the cases where stdlib returns a string a provider rejects, so
        the pinning is the point of the test rather than the lookup itself.
        """
        result = resolve_media([("audio", f"https://example.com/clip{suffix}")])
        assert result[0]["mime_type"] == expected

    def test_id3_tagged_flac_is_not_reported_as_mpeg(self, tmp_path):
        """Test that an ID3 header prepended to a FLAC stream does not hide it.

        A prefix-only reader sees "ID3" and answers audio/mpeg for audio that is
        really FLAC, putting the wrong codec in front of the model.
        """
        tag = b"ID3\x04\x00\x00\x00\x00\x00\x00"  # empty tag, synchsafe size 0
        f = tmp_path / "tagged"
        f.write_bytes(tag + b'fLaC\x00\x00\x00"' + b"\x00" * 20)
        result = resolve_media([("audio", str(f))])
        assert result[0]["mime_type"] == "audio/flac"

    def test_amr_wideband_is_not_misreported_as_narrowband(self, tmp_path):
        """Test that an AMR-WB stream is not read as the narrowband type.

        Wideband is no longer identified - it is not in the support set - but it
        must not be mistaken for audio/amr either, since that would deliver
        undecodable bytes under a label the provider trusts. The signature keeps
        its trailing newline for this reason.
        """
        f = tmp_path / "voice"
        f.write_bytes(b"#!AMR-WB\n" + b"\x00" * 8)
        result = resolve_media([("audio", str(f))])
        assert result[0]["mime_type"] != "audio/amr"

    def test_ogg_speex_is_not_labelled_as_supported_ogg(self, tmp_path):
        """Test that a Speex payload is not reported with the Vorbis container type.

        This is the one probe branch kept for an unsupported format: answering
        audio/ogg here would claim a supported container for bytes no provider can
        decode, which is worse than naming the format and warning.
        """
        f = tmp_path / "voice"
        f.write_bytes(_ogg(b"Speex   ") + b"\x00" * 20)
        result = resolve_media([("audio", str(f))])
        assert result[0]["mime_type"] == "audio/speex"

    @pytest.mark.parametrize(
        "declared,expected",
        [("audio", "audio/mp4"), ("video", "video/mp4")],
    )
    def test_mp4_follows_the_declared_modality(self, declared, expected):
        """Test that one container resolves to the modality the caller asked for."""
        result = resolve_media([(declared, "https://example.com/clip.mp4")])
        assert result[0]["mime_type"] == expected

    def test_audio_bytes_declared_as_video_warn(self, tmp_path, caplog):
        """Test that a modality contradiction is reported rather than shipped."""
        f = tmp_path / "clip.wav"
        f.write_bytes(AUDIO_SAMPLES["audio/wav"])
        with caplog.at_level("WARNING"):
            resolve_media([("video", str(f))])
        assert "which is audio content" in caplog.text

    @pytest.mark.parametrize("mime", sorted(UNSUPPORTED_AUDIO))
    def test_unsupported_audio_format_warns_but_is_returned(self, mime, caplog):
        """Test that a named format no provider accepts is loud, not dropped."""
        with caplog.at_level("WARNING"):
            result = resolve_media([("audio", f"data:{mime};base64,AAAA")])
        assert result[0]["mime_type"] == mime
        assert "not known to be accepted by any LLM provider" in caplog.text

    def test_unrecognized_audio_degrades_to_octet_stream_quietly(
        self, tmp_path, caplog
    ):
        """Test the cost of pruning the tables: an unidentified file is not diagnosed.

        A .caf file used to resolve to audio/x-caf and warn. Now nothing identifies
        it, so it lands on application/octet-stream with no warning at all. The
        request fails either way; this records that the diagnosis got cheaper, so
        the trade is visible rather than silent.
        """
        f = tmp_path / "clip.caf"
        f.write_bytes(b"caff" + b"\x00" * 20)
        with caplog.at_level("WARNING"):
            result = resolve_media([("audio", str(f))])
        assert result[0]["mime_type"] == "application/octet-stream"
        assert caplog.text == ""

    @pytest.mark.parametrize("mime", sorted(SUPPORTED_MIME_TYPES["audio"]))
    def test_supported_audio_does_not_warn(self, mime, caplog):
        """Test that the advertised set resolves quietly, which keeps the warning useful."""
        with caplog.at_level("WARNING"):
            resolve_media([("audio", f"data:{mime};base64,AAAA")])
        assert caplog.text == ""

    def test_undetectable_audio_does_not_warn(self, caplog):
        """Test that an opaque URL warns about nothing; no type is not a defect."""
        with caplog.at_level("WARNING"):
            result = resolve_media([("audio", "https://example.com/stream/abc123")])
        assert result[0]["mime_type"] == "application/octet-stream"
        assert caplog.text == ""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("audio/x-wav", "audio/wav"),
            ("audio/AMR", "audio/amr"),
            ("audio/vnd.wave", "audio/wav"),
            ("audio/mp4a-latm", "audio/mp4"),
            ("audio/mp3", "audio/mpeg"),
            ("audio/M4A", "audio/mp4"),
            ("audio/X-MS-WMA", "audio/x-ms-wma"),
        ],
    )
    def test_audio_mime_aliases_fold_onto_canonical(self, raw, expected):
        """Test that vendor spellings reach the wire as one canonical string."""
        assert normalize_mime_type(raw) == expected


class TestMimeTableInvariants:
    """Structural checks holding the format tables and the support claim together."""

    def test_tables_carry_only_deliverable_audio(self):
        """Test that no audio format outside the support set is known to the tables.

        Identification and delivery are one statement here: an entry for a format
        no provider takes is knowledge the framework carries but never uses, and
        the reason it looks like support when it is not.
        """
        supported = SUPPORTED_MIME_TYPES["audio"]
        for mime in _EXTENSION_MIME_TYPES.values():
            if mime.startswith("audio/"):
                assert mime in supported, mime
        for _, mime in _MAGIC_SIGNATURES:
            if mime.startswith("audio/"):
                assert mime in supported, mime

    def test_every_supported_audio_type_is_reachable(self):
        """Test that each advertised audio format has a byte path producing it.

        Without this, SUPPORTED_MIME_TYPES could gain an entry no resolver can ever
        emit, and the mismatch surfaces later as an unexplained octet-stream.
        """
        assert set(AUDIO_SAMPLES) == SUPPORTED_MIME_TYPES["audio"]

    @pytest.mark.parametrize("mime", sorted(SUPPORTED_MIME_TYPES["audio"]))
    def test_supported_audio_passes_the_preflight_check(self, mime):
        """Test the preflight integrations call reports each advertised format."""
        assert is_supported_mime(mime, "audio") is True

    @pytest.mark.parametrize("mime", sorted(UNSUPPORTED_AUDIO))
    def test_unsupported_audio_fails_the_preflight_check(self, mime):
        """Test that being nameable is not the same as being deliverable."""
        assert is_supported_mime(mime, "audio") is False
        assert is_supported_mime(mime) is False
