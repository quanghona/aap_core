from __future__ import annotations

import base64
import logging
import mimetypes
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

from a2a.types import AgentCapabilities

if TYPE_CHECKING:
    from .types import ContentType, MediaReference

logger = logging.getLogger(__name__)

# Format knowledge is split into three ordered layers, and the order matters:
#
#   1. magic-byte sniffing - authoritative for local files, where the payload is
#      in hand and the bytes cannot lie about the container.
#   2. an extension table, consulted BEFORE stdlib mimetypes.
#   3. stdlib mimetypes as a fallback for whatever layer 2 does not pin.
#
# Layer 2 exists because stdlib merges /etc/mime.types and its answers therefore
# vary by host and Python version. Measured on this box (CPython 3.10): ".webp"
# resolves to None (image/webp only landed in 3.13), ".opus" to audio/ogg,
# ".wav" to audio/x-wav, ".amr" to audio/AMR and ".aiff" to audio/x-aiff. Those
# strings are what reach a provider's Content-Type header, and a request carrying
# audio/x-wav or audio/AMR is rejected there rather than here - a long way from
# the file that caused it. Pinning the cases that matter makes resolution
# deterministic across machines instead of dependent on the host MIME database.
#
# Normalization (below) then folds vendor aliases onto the canonical type and
# reconciles dual-modality containers with the modality the caller declared, which
# is the step that decides what string goes on the wire.

_MAGIC_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    # Images.
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    # Audio. Only formats in SUPPORTED_MIME_TYPES appear here: a signature that
    # identifies something undeliverable buys a nicer log line and nothing else,
    # so that knowledge is not carried. Two consequences are worth knowing: an
    # extensionless Sun/AMR-WB file now resolves to application/octet-stream, and
    # the "#!AMR\n" signature below carries its newline precisely so that an
    # AMR-WB stream ("#!AMR-WB\n") does not match it and report narrowband.
    (b"fLaC", "audio/flac"),
    (b"ID3", "audio/mpeg"),  # ID3v2 tag; refined by _probe_id3_payload
    (b"#!AMR\n", "audio/amr"),
    # Video. Same rule as audio: only formats in SUPPORTED_MIME_TYPES appear here.
    # ASF is deliberately absent - the container carries audio and video alike, so
    # naming it audio here made a .wmv resolve to audio/x-ms-wma and report a
    # modality contradiction on bytes that match what the caller declared. It is
    # resolved by _probe_asf, which consults the declared modality like the other
    # dual-modality probes.
    (b"FLV\x01", "video/x-flv"),
    (b"\x00\x00\x01\xba", "video/mpeg"),  # MPEG program-stream pack header
)

# Leading bytes of the ASF header object GUID, shared by WMV and WMA.
_ASF_GUID = b"\x30\x26\xb2\x75"

# Four-byte codes at a fixed offset that identify a container whose magic is not
# a leading byte run. Keyed by the leading run to test, then a form/brand check.
_RIFF_FORMS: Dict[bytes, str] = {
    b"WEBP": "image/webp",
    b"WAVE": "audio/wav",
    b"AVI ": "video/avi",
}
_IFF_FORMS: Dict[bytes, str] = {
    b"AIFF": "audio/aiff",
    b"AIFC": "audio/aiff",
}
_AUDIO_BRANDS: Tuple[bytes, ...] = (b"M4A ", b"M4B ", b"M4P ", b"M4SV")
# QuickTime and 3GPP are ISO base media files too, but they are not MP4: shipping
# a .mov as video/mp4 puts the wrong container in front of a provider that checks.
_QUICKTIME_BRANDS: Tuple[bytes, ...] = (b"qt  ",)
_THREEGP_BRANDS: Tuple[bytes, ...] = (b"3gp4", b"3gp5", b"3gp6", b"3gp7", b"3gg6")


def _probe_riff(data: bytes) -> str | None:
    """Resolve a RIFF container by its four-byte form at offset 8."""
    return _RIFF_FORMS.get(data[8:12])


def _probe_iff(data: bytes) -> str | None:
    """Resolve a FORM (AIFF/AIFC) container by its four-byte form at offset 8."""
    return _IFF_FORMS.get(data[8:12]) if data[:4] == b"FORM" else None


def _probe_ogg(data: bytes) -> str | None:
    """Resolve an Ogg stream by the codec identification header of its first packet.

    The container alone is not informative: Ogg carries Opus, FLAC, Vorbis and
    Speex payloads, and "audio/ogg" for an Opus stream loses the codec the
    provider needs. Payload ids sit at offset 28 once the capture pattern and
    page header are skipped.
    """
    if data[28:36].startswith(b"OpusHead"):
        return "audio/opus"
    if data[28:33] == b"\x7fFLAC":
        return "audio/flac"
    # Speex is not in the support set, but this branch stays: dropping it falls
    # through to the Vorbis default and labels an undeliverable payload
    # audio/ogg, which is a false claim of support rather than a missing one.
    if data[28:33] == b"Speex":
        return "audio/speex"
    return "audio/ogg"  # Vorbis ("\x01vorbis") and any other Ogg payload


def _probe_iso_bmff(data: bytes, content_type: ContentType | None) -> str | None:
    """Resolve an ISO base media file (MP4/M4A/QuickTime/3GP) by its ftyp brand.

    The same container family carries audio and video, and a bare ".m4a"/".mp4"
    file is byte-for-byte indistinguishable in its header from a video file
    beyond the brand field. What an API must be told depends on how the caller
    declared the media, so the declared modality wins when it is known - except
    where the brand names a different container outright, since QuickTime and
    3GPP are not MP4 whatever the caller called them.
    """
    if data[4:8] != b"ftyp":
        return None
    brand = data[8:12]
    if brand in _QUICKTIME_BRANDS:
        return "video/quicktime"
    if brand in _THREEGP_BRANDS:
        return "video/3gpp"
    if content_type == "audio":
        return "audio/mp4"
    if content_type == "video":
        return "video/mp4"
    return "audio/mp4" if brand in _AUDIO_BRANDS else "video/mp4"


def _probe_asf(data: bytes, content_type: ContentType | None) -> str | None:
    """Resolve an ASF stream (WMV/WMA) by the modality the caller declared.

    One header GUID covers both, and separating them means parsing the stream
    properties for a codec URL - machinery the delivery contract does not need,
    because what a provider must be told depends on how the media was declared,
    exactly as for MP4 and Matroska. Undeclared falls back to audio, which is
    the reading this table had before the video modality was wired.
    """
    if not data.startswith(_ASF_GUID):
        return None
    if content_type == "video":
        return "video/x-ms-wmv"
    return "audio/x-ms-wma"


def _probe_ebml(data: bytes, content_type: ContentType | None) -> str | None:
    """Resolve a Matroska/WebM container by its DocType element.

    DocType sits at an offset that depends on the preceding element sizes, so the
    head is searched rather than indexed. WebM is the common case for audio-only
    Opus delivery; the declared modality decides the top-level type, as for MP4.
    """
    is_webm = b"webm" in data[4:32]
    family = "webm" if is_webm else "x-matroska"
    if content_type in ("audio", "video"):
        return f"{content_type}/{family}"
    return f"video/{family}"


def _probe_mpeg_sync(data: bytes) -> str | None:
    """Distinguish an MPEG audio frame from an ADTS (AAC) frame by its header bits.

    Both open with the 12-bit syncword 0xFFFx, so the leading bytes alone are not
    decisive. Bit 2 of the second byte is the layer-low bit for MPEG audio and the
    always-zero bit for ADTS, and the two predicates below are mutually exclusive:
    ADTS headers observed in the wild are 0xF0/0xF1/0xF8/0xF9, MPEG audio frames
    always carry a non-zero layer field.
    """
    if len(data) < 2 or data[0] != 0xFF or (data[1] & 0xE0) != 0xE0:
        return None
    if (data[1] & 0xF6) == 0xF0:
        return "audio/aac"
    if (data[1] & 0x06) != 0:
        return "audio/mpeg"
    return None


def _probe_id3_payload(data: bytes) -> str | None:
    """Follow an ID3v2 tag to the payload it wraps.

    Tagged MP3 files are extremely common, and so are tagged FLAC files: an ID3
    header prepended to a FLAC stream hides the "fLaC" magic from a naive prefix
    test, which would report audio/mpeg for audio that is really FLAC. The tag
    size is a 28-bit synchsafe integer at offset 6, so the payload offset is
    computed rather than guessed.
    """
    if len(data) < 10:
        return None
    size = (
        (data[6] & 0x7F) << 21
        | (data[7] & 0x7F) << 14
        | (data[8] & 0x7F) << 7
        | (data[9] & 0x7F)
    )
    payload = data[10 + size : 10 + size + 12]
    return _sniff_mime_type(payload) if payload else None


def _sniff_mime_type(
    data: bytes, content_type: ContentType | None = None
) -> str | None:
    """Detect MIME type from magic bytes, probing containers where needed.

    Args:
        data: Leading bytes of the media payload; only the first ~32 bytes are
            inspected, so passing a whole file is safe but unnecessary.
        content_type: The modality the caller declared, used only to break ties
            for containers that carry both audio and video (MP4, Matroska).

    Returns:
        str | None: The sniffed MIME type, or None when the bytes are unrecognized.
    """
    for signature, mime_type in _MAGIC_SIGNATURES:
        if data.startswith(signature):
            if signature == b"ID3":
                return _probe_id3_payload(data) or mime_type
            return mime_type
    if data.startswith(_ASF_GUID):
        return _probe_asf(data, content_type)
    if data.startswith(b"RIFF"):
        return _probe_riff(data)
    if data.startswith(b"FORM"):
        return _probe_iff(data)
    if data.startswith(b"OggS"):
        return _probe_ogg(data)
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return _probe_ebml(data, content_type)
    if data.startswith(b"\x00\x00\x00") and data[4:8] in (b"ftyp", b"moov", b"mdat"):
        return _probe_iso_bmff(data, content_type)
    return _probe_mpeg_sync(data)


# Extension table consulted before stdlib mimetypes. Values are the canonical
# type, never the vendor alias stdlib prefers (it reports audio/x-wav for .wav
# and audio/AMR for .amr on this platform). Dual-modality containers (.mp4,
# .webm, .mkv) are deliberately absent: their MIME depends on the declared
# modality, so they are resolved by sniffing or by normalization instead.
_EXTENSION_MIME_TYPES: Dict[str, str] = {
    # Images.
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",  # missing from stdlib before 3.13
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".avif": "image/avif",
    ".svg": "image/svg+xml",
    ".ico": "image/vnd.microsoft.icon",
    # Audio. Entries exist only for formats in SUPPORTED_MIME_TYPES. The ones
    # that rule removed (.au, .snd, .awb, .ac3, .dts, .caf, .mka, .3gp, .3g2,
    # .mid) now fall through to stdlib mimetypes, which still names most of them
    # (audio/basic, audio/AMR-WB, audio/ac3) so the unsupported-format warning
    # fires anyway; for .caf, .mka and .midi stdlib knows nothing and the result
    # is application/octet-stream.
    ".mp3": "audio/mpeg",
    ".mp2": "audio/mpeg",
    ".mp1": "audio/mpeg",
    ".wav": "audio/wav",  # stdlib: audio/x-wav
    ".wave": "audio/wav",  # stdlib: None
    ".bwf": "audio/wav",  # stdlib: None
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",  # stdlib: audio/ogg, which drops the codec
    ".m4a": "audio/mp4",
    ".m4b": "audio/mp4",  # stdlib: None
    ".aac": "audio/aac",
    ".adts": "audio/aac",
    ".amr": "audio/amr",  # stdlib: audio/AMR, invalid casing per RFC 2045
    ".aif": "audio/aiff",  # stdlib: audio/x-aiff
    ".aiff": "audio/aiff",
    ".aifc": "audio/aiff",
    ".wma": "audio/x-ms-wma",
    # Video. Entries exist only for formats in SUPPORTED_MIME_TYPES, and only where
    # stdlib is wrong or the container is ambiguous. .mp4 and .webm are absent as
    # dual-modality containers, resolved by sniffing or by
    # _MODALITY_CONTAINER_TYPES instead. .mkv is absent because Matroska has no
    # provider witness in the survey; stdlib still names it, so the
    # unsupported-format warning fires rather than a silent claim of support.
    # .ts is deliberately NOT pinned: this host resolves it to
    # text/vnd.trolltech.linguist, and any pin would have to choose between
    # MPEG transport streams and TypeScript files with the same suffix.
    ".avi": "video/avi",  # stdlib: video/x-msvideo, not the registered type
    ".mov": "video/quicktime",
    ".qt": "video/quicktime",
    ".mpg": "video/mpeg",
    # .mpeg names a layer of the standard, not a stream kind: it is used for MP3
    # audio and for MPEG-1/2 video alike. Pinned to the video type and reconciled
    # to audio by declaration (below) so that both readings are deterministic on
    # any host. .mp1/.mp2/.mp3 stay audio-only pins - those suffixes are not
    # ambiguous, and folding them to video would ship a mislabelled MP3 quietly
    # instead of warning.
    ".mpeg": "video/mpeg",
    ".m4v": "video/mp4",
    ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv",
}

# Vendor and legacy spellings folded onto the canonical type after lookup. The
# keys must be lowercase; normalization lowercases before matching.
_MIME_ALIASES: Dict[str, str] = {
    "audio/x-wav": "audio/wav",
    "audio/vnd.wave": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/x-pn-wav": "audio/wav",
    "audio/x-m4a": "audio/mp4",
    "audio/mp4a-latm": "audio/mp4",
    "audio/x-aiff": "audio/aiff",
    "audio/x-aifc": "audio/aiff",
    "audio/aifc": "audio/aiff",
    "audio/x-aac": "audio/aac",
    "audio/aac-adts": "audio/aac",
    "audio/x-flac": "audio/flac",
    "audio/ogg; codecs=opus": "audio/opus",
    # Gemini names these formats with non-registered spellings; folding them onto
    # the canonical type keeps one string on the wire whatever the source was.
    "audio/mp3": "audio/mpeg",
    "audio/m4a": "audio/mp4",
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
    "image/x-ms-bmp": "image/bmp",
    "image/x-icon": "image/vnd.microsoft.icon",
    "video/x-m4v": "video/mp4",
    # Gemini names several video formats with non-registered spellings ("video/mov",
    # "video/wmv", "video/mpg"); stdlib prefers video/x-msvideo for .avi. Folding
    # them keeps one string on the wire whatever the source was.
    "video/x-msvideo": "video/avi",
    "video/msvideo": "video/avi",
    "video/mov": "video/quicktime",
    "video/mpg": "video/mpeg",
    "video/wmv": "video/x-ms-wmv",
    "video/3gp": "video/3gpp",
}

# Containers whose top-level type depends on what the media is being carried
# *as*, keyed by (guessed_mime, declared_content_type). Reached when only a
# filename is available (URLs), where sniffing is impossible.
_MODALITY_CONTAINER_TYPES: Dict[Tuple[str, str], str] = {
    ("video/mp4", "audio"): "audio/mp4",
    ("audio/mp4", "video"): "video/mp4",
    ("video/webm", "audio"): "audio/webm",
    ("audio/webm", "video"): "video/webm",
    ("video/x-matroska", "audio"): "audio/x-matroska",
    ("audio/x-matroska", "video"): "video/x-matroska",
    ("application/ogg", "audio"): "audio/ogg",
    ("application/ogg", "video"): "video/ogg",
    # 3GPP carries audio-only and video files under one suffix, and stdlib always
    # answers audio/3gpp, so a video declared as such would otherwise be reported
    # as contradicting its own declaration. No ASF pair here: .wma and .wmv are
    # unambiguous spellings, and rewriting one into the other would ship a
    # mislabelled file quietly instead of warning. The bytes case is decided by
    # _probe_asf, where the container genuinely is ambiguous.
    ("audio/3gpp", "video"): "video/3gpp",
    ("video/3gpp", "audio"): "audio/3gpp",
    ("audio/3gpp2", "video"): "video/3gpp2",
    ("video/mpeg", "audio"): "audio/mpeg",
}

# The MIME types this framework will deliver to a model, grouped by the modality
# a provider consumes them as. Membership requires that at least one LLM API
# accept the format as a native content part of that modality. Two rules follow
# from that standard and are not visible in the data:
#
#   - Speech-to-text endpoints are not evidence. An STT upload API accepting
#     audio/mpeg says nothing about whether a chat model accepts an audio content
#     part, and this modality means the latter.
#   - Formats the interface cannot express are excluded even where a provider
#     lists them: audio/l16, audio/alaw and audio/mulaw are accepted by Gemini,
#     but they are headerless PCM whose sample rate is supplied out of band, and a
#     MediaReference carries bytes plus a MIME type only.
#
# The lookup tables above are kept to this set: no magic signature or extension
# entry exists for an audio format that is not listed here, so "the framework
# recognizes it" and "the framework can deliver it" are one statement rather than
# two that a test has to hold apart. Formats outside the set are still named when
# stdlib mimetypes or a container probe reports them (audio/basic, audio/ac3,
# audio/speex), and resolve_media warns on those.
SUPPORTED_MIME_TYPES: Dict[ContentType, frozenset] = {
    "image": frozenset(
        {
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/bmp",
            "image/webp",
            "image/tiff",
            "image/heic",
            "image/heif",
            "image/avif",
        }
    ),
    "audio": frozenset(
        {
            "audio/wav",  # OpenAI input_audio, Gemini, Qwen-Omni, Voxtral
            "audio/mpeg",  # OpenAI ("mp3"), Gemini, Qwen
            "audio/mp4",  # Qwen (m4a/mp4), Gemini
            "audio/aac",  # Gemini, Qwen, Voxtral
            "audio/flac",  # OpenAI Responses, Gemini, Qwen, Voxtral
            "audio/ogg",  # Gemini, Qwen, Voxtral
            "audio/opus",  # Gemini, Qwen, Voxtral
            "audio/aiff",  # Gemini, Qwen
            "audio/amr",  # Qwen
            "audio/webm",  # Gemini, Qwen
            "audio/x-ms-wma",  # Qwen
        }
    ),
    # Every entry below is accepted as a native video content part by at least one
    # surveyed chat API. Gemini is the only provider with a published video format
    # list, so it is the source for all eight entries (its nine spellings fold onto
    # these types - mpg onto mpeg, mov onto quicktime, wmv onto x-ms-wmv); Qwen's
    # DashScope API takes a video_url content part and documents it with an mp4
    # sample but publishes no container list, so nothing rests on Qwen alone. Two
    # exclusions follow from that standard and are worth stating:
    #
    #   - video/x-matroska: was listed here on the strength of a source that turned
    #     out to document no video containers at all. .mkv still resolves to the
    #     type through stdlib, so it is named and warned about, not claimed.
    #   - video/ogg (.ogv) and video/3gpp2: named by stdlib, accepted by nobody
    #     surveyed, so they stay outside the set like the pruned audio formats.
    #
    # Video is delivered as a container, never as frames: frame sampling is a
    # provider-side policy (Gemini samples at 1 FPS, DashScope exposes fps and
    # max_frames, transformers takes num_frames) and MediaReference carries bytes
    # plus a MIME type only, so no per-part sampling option is expressible here.
    "video": frozenset(
        {
            "video/mp4",  # Gemini, Qwen
            "video/mpeg",  # Gemini (mpg/mpeg)
            "video/quicktime",  # Gemini (publishes the alias "video/mov")
            "video/avi",  # Gemini
            "video/webm",  # Gemini
            "video/x-flv",  # Gemini
            "video/x-ms-wmv",  # Gemini (publishes the alias "video/wmv")
            "video/3gpp",  # Gemini
        }
    ),
    # Listed for completeness of the tables. No integration maps these to native
    # parts yet; image, audio and video are mapped in the integration packages,
    # and document is wired through the same gate without format tables.
    "document": frozenset(
        {
            "application/pdf",
            "text/plain",
            "text/markdown",
            "text/html",
            "application/json",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }
    ),
    "text": frozenset({"text/plain"}),
}

# Top-level MIME type to the modality a provider consumes it as. Used to catch
# media whose declared modality contradicts its format, e.g. ("audio", x.png).
_TOP_LEVEL_MODALITIES: Dict[str, ContentType] = {
    "image": "image",
    "audio": "audio",
    "video": "video",
}


def _mime_from_extension(name: str) -> str | None:
    """Guess a MIME type from a filename or URL path, stdlib as last resort."""
    suffix = Path(name).suffix.lower()
    mime_type = _EXTENSION_MIME_TYPES.get(suffix)
    if mime_type:
        return mime_type
    return mimetypes.guess_type(f"x{suffix}")[0] if suffix else None


def normalize_mime_type(
    mime_type: str | None, content_type: ContentType | None = None
) -> str:
    """Fold a MIME type onto the canonical string safe to send to a provider.

    Three steps, in order: lowercase (stdlib yields "audio/AMR", which violates
    RFC 2045 casing and is rejected by strict APIs), alias folding
    ("audio/x-wav" -> "audio/wav"), then container reconciliation, which rewrites
    a dual-modality container to the modality the caller declared. Unrecognized
    input is returned lowercased rather than guessed at, so a wrong answer is
    never invented where an unknown one was.

    Args:
        mime_type: Any type string, including None.
        content_type: Declared modality, used only for container reconciliation.

    Returns:
        str: The canonical type, or "application/octet-stream" when nothing is known.
    """
    if not mime_type:
        return "application/octet-stream"
    lowered = mime_type.strip().lower()
    if not lowered:
        return "application/octet-stream"
    lowered = _MIME_ALIASES.get(lowered, lowered)
    if content_type:
        lowered = _MODALITY_CONTAINER_TYPES.get((lowered, content_type), lowered)
    return lowered


def modality_of_mime(mime_type: str) -> ContentType | None:
    """Report the modality a MIME type is consumed as, or None when undecidable.

    Args:
        mime_type: A canonical MIME type, e.g. "audio/mp4".

    Returns:
        ContentType | None: "image"/"audio"/"video" from the top-level type,
            "document" for the document formats listed in SUPPORTED_MIME_TYPES,
            or None for opaque or ambiguous types such as
            "application/octet-stream".
    """
    lowered = (mime_type or "").strip().lower()
    top_level = lowered.split("/")[0]
    if top_level in _TOP_LEVEL_MODALITIES:
        return _TOP_LEVEL_MODALITIES[top_level]  # type: ignore[return-value]
    if lowered == "application/octet-stream":
        return None
    for modality in ("document", "text"):
        if lowered in SUPPORTED_MIME_TYPES[modality]:
            return modality  # type: ignore[return-value]
    return None


def is_supported_mime(mime_type: str, content_type: ContentType | None = None) -> bool:
    """Report whether a MIME type is in the recognized set for a modality.

    This is the pre-flight check an integration uses to decide whether a resolved
    reference can become a native content part at all, as distinct from
    _modality_ok, which asks whether the model takes the modality. A chain
    targeting one provider will find a type supported here yet unusable there
    (audio/amr reaches Qwen and no other surveyed API), so this answers "is this
    real audio input as far as any provider is concerned", never "will this call
    succeed".

    Args:
        mime_type: The type to check.
        content_type: When given, the type must also belong to that modality.

    Returns:
        bool: True only for a type listed in SUPPORTED_MIME_TYPES (and matching
            the modality, if one was passed). A format the tables name but no
            provider takes, such as audio/basic, is False here.
    """
    lowered = (mime_type or "").strip().lower()
    if content_type:
        return lowered in SUPPORTED_MIME_TYPES.get(content_type, frozenset())
    return any(lowered in types for types in SUPPORTED_MIME_TYPES.values())


def resolve_media(
    media: Sequence[Tuple[ContentType, str]] | None,
) -> List[MediaReference]:
    """Resolve media references carried in AgentMessage fields to MediaRef objects.

    Transport classification is modality-agnostic: any ContentType value can be
    resolved today. Each (content_type, value) pair is classified by its value:
    - "http://" or "https://" prefix -> kind="url", passed through as-is.
    - "data:" prefix -> kind="base64", MIME type and payload extracted from the URI.
    - "file://" prefix or anything else -> treated as a local path; the file is read
      and base64-encoded, with MIME type sniffed from magic bytes (falling back to
      the extension table, then mimetypes, then "application/octet-stream").

    Every resolved type then goes through normalize_mime_type, which is what puts
    a provider-acceptable string on the wire: sniffing and stdlib guessing both
    report formats the way their sources spell them (audio/x-wav, audio/AMR,
    video/mp4 for an .m4a named .mp4), and a request carrying those is rejected by
    the provider instead, far from the file that caused it.

    Two conditions are reported and never corrected: a type whose modality
    contradicts what the caller declared, and a type correctly identified but
    absent from SUPPORTED_MIME_TYPES for that modality, i.e. one no surveyed
    provider accepts as that kind of input. Both warn and both still return the
    reference, because losing media quietly is the worse failure.

    A local path that does not exist raises FileNotFoundError.

    Args:
        media: List of (content_type, value) tuples, e.g. a filtered slice of
            AgentMessage.query_media.

    Returns:
        List[MediaRef]: Resolved references, each tagged with its content_type,
        order preserved. Empty list if media is empty.
    """
    refs: List[MediaReference] = []
    for content_type, value in media or []:
        if value.startswith(("http://", "https://")):
            kind, payload = "url", value
            raw_mime = _mime_from_extension(value.split("?")[0].split("#")[0])
        elif value.startswith("data:"):
            header, _, payload = value.partition(",")
            kind = "base64"
            raw_mime = header[len("data:") :].split(";")[0]
        else:
            path = Path(value.removeprefix("file://"))
            data = path.read_bytes()
            kind = "base64"
            raw_mime = (
                _sniff_mime_type(data, content_type)
                or _mime_from_extension(path.name)
                or mimetypes.guess_type(path.name)[0]
            )
            payload = base64.b64encode(data).decode("ascii")
        mime_type = normalize_mime_type(raw_mime, content_type)
        detected = modality_of_mime(mime_type)
        if detected is not None and detected != content_type:
            # The caller declared one modality and the bytes say another. Most
            # often a mis-tagged query_media entry; passing it through silently
            # would send an image part to an audio endpoint, or worse, make a
            # model answer about media it was never given.
            logger.warning(
                f"Media declared as {content_type!r} has format {mime_type}, "
                f"which is {detected} content; check the content_type in query_media."
            )
        elif (
            mime_type != "application/octet-stream"
            and content_type in SUPPORTED_MIME_TYPES
            and mime_type not in SUPPORTED_MIME_TYPES[content_type]
        ):
            # The format was identified and it is simply not one any chat model
            # takes as this modality (audio/basic from stdlib, audio/speex from
            # the Ogg probe, image/svg+xml from an extension). Without this the
            # request fails at the provider with a format error that names
            # nothing the caller can act on. The reference is still returned:
            # dropping media silently is worse than a loud warning, and an
            # integration talking to a provider not surveyed here may accept it.
            logger.warning(
                f"Media format {mime_type} is not known to be accepted by any LLM "
                f"provider as {content_type} input; the request may be rejected. "
                f"Supported {content_type} formats: "
                f"{sorted(SUPPORTED_MIME_TYPES[content_type])}."
            )
        refs.append(
            {
                "content_type": content_type,
                "kind": kind,
                "value": payload,
                "mime_type": mime_type,
            }
        )
    return refs


def agent_capabilities(
    streaming: bool = False,
    push_notifications: bool = False,
    extensions: Sequence[Any] | None = None,
    extended_agent_card: bool = False,
) -> AgentCapabilities:
    """Build the a2a ``AgentCard.capabilities`` field for an agent.

    This is the *transport* capability declaration of an agent card: whether the
    agent's server can stream responses (SSE), accept push-notification (webhook)
    updates, declares protocol extensions, or exposes an authenticated extended
    card. It is unrelated to the media-modality capabilities reported by
    ``BaseLLMChain.detect_capabilities`` (image/audio/video/document), which feed
    ``default_input_modes``/``default_output_modes`` instead.

    The defaults are all-False because an in-process ``BaseAgent`` implements none
    of these transports: ``aexecute`` is async execution, not SSE streaming, and
    ``state_change_callback`` is an in-process hook, not a webhook. Flip a flag to
    True only when the agent is actually hosted behind an a2a server that
    implements it, so clients are never advertised a transport that then stalls.

    Args:
        streaming: Server supports streaming (SSE) task responses.
        push_notifications: Server supports push-notification (webhook) updates.
        extensions: Declared ``AgentExtension`` entries for protocol extensions.
        extended_agent_card: An authenticated extended agent card is available.

    Returns:
        AgentCapabilities: A populated capabilities message for ``AgentCard``.
    """
    return AgentCapabilities(
        streaming=streaming,
        push_notifications=push_notifications,
        extensions=list(extensions) if extensions else [],
        extended_agent_card=extended_agent_card,
    )


# --- Hugging Face model-card capability detection (framework-agnostic) ---
#
# Shared by the integration packages: the task-tag grammar below is a property
# of Hugging Face, not of any one agent framework, so it lives in core and is
# maintained once.

# Hugging Face repo ids are the only identifiers that resolve to a model card.
_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_REPO_ID_PREFIXES = ("huggingface:", "hf:")

# Modality keywords, matched against the input side of an HF task tag.
_MODALITY_TAG_KEYWORDS: Dict[ContentType, Tuple[str, ...]] = {
    "image": ("image", "visual", "vision", "chart", "ocr"),
    "audio": ("audio", "speech", "voice"),
    "video": ("video", "frame"),
    "document": ("document", "pdf"),
}

# The missing-extra warning should surface once per process, not per lookup.
_HUB_UNAVAILABLE_LOGGED = False


def task_tags_to_capabilities(
    task_tags: Iterable[str | None],
) -> Dict[ContentType, bool]:
    """Map the task tags a Hugging Face repo declares onto input modalities.

    Only the input side of each tag counts, so "text-to-speech" does not imply
    audio input. When a repo declares at least one task, the tags are read as a
    complete statement of what it accepts: a modality no tag mentions reports
    False rather than staying unknown. That is what separates variants of one
    family, e.g. google/gemma-3n-e4b-it (tagged audio-text-to-text) from
    google/gemma-3-27b-it (tagged image-text-to-text only). A repo published
    under an incomplete tag set will therefore under-report; media_support is
    the override. With no usable tag at all the result is empty, i.e.
    undecidable, which keeps auto mode fail-loud.

    Args:
        task_tags: Declared task identifiers, e.g. "image-text-to-text".

    Returns:
        Dict[ContentType, bool]: One entry per gated modality, or empty when the
        repo declares no usable task.
    """
    tags = [tag.lower() for tag in task_tags if isinstance(tag, str) and tag.strip()]
    if not tags:
        return {}
    if any(tag == "any-to-any" for tag in tags):
        return {modality: True for modality in _MODALITY_TAG_KEYWORDS}
    # Document/PDF support is not expressed by the task-tag grammar, so it is
    # only reported when positively matched; never denied from absence.
    capabilities: Dict[ContentType, bool] = {
        modality: False for modality in ("image", "audio", "video")
    }
    for tag in tags:
        input_side = tag.split("-to-")[0]
        for modality, keywords in _MODALITY_TAG_KEYWORDS.items():
            if any(keyword in input_side for keyword in keywords):
                capabilities[modality] = True
    return capabilities


def extract_repo_id(candidates: Iterable[str | None]) -> str | None:
    """Pick the first candidate that is a Hugging Face repo id (namespace/name).

    Provider model names are not repo ids ("gpt-4o", "gemma3:12b"), so they are
    rejected here rather than looked up and 404'd. Qualifiers are stripped so
    "org/repo:q8_0" or "org/repo@rev" still resolve.

    Args:
        candidates: Identifier strings in priority order; non-strings ignored.

    Returns:
        str | None: The repo id, or None if no candidate looks like one.
    """
    for raw in candidates:
        if not isinstance(raw, str) or not raw.strip():
            continue
        name = raw.strip()
        lowered = name.lower()
        for prefix in _REPO_ID_PREFIXES:
            if lowered.startswith(prefix):
                name = name[len(prefix) :]
                break
        name = re.split(r"[@:]", name, maxsplit=1)[0]
        if _REPO_ID_PATTERN.match(name):
            return name
    return None


def huggingface_capabilities(repo_id: str) -> Dict[ContentType, bool]:
    """Look up a model's input modalities from its Hugging Face model card.

    Performs one network request; callers are expected to cache the result.
    Requires huggingface-hub (the "hf" extra of the aap integration packages);
    without it, or when offline, or for a private or renamed repo, the lookup
    reports undecidable so auto mode keeps the fail-loud behavior instead of
    breaking the request path.

    Args:
        repo_id: The Hugging Face repo id, e.g. "google/gemma-3-4b-it".

    Returns:
        Dict[ContentType, bool]: Capabilities read from the repo's declared
        tasks, possibly empty.
    """
    global _HUB_UNAVAILABLE_LOGGED
    try:
        from huggingface_hub import model_info  # type: ignore[import-not-found]
    except ImportError:
        if not _HUB_UNAVAILABLE_LOGGED:
            _HUB_UNAVAILABLE_LOGGED = True
            logger.warning(
                "huggingface-hub is not installed, so Hugging Face capability "
                'detection is off. Install the "hf" extra of the aap '
                "integration package in use to enable it."
            )
        return {}
    try:
        info = model_info(repo_id)
    except Exception as exc:  # noqa: BLE001 - detection must never break invoke()
        logger.debug("Hugging Face metadata lookup failed for %s: %s", repo_id, exc)
        return {}
    tags = [getattr(info, "pipeline_tag", None)]
    tags += [tag for tag in getattr(info, "tags", None) or [] if "-to-" in tag]
    return task_tags_to_capabilities(tags)


def remove_thinking(response: str) -> str:
    """A snippet to remove thinking
    Note on the model Qwen 3: the openning <think> tag is missing. So we need to manually add it to beginning of the response

    Args:
        response (str): raw response

    Returns:
        str: response without thinking
    """
    if "</think>" in response and "<think>" not in response:
        response = "<think>" + response
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
