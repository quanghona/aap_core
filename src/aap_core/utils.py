from __future__ import annotations

import base64
import logging
import mimetypes
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

if TYPE_CHECKING:
    from .types import ContentType, MediaReference

logger = logging.getLogger(__name__)

# Magic-byte signatures used to sniff MIME types without external dependencies.
# Image formats are complete for the current scope; audio/video/document
# signatures get appended here as those modalities are wired up (the transport
# classification below is already modality-agnostic).
_MAGIC_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def _sniff_mime_type(data: bytes) -> str | None:
    """Detect MIME type from magic bytes. Returns None when unknown."""
    for signature, mime_type in _MAGIC_SIGNATURES:
        if data.startswith(signature):
            return mime_type
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


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
      mimetypes, then to "application/octet-stream").

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
            mime_type = mimetypes.guess_type(value.split("?")[0])[0]
            refs.append(
                {
                    "content_type": content_type,
                    "kind": "url",
                    "value": value,
                    "mime_type": mime_type or "application/octet-stream",
                }
            )
        elif value.startswith("data:"):
            header, _, payload = value.partition(",")
            mime_type = header[len("data:") :].split(";")[0]
            refs.append(
                {
                    "content_type": content_type,
                    "kind": "base64",
                    "value": payload,
                    "mime_type": mime_type or "application/octet-stream",
                }
            )
        else:
            path = Path(value.removeprefix("file://"))
            data = path.read_bytes()
            mime_type = (
                _sniff_mime_type(data)
                or mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
            refs.append(
                {
                    "content_type": content_type,
                    "kind": "base64",
                    "value": base64.b64encode(data).decode("ascii"),
                    "mime_type": mime_type,
                }
            )
    return refs


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
