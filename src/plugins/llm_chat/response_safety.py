"""Recognize tool-call envelopes accidentally emitted as visible reply text."""

import json
import re


INVALID_REPLY = "这次回复异常，请稍后再试。"
_TOOL_NAMES = frozenset({
    "generate_image", "send_voice", "search_web", "recall_history", "send_poke", "send_meme",
})
_LEADING_FENCE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_PARTIAL_CALL = re.compile(
    r'^\{\s*"name"\s*:\s*"(?:functions\.)?('
    + "|".join(sorted(_TOOL_NAMES))
    + r')"\s*,\s*"(?:parameters|arguments)"\s*:',
)


def _is_tool_envelope(value) -> bool:
    if isinstance(value, list):
        return bool(value) and any(_is_tool_envelope(item) for item in value)
    if not isinstance(value, dict):
        return False
    name = value.get("name")
    if isinstance(name, str) and name.removeprefix("functions.") in _TOOL_NAMES:
        return "parameters" in value or "arguments" in value
    return any(_is_tool_envelope(value.get(key)) for key in ("function", "tool_calls", "tool_call"))


def is_tool_call_text(text: str) -> bool:
    """Reject a leading call envelope, not ordinary JSON or explained code examples.

    Inspect the full reply before cleanup/truncation, and discard the whole
    malformed reply (including any self-correction/reasoning after the JSON).
    Never turn this text into an executable tool call.
    """
    candidate = _LEADING_FENCE.sub("", (text or "").lstrip(), count=1).lstrip()
    if not candidate.startswith(("{", "[")):
        return False
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate)
    except ValueError:
        return bool(_PARTIAL_CALL.match(candidate))
    return _is_tool_envelope(value)
