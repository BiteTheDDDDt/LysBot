import re

from .response_safety import INVALID_REPLY, is_tool_call_text


def normalize_qq_text(text: str) -> str:
    """Remove prose decoration while leaving code and math spans untouched."""
    chunks = re.split(r'(```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$|\$[^$\n]+\$)', text)
    for i in range(0, len(chunks), 2):
        part = re.sub(r'\*\*([^*\n]+)\*\*', r'\1', chunks[i])
        part = re.sub(r'(?m)^\s{0,3}#{1,6}\s+', '', part)
        part = re.sub(r'(?m)^\s*\*\s+', '• ', part)
        chunks[i] = part
    return ''.join(chunks)


def clean_response(text: str, *, preserve_paragraphs: bool = False) -> str:
    """Remove explicit wrappers without guessing which prose is the answer.

    Paragraphs, lists, code and formulas are content in both reply modes.
    ``preserve_paragraphs`` remains accepted for existing callers.
    """
    if is_tool_call_text(text):
        return INVALID_REPLY
    text = (text or "").strip()
    # Only the leading, complete protocol block is metadata. Literal tags in
    # examples, comparisons and Markdown links are ordinary answer content.
    text = re.sub(r'\A(?:<think>.*?</think>\s*)+', '', text, flags=re.IGNORECASE | re.DOTALL)
    from .config import PERSONA
    names = ["助手", "assistant", PERSONA.name, *PERSONA.aliases]
    role = "(?:" + "|".join(re.escape(name) for name in names) + ")"
    text = re.sub(rf'\A(?:\[{role}\][:：]?\s*|{role}[:：]\s*)', '', text, flags=re.IGNORECASE)
    if is_tool_call_text(text):
        return INVALID_REPLY
    return normalize_qq_text(text).strip()
