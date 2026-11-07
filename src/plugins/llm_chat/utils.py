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


# 表情包发送后，第二轮 LLM 偶尔会输出无意义的 meta 描述（"已发送"、"好的" 之类）
# 这种情况下应该只发表情，不发文字。
_META_MEME_FILLER_KEYWORDS = (
    "已发送", "已发", "发送了", "发了", "发出", "已成功",
    "已选好", "已选", "已挑", "选好了",
    "好的", "好哒", "ok", "okk", "嗯嗯", "嗯", "收到",
    "完成", "搞定", "done",
    "表情已发", "表情包已发", "图已发", "图片已发",
)


def is_meta_meme_filler(text: str) -> bool:
    """判定一段文字是不是 send_meme 后的 meta 废话（可丢弃，只发表情）。"""
    if not text:
        return True
    s = text.strip()
    # 非常短的回复：≤ 8 字（覆盖 "已发送！" "好的~" 这种）
    if len(s) > 8:
        return False
    low = s.lower().rstrip("。.!！~～")
    for kw in _META_MEME_FILLER_KEYWORDS:
        if kw in low:
            return True
    return False
