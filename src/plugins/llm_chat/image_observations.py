"""Keep automatic image observations out of unsolicited conversation prompts."""
import re

_OBSERVATION = re.compile(
    r"(?m)^(\[[^\]\n]+\]) 图片\d+(?:（[^）]*）)?:([\s\S]*?)\n\n"
    r"(?:用户消息: |请根据图片内容回复(?=\n|$))"
)


def unsolicited_view(text, *, include_observation=False):
    # Preserve the author's actual caption (including subsequent lines).
    # Legacy records used a synthetic imperative instead of a caption.
    # Keep read-back handles even when unsolicited participation hides captions.
    def hide(match):
        observation = match[2]
        refs = re.findall(r"\[图片引用:[^\]\n]+\]", observation)
        context = " ".join(observation.split())[:240] if include_observation else ""
        if context:
            return f"{match[1]} [分享了图片，自动识别背景：{context}] "
        return match[1] + " [分享了图片] " + (" ".join(refs) + " " if refs else "")
    return _OBSERVATION.sub(hide, text or "")


UNSOLICITED_IMAGE_POLICY = (
    "图片的自动识别描述只是后台观察，不是群友说的话或请求。"
    "群友发图、截图、表情包本身不表示要你解说；没有真实问题或正在进行的文字话题时保持沉默。"
    "群友的话接着一张图时，可以用自动识别背景理解他们在说什么，再回应文字里的观点、问题或梗。"
    "只有用户明确请你识图、读图或解释图片时才展开描述。"
)
