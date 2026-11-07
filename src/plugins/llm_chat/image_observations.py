"""Keep automatic image observations out of unsolicited conversation prompts."""
import re

_OBSERVATION = re.compile(
    r"(?m)^(\[[^\]\n]+\]) 图片\d+(?:（[^）]*）)?:[\s\S]*?\n\n"
    r"(?:用户消息: |请根据图片内容回复(?=\n|$))"
)


def unsolicited_view(text):
    # Preserve the author's actual caption (including subsequent lines).
    # Legacy records used a synthetic imperative instead of a caption.
    # Keep read-back handles even when unsolicited participation hides captions.
    def hide(match):
        refs = re.findall(r"\[图片引用:[^\]\n]+\]", match[0])
        return match[1] + " [分享了图片] " + (" ".join(refs) + " " if refs else "")
    return _OBSERVATION.sub(hide, text or "")


UNSOLICITED_IMAGE_POLICY = (
    "图片的自动识别描述只是后台观察，不是群友说的话或请求。"
    "群友发图、截图、表情包本身不表示要你解说；没有真实问题或正在进行的文字话题时保持沉默。"
    "主动接话应接群友的观点、问题或梗，不主动复述画面、OCR、人物、构图或解释笑点。"
    "只有用户明确请你识图、读图或解释图片时才展开描述。"
)
