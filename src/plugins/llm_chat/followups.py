"""Short-lived eligibility for implicit followups, not automatic replies."""
import time
from dataclasses import dataclass

from .config import config


@dataclass
class Window:
    expires: float
    answer: str


windows: dict[int, Window] = {}


def active(group):
    window = windows.get(group)
    if window and time.monotonic() >= window.expires:
        windows.pop(group, None)
        return None
    return window


def skip(group):
    """Let the group continue on a shorter deadline after a skipped turn."""
    window = active(group)
    if window:
        window.expires = min(window.expires, time.monotonic() + config.followup_window_seconds)


def answered(group, text, *, direct, closed=False):
    if closed:
        windows.pop(group, None)
        return
    if not text or not group:
        return
    timeout = config.followup_window_seconds
    expires = time.monotonic() + (max(timeout, 600) if timeout > 0 else 0)
    answer = text[:600]
    window = windows.get(group)
    if not direct and window:
        # Messages already in the buffer still belong to this exchange.
        window.answer = answer
        window.expires = expires
        return
    for key in list(windows):
        active(key)
    if group not in windows and len(windows) >= 1024:
        windows.pop(next(iter(windows)))
    windows[group] = Window(expires, answer)


def context(group):
    window = active(group)
    if not window:
        return ""
    return (
        "这是未引用、未@的潜在接话。先判断当前在跟谁说，再看内容。"
        "对方回应你上一条的具体问题、邀请或梗，或补充、纠正、反馈同一件事时，可以自然续上。"
        "不同群友也可以接住你的话，判断依据是具体内容。话题相同或时间相邻只能辅助判断；群友也可能是在彼此交流。"
        "填写 followup_target=bot|other|unclear。选 bot 时在 followup_evidence 写清两句的具体对应；"
        "对象是别人或不清楚时旁听。确定在接你的话后，再判断现在出声能补什么。"
        "具体追问、纠正、新线索或值得接的梗可以回应，重复前面的内容或群友已经接住时可以 silent。"
        "几句连着说的内容当成一轮来听，有一个值得回应的点就只接那个点。上一条回复（对话资料）：\n" + window.answer
    )
