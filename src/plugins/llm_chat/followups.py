"""Short-lived eligibility for implicit followups, not automatic replies."""
import time
from dataclasses import dataclass

from .config import config


@dataclass
class Window:
    users: set[int]
    expires: float
    quiet_expires: float
    remaining: int
    answer: str
    interruptions: int = 0


windows: dict[int, Window] = {}


def active(group):
    window = windows.get(group)
    # A quiet exchange may resume after a few minutes; once others speak, use
    # the normal short window so unrelated group conversation cannot inherit it.
    deadline = (window.quiet_expires if window.interruptions == 0 else window.expires) if window else 0
    if window and (time.monotonic() >= deadline or window.remaining <= 0):
        windows.pop(group, None)
        return None
    return window


def eligible(group, user, directed_elsewhere=False):
    window = active(group)
    return bool(window and user in window.users and not directed_elsewhere)


def observe(group, user, directed_elsewhere=False):
    window = active(group)
    if not window:
        return
    if user in window.users and directed_elsewhere:
        windows.pop(group, None)
    elif user not in window.users:
        window.interruptions += 1
        if window.interruptions >= 3:
            windows.pop(group, None)


def answered(group, users, text, *, direct, closed=False):
    if closed:
        windows.pop(group, None)
        return
    if not text or not group:
        return
    if direct:
        for key in list(windows):
            active(key)
        if len(windows) >= 1024:
            windows.pop(next(iter(windows)))
        now = time.monotonic()
        timeout = config.followup_window_seconds
        quiet_timeout = max(timeout, 600) if timeout > 0 else 0
        windows[group] = Window(set(users), now + timeout, now + quiet_timeout,
                                config.followup_max_turns, text[:600])
    else:
        window = active(group)
        if window:
            window.remaining -= 1
            window.answer = text[:600]
            window.users.update(users)
            # Keep both original deadlines; implicit replies cannot sustain a loop forever.


def context(group):
    window = active(group)
    if not window:
        return ""
    return (
        "这是未引用、未@的潜在接话。先判断当前在跟谁说，再看内容。"
        "对方回应你上一条的具体问题、邀请或梗，或补充、纠正、反馈同一件事时，可以自然续上。"
        "同一个人、话题相同或时间相邻只能辅助判断；修辞反问后群友继续聊天也可能是在彼此交流。"
        "填写 followup_target=bot|other|unclear。选 bot 时在 followup_evidence 写清两句的具体对应；"
        "对象是别人或不清楚时旁听。接话只补本轮的新内容。上一条回复（对话资料）：\n" + window.answer
    )
