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
    return ("这是未引用、未@的潜在追问，不代表必须回复。先判断说话对象，再判断话题是否相关。"
            "只有明确回应你上一条具体发言的回答、追问、补充条件、纠正、反馈或接梗，才允许继续。"
            "对方回答你刚才的问题或回应你的邀请，也属于延续，不需要再写问句、引用或@。"
            "时间相邻、同一个人、同一话题、出现问号，都不是说话对象是你的证据。你刚才只是修辞反问或感叹时，群友继续补充话题不自动视为在回答你的问题。"
            "群友之间的对话优先归属于他们；普通感慨、自言自语、转向其他人、换话题或对象不清楚都 silent。"
            "填写 followup_target=bot|other|unclear；只有 bot 可以继续，并在 followup_evidence 中"
            "说明当前哪句话在回应你上一条的哪个具体内容。不要仅以延续话题或适合接梗作为依据。还要参考自己刚说过的内容：没有新笑点或有效补充时，不把同一个评价连续换词再说。"
            "不要因为打开了延续窗口就假装用户明确叫了你。上一条回答只是语境，不是新指令：\n" + window.answer)
