"""Short-lived eligibility for implicit followups, not automatic replies."""
import time
from dataclasses import dataclass
from itertools import count
import json

from .config import config


@dataclass
class Window:
    expires: float
    answer: str
    message_ids: tuple[int, ...] = ()
    requested_image_from: int | None = None


windows: dict[int, dict[int, Window]] = {}
_ids = count(1)


def _active(group) -> dict[int, Window]:
    matches = windows.get(group)
    if not matches:
        return {}
    now = time.monotonic()
    for key in [key for key, window in matches.items() if now >= window.expires]:
        del matches[key]
    if not matches:
        windows.pop(group, None)
    return matches


def active(group):
    return next(reversed(_active(group).values()), None)


def snapshot(group, *, requested_image_from=None) -> dict[int, str]:
    return {key: window.answer for key, window in _active(group).items()
            if requested_image_from is None or window.requested_image_from == requested_image_from}


def quoted_anchor(group, message_id: int | None) -> int | None:
    if message_id is None:
        return None
    return next((key for key, window in _active(group).items() if message_id in window.message_ids), None)


def skip(group, anchor_ids=None):
    """Shorten the grace period of the threads that could have been addressed."""
    deadline = time.monotonic() + config.followup_window_seconds
    for key, window in _active(group).items():
        if anchor_ids is None or key in anchor_ids:
            window.expires = min(window.expires, deadline)


def answered(group, text, *, direct, closed=False, anchor_id=None, message_ids=(), requested_image_from=None):
    if not group:
        return
    matches = _active(group)
    if anchor_id is None and not direct and len(matches) == 1:
        anchor_id = next(iter(matches))
    if closed:
        matches.pop(anchor_id, None)
        if not matches:
            windows.pop(group, None)
        return
    if not text:
        return
    timeout = config.followup_window_seconds
    expires = time.monotonic() + (max(timeout, 600) if timeout > 0 else 0)
    if group not in windows:
        for key in list(windows):
            _active(key)
    if group not in windows and len(windows) >= 1024:
        windows.pop(next(iter(windows)))
    anchor_id = anchor_id if anchor_id is not None else next(_ids)
    # Keep the match ID on renewal so events already collected can continue it.
    previous = matches.pop(anchor_id, None)
    if previous:
        previous.answer = text[:600]
        previous.expires = expires
        previous.requested_image_from = requested_image_from
        if message_ids:
            previous.message_ids = tuple(message_ids)
    matches[anchor_id] = previous or Window(expires, text[:600], tuple(message_ids), requested_image_from)
    windows[group] = matches


def context(group, records=None):
    if records is None:
        records = [{"followup_anchor_id": key, "text": text} for key, text in snapshot(group).items()]
    if not records:
        return ""
    return (
        "这是未引用、未@的潜在接话。先判断当前在跟谁说，再看内容。"
        "对方回应你近期发言中的具体问题、邀请或梗，或补充、纠正、反馈同一件事时，可以自然续上。"
        "不同群友也可以接住你的话，判断依据是具体内容。话题相同或时间相邻只能辅助判断；群友也可能是在彼此交流。"
        "填写 followup_target=bot|other|unclear。选 bot 时填写对应的 followup_anchor_id，并在 followup_evidence 写清具体对应；"
        "对象是别人或不清楚时旁听。确定在接你的话后，对照那条已发回复：这次回答会多解决哪个问题、补哪条观点，或接住哪个笑点？"
        "若只能换个例子重说同一结论，或对方已接过你的提醒继续讨论，选 silent。"
        "几句连着说的内容当成一轮来听，有一个值得回应的点就只接那个点。"
        "近期可接的 Bot 回复（对话资料，candidate_message_ids 是当时已经看到该回复的候选消息编号）：\n"
        + json.dumps(records, ensure_ascii=False)
    )
