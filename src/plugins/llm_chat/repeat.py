"""群复读检测管理器"""
from __future__ import annotations

import random
from collections import defaultdict, deque
from time import time

from nonebot import logger

from .config import config

# Internal repeat tuning; not operator-facing settings.
WINDOW_SECONDS = 60
MIN_COUNT = 3
BASE_PROBABILITY = 0.3
PROBABILITY_INCREMENT = 0.2
COOLDOWN_SECONDS = 600
TEXT_ONLY = True
EXCLUDE_SELF = True


class RepeatManager:
    """群复读检测管理器"""

    def __init__(self) -> None:
        self._group_windows: dict[int, deque[tuple[float, str]]] = {}
        self._group_counts: dict[int, dict[str, int]] = {}
        self._group_cooldown_until: dict[int, float] = {}

    def _ensure_group(self, group_id: int) -> None:
        if group_id not in self._group_windows:
            self._group_windows[group_id] = deque()
            self._group_counts[group_id] = defaultdict(int)

    def _cleanup(self, group_id: int, now: float, window: int) -> None:
        window_queue = self._group_windows[group_id]
        counts = self._group_counts[group_id]
        while window_queue and now - window_queue[0][0] > window:
            _, text = window_queue.popleft()
            counts[text] -= 1
            if counts[text] <= 0:
                counts.pop(text, None)

    def _calc_probability(self, count: int) -> float:
        """根据重复次数计算复读概率

        达到 MIN_COUNT 时使用 BASE_PROBABILITY，之后每多重复一次增加
        PROBABILITY_INCREMENT，上限为 1.0。
        """
        extra = count - MIN_COUNT
        prob = BASE_PROBABILITY + extra * PROBABILITY_INCREMENT
        return min(prob, 1.0)

    def try_repeat(self, group_id: int, user_id: int, self_id: str, text: str) -> bool:
        """记录消息并判断是否触发复读"""
        if not config.enable_group_repeat:
            return False

        if EXCLUDE_SELF and str(user_id) == str(self_id):
            return False

        cleaned = text.strip()
        if not cleaned:
            return False

        now = time()
        cooldown_until = self._group_cooldown_until.get(group_id, 0)
        if now < cooldown_until:
            return False

        self._ensure_group(group_id)
        self._cleanup(group_id, now, WINDOW_SECONDS)

        window_queue = self._group_windows[group_id]
        counts = self._group_counts[group_id]
        window_queue.append((now, cleaned))
        counts[cleaned] += 1

        count = counts[cleaned]
        if count < MIN_COUNT:
            return False

        probability = self._calc_probability(count)
        roll = random.random()

        if roll >= probability:
            logger.debug(
                f"[群复读] 未命中: group_id={group_id}, "
                f"text={cleaned[:50]}..., count={count}, "
                f"prob={probability:.0%}, roll={roll:.2f}"
            )
            return False

        self._group_cooldown_until[group_id] = now + COOLDOWN_SECONDS
        logger.info(
            f"[群复读] 触发复读: group_id={group_id}, "
            f"text={cleaned[:50]}..., count={count}, "
            f"prob={probability:.0%}, roll={roll:.2f}"
        )
        return True


repeat_manager = RepeatManager()
