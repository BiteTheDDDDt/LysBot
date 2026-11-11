"""情感分析模块 - 检测群友对bot的态度"""
import time
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import deque

from nonebot import logger

@dataclass
class SentimentRecord:
    """情感记录"""
    content: str
    timestamp: float
    user_id: int
    sentiment_type: str  # 'negative', 'neutral', 'positive'
    confidence: float  # 置信度 0-1


@dataclass
class GroupSentimentState:
    """群情感状态"""
    # 最近情感记录（滑动窗口）
    records: deque = field(default_factory=lambda: deque(maxlen=50))
    # 当前厌烦度 0-100
    annoyance_score: float = 0.0
    # 最后更新时间
    last_update: float = field(default_factory=time.time)
    # 连续负面记录数
    consecutive_negative: int = 0
    # 减少主动参与的冷却结束时间
    quiet_mode_until: Optional[float] = None


class SentimentAnalyzer:
    """情感分析器 - 基于LLM的智能情感分析"""

    def __init__(self):
        # 群情感状态 {group_id: GroupSentimentState}
        self._group_states: Dict[int, GroupSentimentState] = {}
        # 群友持续不想互动时，减少主动参与的时长
        self.quiet_mode_duration = 1800  # 30分钟
        # 负面记录衰减时间
        self.decay_time = 600  # 10分钟
        # LLM分析缓存 {cache_key: (result, timestamp)}
        self._llm_cache: Dict[tuple[str, int, str], tuple] = {}
        # 缓存有效期
        self._cache_ttl = 60  # 60秒
        # 正在进行的分析任务 {cache_key: task}
        self._pending_tasks: Dict[tuple[str, int, str], asyncio.Task] = {}

    def _get_state(self, group_id: int) -> GroupSentimentState:
        """获取或创建群情感状态"""
        if group_id not in self._group_states:
            self._group_states[group_id] = GroupSentimentState()
        return self._group_states[group_id]

    async def _analyze_with_llm(self, message: str, *, session_id: str = "_unknown_") -> tuple[str, float]:
        """使用LLM分析情感

        Returns:
            (sentiment_type, confidence): 情感类型和置信度
        """
        from .conversation import current_generation, ensure_current_session

        # A shared task retains its first caller's conversation context. A new
        # generation must neither join that task nor reuse its cached result.
        ensure_current_session()
        cache_key = (session_id, current_generation(session_id), message)
        now = time.time()

        if cache_key in self._llm_cache:
            result, timestamp = self._llm_cache[cache_key]
            if now - timestamp < self._cache_ttl:
                return result

        task = self._pending_tasks.get(cache_key)
        if task is None:
            async def analyze_once():
                from .config import config
                try:
                    try:
                        result = await asyncio.wait_for(
                            self._do_llm_analysis(message, session_id=session_id),
                            timeout=max(0.1, float(config.sentiment_total_timeout)),
                        )
                    except asyncio.TimeoutError:
                        result = ("neutral", 0.0)
                    completed_at = time.time()
                    self._llm_cache[cache_key] = (result, completed_at)
                    for key, (_, timestamp) in list(self._llm_cache.items()):
                        if completed_at - timestamp >= self._cache_ttl:
                            self._llm_cache.pop(key, None)
                    while len(self._llm_cache) > 512:
                        self._llm_cache.pop(next(iter(self._llm_cache)))
                    return result
                finally:
                    if self._pending_tasks.get(cache_key) is asyncio.current_task():
                        self._pending_tasks.pop(cache_key, None)
            task = asyncio.create_task(analyze_once(), name="sentiment-analysis")
            self._pending_tasks[cache_key] = task
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        # A cancelled waiter must not cancel work shared by another message.
        return await asyncio.shield(task)

    async def close(self) -> None:
        tasks = list(self._pending_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._pending_tasks.clear()

    async def _do_llm_analysis(self, message: str, *, session_id: str = "_unknown_") -> tuple[str, float]:
        """实际调用LLM进行分析"""
        try:
            from .models import model_manager, Message as LLMMessage

            prompt = f"""分析以下群聊消息中对AI助手（bot）的情感态度。

消息内容：
"{message[:200]}"

结合群聊口吻判断真实态度：朋友间互损、口头脏话和自嘲本身不等于厌烦，不仅凭攻击性词语判负面。明确拒绝继续、嫌你打扰或持续不耐烦才按负面程度判断。
请判断这条消息对AI助手的情感倾向，只输出以下格式之一：
- negative:高 - 明显表现出厌烦、讨厌、让bot闭嘴、攻击性语言
- negative:中 - 有些冷淡、嫌弃、暗示bot话多
- negative:低 - 轻微不满、敷衍
- neutral - 普通对话，无明显情感
- positive - 友好、喜欢、认可bot

只输出上述格式中的一个，不要有其他内容。"""

            messages = [
                LLMMessage(role="system", content=prompt),
                LLMMessage(role="user", content="分析")
            ]

            # role="sentiment" 决定使用哪个模型（轻量级即可）
            result = await model_manager.chat(messages, role="sentiment", caller="sentiment:analyze", session_id=session_id)
            result = result.strip().lower()

            return {
                "negative:高": ("negative", 0.9),
                "negative:中": ("negative", 0.7),
                "negative:低": ("negative", 0.5),
                "positive": ("positive", 0.6),
            }.get(result, ("neutral", 0.0))
        except Exception as error:
            logger.warning(f"[情感分析] 模型不可用: {type(error).__name__}")
            return ("neutral", 0.0)

    async def analyze(
        self,
        message: str,
        user_id: int,
        group_id: int,
        is_to_bot: bool = False,
        directed_at_other: bool = False,
    ) -> SentimentRecord:
        """分析单条消息的情感（优先使用LLM）

        Returns:
            SentimentRecord: 情感记录
        """
        from .conversation import ensure_current_session

        ensure_current_session()
        state = self._get_state(group_id)

        # 清理消息
        clean_msg = message.strip()
        if not clean_msg:
            return SentimentRecord(
                content=message,
                timestamp=time.time(),
                user_id=user_id,
                sentiment_type='neutral',
                confidence=0.0
            )

        # 检查是否是真的在说bot
        from .proactive import proactive_manager
        bot_just_spoke = proactive_manager.just_spoke(group_id, within_seconds=20.0)
        actually_about_bot = is_to_bot or (bot_just_spoke and not directed_at_other)

        if not actually_about_bot:
            return SentimentRecord(
                content=message,
                timestamp=time.time(),
                user_id=user_id,
                sentiment_type='neutral',
                confidence=0.0
            )

        # 使用LLM分析
        sentiment_type, confidence = await self._analyze_with_llm(clean_msg, session_id=f"group_{group_id}")

        ensure_current_session()

        record = SentimentRecord(
            content=message,
            timestamp=time.time(),
            user_id=user_id,
            sentiment_type=sentiment_type,
            confidence=confidence
        )

        self._update_state(state, record)
        return record

    def _update_state(self, state: GroupSentimentState, record: SentimentRecord):
        """更新群情感状态"""
        now = time.time()

        # 衰减旧记录的影响
        self._apply_decay(state, now)

        # 添加新记录
        state.records.append(record)
        state.last_update = now

        # 更新连续负面计数
        if record.sentiment_type == 'negative':
            state.consecutive_negative += 1
        else:
            state.consecutive_negative = 0

        # 更新厌烦度
        if record.sentiment_type == 'negative':
            # 负面记录增加厌烦度
            increment = record.confidence * 15  # 每次最多+15
            # 连续负面额外增加
            if state.consecutive_negative >= 2:
                increment += state.consecutive_negative * 5
            state.annoyance_score = min(100, state.annoyance_score + increment)

        elif record.sentiment_type == 'positive':
            # 正面记录降低厌烦度
            decrement = record.confidence * 10
            state.annoyance_score = max(0, state.annoyance_score - decrement)

        # 持续厌烦时减少主动参与
        if state.annoyance_score >= 60:
            state.quiet_mode_until = now + self.quiet_mode_duration
            logger.info(f"[情感分析] 群状态进入减少主动参与模式，厌烦度={state.annoyance_score:.1f}")

        logger.debug(f"[情感分析] 更新状态: 类型={record.sentiment_type}, "
                    f"置信度={record.confidence:.2f}, 厌烦度={state.annoyance_score:.1f}")

    def _apply_decay(self, state: GroupSentimentState, now: float):
        """应用时间衰减"""
        time_passed = now - state.last_update
        if time_passed > self.decay_time:
            # 每10分钟衰减一次
            decay_factor = time_passed / self.decay_time
            decay_amount = decay_factor * 5  # 每次衰减5点
            state.annoyance_score = max(0, state.annoyance_score - decay_amount)

    def get_annoyance_score(self, group_id: int) -> float:
        """获取当前厌烦度分数 0-100"""
        state = self._get_state(group_id)
        self._apply_decay(state, time.time())
        return state.annoyance_score

    def is_quiet_mode(self, group_id: int) -> bool:
        """检查是否处于减少主动参与的冷却期"""
        state = self._get_state(group_id)
        if state.quiet_mode_until is None:
            return False
        if time.time() > state.quiet_mode_until:
            state.quiet_mode_until = None
            return False
        return True

    def should_speak_less(self, group_id: int) -> tuple[bool, float]:
        """判断是否应该减少发言

        Returns:
            (是否减少, 概率倍数)
        """
        score = self.get_annoyance_score(group_id)

        if score >= 80:
            # 高度厌烦，大幅降低概率
            return True, 0.1
        elif score >= 60:
            # 中度厌烦，降低概率
            return True, 0.3
        elif score >= 40:
            # 轻度厌烦，稍微降低
            return True, 0.6

        return False, 1.0

    def get_mood_hint(self, group_id: int) -> str:
        """获取情绪提示，用于注入系统提示词"""
        score = self.get_annoyance_score(group_id)

        if score >= 80:
            return "【当前氛围】群友互动意愿很低，暂时不主动发言；对方明确不想继续时，就结束当前互动。"
        elif score >= 60:
            return "【当前氛围】群友可能觉得回复过多，减少主动参与，必要的回应保持简短。"
        elif score >= 40:
            return "【当前氛围】群友们对你有些冷淡，请谨慎发言，不要说太多。"

        return ""

    def get_state_summary(self, group_id: int) -> dict:
        """获取状态摘要（用于调试）"""
        state = self._get_state(group_id)
        return {
            "annoyance_score": round(state.annoyance_score, 1),
            "consecutive_negative": state.consecutive_negative,
            "is_quiet_mode": self.is_quiet_mode(group_id),
            "record_count": len(state.records),
        }


# 全局分析器实例
sentiment_analyzer = SentimentAnalyzer()
