"""主动发言模块 - 智能判断版"""
import asyncio
import random
import time
from typing import Dict, Set, List, Optional
from dataclasses import dataclass

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot

from .models import model_manager
from .memory import memory_manager
from .config import config
from .utils import clean_response
from .sentiment import sentiment_analyzer
from .context_builder import build_context_messages
from .reply_policy import PROACTIVE_REPLY, constrain_plan, enforce_reply_budget
from .planner import PlannerDecision, execute_planned_side_effect, plan_chat_action
from .turns import is_meme_only, prepare_planned_meme, send_response_turn, wants_meme
from .conversation import ensure_send_allowed, invalidate, serialized_turn

# Internal participation tuning; not operator-facing settings.
IDLE_THRESHOLD_SECONDS = 3600
CHECK_WINDOW_SECONDS = 180
PLANNER_COOLDOWN_SECONDS = 60
# Contemplative/negative topics that justify an unsolicited check-in.
CONVERSATION_KEYWORDS = (
    "意义", "存在", "无聊", "空虚",
    "痛苦", "难受", "烦恼",
    "放弃", "放下", "想死",
    "累", "疲惫", "绝望",
)
KEYWORD_TRIGGER_PROBABILITY = 0.1



@dataclass
class RecentMessage:
    """最近消息记录"""
    content: str
    timestamp: float
    user_id: int


class ProactiveManager:
    """主动发言管理器 - 智能判断版"""

    def __init__(self):
        # 群消息计数器 {group_id: count}
        self.msg_counters: Dict[int, int] = {}
        # 冷却时间记录 {group_id: last_trigger_time}
        self.cooldowns: Dict[int, float] = {}
        # 主动发言规划冷却 {group_id: last_plan_time}，无论规划结果如何都冷却
        self.planner_cooldowns: Dict[int, float] = {}
        # 正在处理中的群，防止重复触发
        self.processing: Set[int] = set()
        # 刚刚主动发言的群 {group_id: timestamp}
        self._just_spoke: Dict[int, float] = {}
        # 最近消息缓存 {group_id: [RecentMessage]}，用于判断窗口
        self._recent_messages: Dict[int, List[RecentMessage]] = {}
        # bot最后主动发言时间 {group_id: timestamp}
        self._last_spoke: Dict[int, float] = {}
        # 最近一次自我思考内容缓存 {group_id: content}
        self._last_think: Dict[int, str] = {}
        # 判断不触发时，推迟相同群下次空闲判断的冷却 {group_id: timestamp}
        self._idle_retry_cooldowns: Dict[int, float] = {}
        # Planner 主动选择 wait 后的延迟重规划任务。
        self._wait_tasks: Dict[int, asyncio.Task] = {}

    def cancel_wait(self, group_id: int) -> None:
        """Cancel a pending proactive wait, normally because a direct turn won."""
        task = self._wait_tasks.pop(group_id, None)
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()

    def _schedule_wait_replan(
        self,
        *,
        bot: Bot,
        group_id: int,
        message: str,
        user_id: int,
        decision: PlannerDecision,
        wait_started_at: Optional[float],
        wait_replans: int,
    ) -> None:
        """Schedule one bounded proactive replan while preserving a 120s deadline."""
        loop = asyncio.get_running_loop()
        started_at = wait_started_at if wait_started_at is not None else loop.time()
        max_total = max(1, int(config.planner_wait_max_seconds))
        max_replans = max(1, int(config.planner_wait_max_replans))
        elapsed = loop.time() - started_at
        remaining = max_total - elapsed
        if remaining <= 0 or wait_replans >= max_replans:
            logger.info(
                f"[主动发言] 群 {group_id} wait 已结束 "
                f"(elapsed={elapsed:.1f}s replans={wait_replans}/{max_replans})"
            )
            self.cancel_wait(group_id)
            return

        delay = min(float(decision.wait_delay_seconds), remaining)
        self.cancel_wait(group_id)

        async def resume() -> None:
            try:
                logger.info(
                    f"[主动发言] 群 {group_id} wait {delay:.1f}s 后重规划 "
                    f"({wait_replans + 1}/{max_replans})"
                )
                await asyncio.sleep(delay)
                await self.try_proactive(
                    bot,
                    group_id,
                    message,
                    user_id,
                    _deferred=True,
                    _wait_started_at=started_at,
                    _wait_replans=wait_replans + 1,
                )
            except asyncio.CancelledError:
                logger.debug(f"[主动发言] 群 {group_id} 的 wait 被取消")
                raise
            finally:
                if self._wait_tasks.get(group_id) is asyncio.current_task():
                    self._wait_tasks.pop(group_id, None)

        self._wait_tasks[group_id] = asyncio.create_task(
            resume(),
            name=f"planner-wait-proactive-{group_id}",
        )


    def mark_just_spoke(self, group_id: int):
        """记录 Bot 最近一次发言时间。"""
        now = time.time()
        self._just_spoke[group_id] = now
        self._last_spoke[group_id] = now

    def just_spoke(self, group_id: int, within_seconds: float = 2.0) -> bool:
        """检查是否刚刚主动发言过"""
        if group_id not in self._just_spoke:
            return False
        return time.time() - self._just_spoke[group_id] < within_seconds

    def is_on_cooldown(self, group_id: int) -> bool:
        """检查是否在冷却中或消息过多"""
        if self.msg_counters.get(group_id, 0) >= 100:
            return False
        if group_id not in self.cooldowns:
            return False
        return time.time() - self.cooldowns[group_id] < config.proactive_cooldown


    def set_cooldown(self, group_id: int):
        """设置冷却"""
        self.cooldowns[group_id] = time.time()

    def is_on_planner_cooldown(self, group_id: int) -> bool:
        """检查是否处于主动发言规划冷却期。"""
        if group_id not in self.planner_cooldowns:
            return False
        return time.time() - self.planner_cooldowns[group_id] < PLANNER_COOLDOWN_SECONDS

    def set_planner_cooldown(self, group_id: int):
        """设置主动发言规划冷却。"""
        self.planner_cooldowns[group_id] = time.time()

    def is_idle(self, group_id: int) -> bool:
        """检查是否空闲（超过阈值时间没发言）"""
        # 检查是否在空闲重试冷却中
        if time.time() < self._idle_retry_cooldowns.get(group_id, 0):
            return False

        if group_id not in self._last_spoke:
            # 从未发言过，视为空闲
            return True
        return time.time() - self._last_spoke[group_id] > IDLE_THRESHOLD_SECONDS

    def add_recent_message(self, group_id: int, content: str, user_id: int):
        """添加最近消息到缓存"""
        from .image_observations import unsolicited_view
        content = unsolicited_view(content)
        if group_id not in self._recent_messages:
            self._recent_messages[group_id] = []

        self._recent_messages[group_id].append(RecentMessage(
            content=content,
            timestamp=time.time(),
            user_id=user_id
        ))

        # 清理超过检查窗口的旧消息
        cutoff = time.time() - CHECK_WINDOW_SECONDS
        self._recent_messages[group_id] = [
            m for m in self._recent_messages[group_id]
            if m.timestamp > cutoff
        ]

    def get_recent_messages_text(self, group_id: int, window_seconds: int = None) -> str:
        """获取最近检查窗口内的消息文本

        Args:
            group_id: 群号
            window_seconds: 时间窗口秒数，默认使用 CHECK_WINDOW_SECONDS
        """
        if window_seconds is None:
            window_seconds = CHECK_WINDOW_SECONDS

        if group_id not in self._recent_messages:
            return ""

        cutoff = time.time() - window_seconds
        recent = [m for m in self._recent_messages[group_id] if m.timestamp > cutoff]

        if not recent:
            return ""

        # 格式化为简洁的上下文
        lines = []
        for m in recent:
            lines.append(f"用户{m.user_id}: {m.content}")
        return "\n".join(lines)

    def increment_counter(self, group_id: int) -> int:
        """增加消息计数，返回当前计数"""
        if group_id not in self.msg_counters:
            self.msg_counters[group_id] = 0
        self.msg_counters[group_id] += 1
        return self.msg_counters[group_id]

    def reset_counter(self, group_id: int):
        """重置计数器"""
        self.msg_counters[group_id] = 0

    def check_keywords(self, message: str) -> bool:
        """检查消息是否包含关键词"""
        message_lower = message.lower()
        for keyword in CONVERSATION_KEYWORDS:
            if keyword in message_lower:
                return True
        return False

    def should_trigger_by_keyword(self) -> bool:
        """检查是否应该通过关键词触发（兜底机制）"""
        return random.random() < KEYWORD_TRIGGER_PROBABILITY



    async def generate_proactive_response(
        self,
        group_id: int,
        recent_context: str,
        trigger_message: str = "",
        planner_decision: Optional[PlannerDecision] = None,
        planned_meme: Optional[dict] = None,
    ) -> str:
        """生成主动发言内容"""
        planner_decision = constrain_plan(planner_decision, PROACTIVE_REPLY)
        session_id = f"group_{group_id}"

        # 获取历史记录作为上下文（主动发言时检查压缩）
        history = await memory_manager.get_history(0, group_id, check_compress=True)

        # Previous participation decisions are not user requests or fresh evidence.
        think_context = ""
        from .image_observations import UNSOLICITED_IMAGE_POLICY, unsolicited_view
        recent_context = unsolicited_view(recent_context)
        trigger_message = unsolicited_view(trigger_message)



        # 使用最近窗口消息作为检索 query，并补充自我思考，空时回退到触发消息
        query_text = recent_context.strip() if recent_context else ""
        if think_context:
            query_text = f"{query_text}\n{think_context}" if query_text else think_context
        if not query_text and trigger_message:
            query_text = trigger_message.strip()

        from .proactive_style import PROACTIVE_VALUE_POLICY
        extra_system = PROACTIVE_VALUE_POLICY + UNSOLICITED_IMAGE_POLICY + "\n本轮是主动参与；围绕触发消息自然接话，不必复述或总结。"
        # 主动发言原本只把最近 5 条普通历史交给 chat 模型；保留这个 token 边界。
        # Participation hides image observations to avoid unsolicited narration;
        # execution still needs the actual nearby evidence to answer a question.
        normal_history = [m for m in history if not getattr(m, "is_thinking", False)]
        context_history = normal_history[-5:] if len(normal_history) > 5 else normal_history
        messages = await build_context_messages(
            history=context_history,
            session_id=session_id,
            query=query_text,
            user_id=0,
            group_id=group_id,
            user_content=trigger_message.strip() or recent_context.strip(),
            recent_context=recent_context,
            think_context=think_context,
            extra_system=extra_system,
            reply_policy=PROACTIVE_REPLY,
            planner_decision=planner_decision,
            planned_meme=planned_meme,
            related_user_id=0,
        )

        try:
            response = await model_manager.run_turn(
                messages,
                model=memory_manager.get_session_model(0, group_id) or None,
                role="chat",
                caller="proactive:speak",
                reply_policy=PROACTIVE_REPLY,
                session_id=f"group_{group_id}",
                required_tool=(planner_decision.required_tool if planner_decision else None),
                media_followup=(planner_decision.media_followup if planner_decision else True),
            )
            return response
        except Exception as e:
            logger.error(f"[主动发言] 生成失败: {e}")
            return ""

    async def try_proactive(self, bot: Bot, group_id: int, message: str, user_id: int = 0, *,
                            _deferred: bool = False, _wait_started_at=None, _wait_replans=0,
                            _recorded: bool = False) -> bool:
        if not _deferred and not _recorded:
            self.add_recent_message(group_id, message, user_id)
            self.increment_counter(group_id)
        return await self._try_proactive(bot, group_id, message, user_id,
            _deferred=_deferred, _wait_started_at=_wait_started_at, _wait_replans=_wait_replans)

    @serialized_turn(lambda self, bot, group_id, *args, **kwargs: f"group_{group_id}", background=True)
    async def _try_proactive(
        self,
        bot: Bot,
        group_id: int,
        message: str,
        user_id: int = 0,
        *,
        _deferred: bool = False,
        _wait_started_at: Optional[float] = None,
        _wait_replans: int = 0,
    ) -> bool:
        """尝试主动发言，返回是否触发了发言

        Args:
            bot: 机器人实例
            group_id: 群号
            message: 当前收到的消息
            user_id: 发送者ID
        """
        if not config.enable_proactive:
            return False

        # 正在处理中，跳过
        if group_id in self.processing:
            return False


        # 检查是否在发言冷却中 (30分钟冷却)
        if self.is_on_cooldown(group_id):
            return False

        # 情感分析：如果被讨厌，大幅降低主动发言概率
        if config.enable_sentiment_analysis and not _deferred:
            should_speak_less, probability_factor = sentiment_analyzer.should_speak_less(group_id)
            if should_speak_less:
                # 主动发言概率降低
                if random.random() > probability_factor:
                    logger.info(f"[主动发言] 群 {group_id} 情感状态不佳，跳过主动发言")
                    return False

        count = self.msg_counters.get(group_id, 0)

        # 判断触发条件：1. 消息达到阈值(5条) 2. 达到空闲阈值(30分钟) 3. 关键词触发(概率)
        should_judge = False
        trigger_reason = ""

        if _deferred:
            should_judge = True
            trigger_reason = "Planner wait 延迟重规划"
        elif count >= config.proactive_msg_threshold:
            should_judge = True
            trigger_reason = f"消息计数达到 {count} 条"
        elif self.check_keywords(message) and self.should_trigger_by_keyword():
            should_judge = True
            trigger_reason = "关键词触发"
        elif self.is_idle(group_id):
            should_judge = True
            trigger_reason = "群聊空闲检测"

        if not should_judge:
            return False

        # 检查是否在判断冷却中 (防止在达到阈值后频繁请求LLM，默认60s)
        if not _deferred and self.is_on_planner_cooldown(group_id):
            return False

        # 获取最近消息上下文
        recent_context = self.get_recent_messages_text(group_id)
        if not recent_context:
            return False

        # 设置为正在处理
        self.processing.add(group_id)
        try:
            logger.info(f"[主动发言] 触发判断: {trigger_reason}, 群: {group_id}")

            planner_decision = await plan_chat_action(
                session_id=f"group_{group_id}",
                group_id=group_id,
                current_message=message,
                recent_context=recent_context,
                mode="proactive",
                bot_self_id=str(bot.self_id),
                is_direct=False,
                default_action="silent",
            )
            if planner_decision.action == "wait":
                self.reset_counter(group_id)
                self.set_planner_cooldown(group_id)
                self._schedule_wait_replan(
                    bot=bot,
                    group_id=group_id,
                    message=message,
                    user_id=user_id,
                    decision=planner_decision,
                    wait_started_at=_wait_started_at,
                    wait_replans=_wait_replans,
                )
                return False

            planner_decision = constrain_plan(planner_decision, PROACTIVE_REPLY)
            is_worth = planner_decision.should_continue
            think_content = planner_decision.reason

            # 无论判断结果如何，都先重置计数器并设置判断冷却
            self.reset_counter(group_id)
            self.set_planner_cooldown(group_id)

            if is_worth:
                # 有意义，触发发言
                logger.info(f"[主动发言] planner action={planner_decision.action}，准备生成回复")

                if think_content:
                    self._last_think[group_id] = think_content
                    await memory_manager.add_message(0, "assistant", think_content, group_id, is_thinking=True)
                    logger.info(f"[主动发言] 判定附带思考: {think_content}")
                # 生成回复 (增加重试机制)
                from .tools import (
                    clear_image_context,
                    clear_meme_context,
                    clear_poke_context,
                    clear_voice_context,
                    format_sent_image_for_memory,
                    format_sent_meme_for_memory,
                    format_sent_voice_for_memory,
                    get_last_sent_image,
                    get_last_sent_voice,
                    image_was_sent_this_round,
                    set_meme_context,
                    set_image_context,
                    set_poke_context,
                    set_voice_context,
                    voice_was_sent_this_round,
                )
                from .utils import is_meta_meme_filler

                set_poke_context(group_id=group_id, user_id=user_id, bot=bot)
                set_meme_context(group_id=group_id, user_id=user_id, bot=bot)
                set_voice_context(group_id=group_id, user_id=user_id, bot=bot)
                set_image_context(group_id=group_id, user_id=user_id, bot=bot)
                try:
                    poke_sent = False
                    if planner_decision.action == "poke":
                        # Unsolicited interaction is scoped to the current speaker.
                        if not user_id:
                            return False
                        planner_decision.poke_target_id = "sender"
                        poke_result = await execute_planned_side_effect(planner_decision)
                        poke_sent = poke_result.startswith("戳一戳已发送给 ")
                        if not poke_sent:
                            logger.warning(f"[主动戳一戳] 未发送 group={group_id}: {poke_result}")
                            return False
                        # The action already happened, even if a later text call fails.
                        self.mark_just_spoke(group_id)
                        self.set_cooldown(group_id)
                        await memory_manager.add_message(
                            0, "assistant", f"[已戳一戳 QQ:{user_id}]", group_id,
                        )
                        if not planner_decision.parts:
                            return True
                    else:
                        await execute_planned_side_effect(planner_decision)
                    planned_meme = await prepare_planned_meme(planner_decision)
                    response = None
                    immediate_media_sent = False
                    if is_meme_only(planner_decision):
                        # No useful unsolicited text fallback when the pick is empty.
                        if not planned_meme:
                            return poke_sent
                        response = ""
                    else:
                        for retry in range(2):
                            response = await self.generate_proactive_response(
                                group_id,
                                recent_context,
                                trigger_message=message,
                                planner_decision=planner_decision,
                                planned_meme=planned_meme,
                            )
                            immediate_media_sent = (
                                image_was_sent_this_round() or voice_was_sent_this_round()
                            )
                            if response or planned_meme or immediate_media_sent:
                                break
                            if not getattr(getattr(response, "agent_run", None), "should_reply", True):
                                return poke_sent
                            logger.warning(f"[主动发言] 第 {retry + 1} 次生成结果为空，尝试重试...")

                    if immediate_media_sent:
                        media_type = (
                            "image" if image_was_sent_this_round() else "voice"
                        )
                        image_memory = format_sent_image_for_memory(get_last_sent_image())
                        voice_memory = format_sent_voice_for_memory(get_last_sent_voice())
                        followup_text = enforce_reply_budget(clean_response(response or ""), PROACTIVE_REPLY)
                        sent_followup = ""
                        cancelled = None
                        if followup_text:
                            try:
                                ensure_send_allowed(group_id)
                                from .agent_runner import deliver_reply as _deliver_agent_reply
                                await _deliver_agent_reply(response, followup_text, lambda: bot.send_group_msg(
                                    group_id=group_id,
                                    message=followup_text,
                                ), bot=bot, group_id=group_id)
                                sent_followup = followup_text
                            except asyncio.CancelledError as exc:
                                cancelled = exc
                            except Exception as exc:
                                logger.warning(
                                    f"[主动发言] 即时媒体已发送，但收尾文字发送失败 "
                                    f"group={group_id} type={media_type}: {exc}"
                                )
                        history_text = "\n".join(
                            part
                            for part in [image_memory or voice_memory, sent_followup]
                            if part
                        )
                        if history_text:
                            try:
                                await memory_manager.add_message(
                                    0,
                                    "assistant",
                                    history_text,
                                    group_id,
                                    model_spec=getattr(response, "model_spec", None) if sent_followup else None,
                                )
                            except Exception as exc:
                                logger.warning(
                                    f"[主动发言] 即时媒体写入记忆失败 group={group_id} "
                                    f"type={media_type}: {exc}"
                                )
                        self.mark_just_spoke(group_id)
                        self.set_cooldown(group_id)
                        logger.info(
                            f"[主动发言] 即时媒体已发送并完成本轮 group={group_id} "
                            f"type={media_type} followup_chars={len(sent_followup)}"
                        )
                        if cancelled is not None:
                            raise cancelled
                        return True

                    if response or planned_meme:
                        # 清理回复内容
                        generated_response = response
                        response = clean_response(response or "")
                        if planned_meme and wants_meme(planner_decision) and is_meta_meme_filler(response):
                            logger.info(f"[meme] 主动发言过滤废话文字回复: {response!r}")
                            response = ""

                        async def send_text_reply(text: str, is_first: bool) -> None:
                            ensure_send_allowed(group_id)
                            from .agent_runner import deliver_reply as _deliver_agent_reply
                            await _deliver_agent_reply(generated_response, text, lambda: bot.send_group_msg(group_id=group_id, message=text), bot=bot, group_id=group_id)

                        sent_turn = await send_response_turn(
                            text=response,
                            decision=planner_decision,
                            planned_meme=planned_meme,
                            send_text=send_text_reply,
                            reply_policy=PROACTIVE_REPLY,
                        )
                        if not sent_turn.sent_anything:
                            return poke_sent

                        # 标记刚发言并设置 30 分钟冷却
                        from . import followups
                        if user_id:
                            followups.answered(group_id, [user_id], sent_turn.text_for_memory, direct=True,
                                               closed=planner_decision.action == "close")
                        self.mark_just_spoke(group_id)
                        self.set_cooldown(group_id)

                        # 记录到历史
                        meme_memory = format_sent_meme_for_memory(sent_turn.sent_meme)
                        history_text = "\n".join(
                            part for part in [meme_memory, sent_turn.text_for_memory] if part
                        )
                        await memory_manager.add_message(0, "assistant", history_text, group_id, model_spec=getattr(generated_response, "model_spec", None) if sent_turn.text_for_memory else None)
                        return True
                    return poke_sent
                finally:
                    clear_poke_context()
                    clear_meme_context()
                    clear_voice_context()
                    clear_image_context()
            else:
                logger.info(f"[主动发言] 判断无意义，暂不发言")
                # 如果是由空闲检测触发的判断且被拒，推迟下一次空闲判断 5 分钟
                if trigger_reason == "群聊空闲检测":
                    self._idle_retry_cooldowns[group_id] = time.time() + 300
                    logger.info(f"[主动发言] 群 {group_id} 空闲判断未通过，推迟 5 分钟后再次探测")

        except Exception as e:
            import traceback
            error_msg = str(e)
            error_type = type(e).__name__
            stack_trace = traceback.format_exc()

            # 记录完整错误信息
            logger.error(
                f"[主动发言] 群 {group_id} 处理异常\n"
                f"  错误类型: {error_type}\n"
                f"  错误信息: {error_msg}\n"
                f"  堆栈跟踪:\n{stack_trace}"
            )

        finally:
            self.processing.discard(group_id)

        return False





# 全局管理器实例
proactive_manager = ProactiveManager()
