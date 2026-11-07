"""聊天处理核心逻辑"""
import json
import re
import asyncio
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime

from nonebot import on_message, logger
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupMessageEvent,
    PrivateMessageEvent,
    MessageEvent,
    Message,
    MessageSegment
)
from nonebot.exception import FinishedException
from nonebot.rule import Rule

from .models import ModelTurnTimeout, model_manager, Message as LLMMessage
from .memory import memory_manager
from .config import config
from .context_builder import build_context_messages
from .reply_policy import (
    MEDIA_REPLY, constrain_plan, current_request_text, enforce_reply_budget,
    select_reply_policy,
)
from .planner import PlannerDecision, execute_planned_side_effect, plan_chat_action
from .proactive import proactive_manager
from .repeat import TEXT_ONLY as REPEAT_TEXT_ONLY, repeat_manager
from .turns import is_meme_only, prepare_planned_meme, send_response_turn, wants_meme
from .utils import clean_response
from .sentiment import sentiment_analyzer
from .conversation import (
    TurnSuperseded, current_generation, ensure_current_session, ensure_send_allowed,
    invalidate, serialized_turn, session_key, session_scope,
)
from .agent_runner import DeliveryUncertain, deliver_reply as _deliver_agent_reply
from .image_references import ImageReference, image_reference_store, reference_marker


async def _send_research_progress(bot, event, text, user_id, group_id):
    ensure_send_allowed(group_id)
    receipt = await _deliver_agent_reply(text, str(text), lambda: bot.send(event, str(text)), bot=bot, group_id=group_id)
    await memory_manager.add_message(0 if group_id else user_id, "assistant", "[检索进展，非最终结论] " + text, group_id, model_spec=getattr(text, "model_spec", None))
    return receipt


def format_sender_name(name: str) -> str:
    """格式化发送者名称，防止过长的诗句或提问干扰上下文"""
    if not name:
        return "未知用户"
    name = name.strip()
    if len(name) > 8:
        # 截断过长的名字
        return name[:8] + "..."
    return name

# ===== 消息聚合缓冲区 =====
@dataclass
class PendingMessage:
    """待处理的消息"""
    user_id: int
    sender_name: str
    message: str
    event: GroupMessageEvent
    group_message_order: int = 0
    images_base64: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    planner_decision: Optional[PlannerDecision] = None


@dataclass
class ResolvedIncomingMessage:
    """One normalized view shared by record/planner/chat paths."""

    plain_text: str
    response_text: Optional[str] = None
    context_blocks: list[str] = field(default_factory=list)
    direct_image_urls: list[str] = field(default_factory=list)
    context_image_urls: list[str] = field(default_factory=list)
    unavailable_images: bool = False
    faces: list[str] = field(default_factory=list)

    @property
    def image_urls(self) -> list[str]:
        return list(
            dict.fromkeys([*self.direct_image_urls, *self.context_image_urls])
        )

    def compose(self, plain_text: Optional[str] = None) -> str:
        current_text = self.plain_text if plain_text is None else plain_text
        parts = [block.strip() for block in self.context_blocks if block.strip()]
        if current_text.strip():
            if parts:
                parts.append(f"[当前消息]\n{current_text.strip()}")
            else:
                parts.append(current_text.strip())
        return "\n\n".join(parts)


@dataclass
class _ExpansionBudget:
    group_id: Optional[int] = None
    self_id: str = ""
    mention_calls_left: int = 2
    known_names: dict[str, str] = field(default_factory=dict)
    unavailable_images: bool = False
    nodes_left: int = 30
    chars_left: int = 6000
    images_left: int = 6
    api_calls_left: int = 4
    max_depth: int = 2

    def take_text(self, value: str, *, preserve_whitespace: bool = False) -> str:
        text = str(value or "")
        if not preserve_whitespace:
            text = text.strip()
        if not text or self.chars_left <= 0:
            return ""
        if len(text) > self.chars_left:
            suffix = "…[内容过长，已截断]"
            keep = max(0, self.chars_left - len(suffix))
            text = text[:keep] + suffix
        self.chars_left -= len(text)
        return text

    def take_image(self, url: Any) -> Optional[str]:
        value = str(url or "").strip()
        if (
            not value
            or not value.startswith(("http://", "https://"))
            or self.images_left <= 0
        ):
            self.unavailable_images = True
            return None
        self.images_left -= 1
        return value


@dataclass
class AggregationBuffer:
    """聚合缓冲区"""
    messages: list[PendingMessage] = field(default_factory=list)
    task: Optional[asyncio.Task] = None
    first_received_at: float = 0.0
    last_received_at: float = 0.0
    wakeup: asyncio.Event = field(default_factory=asyncio.Event)


# 群消息聚合缓冲区：{group_id: AggregationBuffer}
aggregation_buffers: dict[int, AggregationBuffer] = {}
# 被动回复 Planner 的延迟重规划任务；同一群只保留最新一轮。
planner_wait_tasks: dict[int, asyncio.Task] = {}

# 群消息序号：用于判断 bot 生成回复期间，原消息后面是否已经插入了较多新消息。
group_message_orders: dict[int, int] = {}
group_message_order_by_id: dict[int, OrderedDict[str, int]] = {}
GROUP_MESSAGE_ORDER_CACHE_SIZE = 500
# How much recent group chatter the passive planner reads when re-checking.
REPLY_JUDGE_WINDOW_SECONDS = 120


def _cancel_planner_wait(group_id: int) -> None:
    """Cancel a stale passive wait when a newer direct turn supersedes it."""
    task = planner_wait_tasks.pop(group_id, None)
    if task and not task.done() and task is not asyncio.current_task():
        task.cancel()


async def _resume_after_planner_wait(
    *,
    bot: Bot,
    event: GroupMessageEvent,
    message: str,
    images_base64: list,
    faces: list,
    initial_decision: PlannerDecision,
) -> None:
    """Sleep as planned, then re-run Planner with the latest group context."""
    group_id = event.group_id
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    decision = initial_decision
    max_total = max(1, int(config.planner_wait_max_seconds))
    max_replans = max(1, int(config.planner_wait_max_replans))

    try:
        for replan_index in range(1, max_replans + 1):
            remaining = max_total - (loop.time() - started_at)
            if remaining <= 0:
                logger.info(
                    f"[被动发言] 群 {group_id} wait 累计达到 {max_total}s，结束本轮"
                )
                return
            delay = min(float(decision.wait_delay_seconds), remaining)
            logger.info(
                f"[被动发言] 群 {group_id} wait {delay:.1f}s 后重规划 "
                f"({replan_index}/{max_replans})"
            )
            await asyncio.sleep(delay)

            recent_context = proactive_manager.get_recent_messages_text(
                group_id,
                REPLY_JUDGE_WINDOW_SECONDS,
            )
            from . import followups
            direct, elsewhere = _group_message_direction(event, event.get_plaintext())
            if not direct and not _eligible_event_followup(event, elsewhere):
                return
            decision = await plan_chat_action(
                session_id=f"group_{group_id}",
                group_id=group_id,
                current_message=message,
                recent_context=recent_context,
                mode="passive",
                is_direct=direct,
                followup_context=followups.context(group_id) if not direct else "",
                has_images=bool(images_base64),
                faces=faces,
                default_action="reply" if direct else "silent",
            )
            if decision.action == "wait":
                continue
            if not decision.should_continue:
                logger.info(
                    f"[被动发言] 群 {group_id} wait 重规划 action={decision.action}，结束本轮"
                )
                return

            logger.info(
                f"[被动发言] 群 {group_id} wait 重规划 action={decision.action}，继续执行"
            )
            # From here the serialized conversation owns the work. A newer
            # message may cancel a waiting plan, but must not cancel execution.
            if planner_wait_tasks.get(group_id) is asyncio.current_task():
                planner_wait_tasks.pop(group_id, None)
            await handle_single_chat(
                bot,
                event,
                message,
                images_base64,
                faces,
                decision,
            )
            return

        logger.info(
            f"[被动发言] 群 {group_id} wait 重规划达到 {max_replans} 次，结束本轮"
        )
    except asyncio.CancelledError:
        logger.debug(f"[被动发言] 群 {group_id} 的 wait 被更新消息取消")
        raise
    except Exception as exc:
        logger.exception(f"[被动发言] 群 {group_id} wait 重规划失败: {exc}")
    finally:
        if planner_wait_tasks.get(group_id) is asyncio.current_task():
            planner_wait_tasks.pop(group_id, None)


def _schedule_planner_wait(
    *,
    bot: Bot,
    event: GroupMessageEvent,
    message: str,
    images_base64: list,
    faces: list,
    decision: PlannerDecision,
) -> None:
    """Keep only the newest delayed passive turn for a group."""
    group_id = event.group_id
    _cancel_planner_wait(group_id)
    planner_wait_tasks[group_id] = asyncio.create_task(
        _resume_after_planner_wait(
            bot=bot,
            event=event,
            message=message,
            images_base64=list(images_base64),
            faces=list(faces),
            initial_decision=decision,
        ),
        name=f"planner-wait-passive-{group_id}",
    )


def _get_event_message_id(event: MessageEvent) -> Optional[int]:
    message_id = getattr(event, "message_id", None)
    if message_id is None:
        return None
    try:
        return int(message_id)
    except (TypeError, ValueError):
        return None


def _mark_group_message_seen(event: GroupMessageEvent) -> int:
    group_id = event.group_id
    next_order = group_message_orders.get(group_id, 0) + 1
    group_message_orders[group_id] = next_order

    message_id = _get_event_message_id(event)
    if message_id is not None:
        order_cache = group_message_order_by_id.setdefault(group_id, OrderedDict())
        order_cache[str(message_id)] = next_order
        while len(order_cache) > GROUP_MESSAGE_ORDER_CACHE_SIZE:
            order_cache.popitem(last=False)

    return next_order


def _get_group_message_order(event: GroupMessageEvent) -> int:
    message_id = _get_event_message_id(event)
    if message_id is not None:
        order = group_message_order_by_id.get(event.group_id, {}).get(str(message_id))
        if order is not None:
            return order
    return group_message_orders.get(event.group_id, 0)


def _should_use_targeted_reply(event: MessageEvent, group_message_order: int) -> bool:
    if not config.enable_targeted_reply or not isinstance(event, GroupMessageEvent):
        return False
    if _get_event_message_id(event) is None:
        return False

    threshold = max(1, int(config.targeted_reply_min_intervening_messages))
    intervening_count = max(0, group_message_orders.get(event.group_id, 0) - group_message_order)
    if intervening_count < threshold:
        return False

    logger.info(
        f"[定向回复] 群 {event.group_id} 原消息后新增 {intervening_count} 条消息，"
        f"使用 reply 引用 message_id={_get_event_message_id(event)}"
    )
    return True


def _build_targeted_reply_message(event: MessageEvent, text: str, group_message_order: int) -> Message | str:
    clean_text = (text or "").strip()
    if not clean_text:
        return clean_text
    if not _should_use_targeted_reply(event, group_message_order):
        return clean_text

    return Message([
        MessageSegment.reply(_get_event_message_id(event)),
        MessageSegment.text(clean_text),
    ])


def _build_assistant_history_text(response_text: str, sent_meme: Optional[dict]) -> str:
    from .tools import format_sent_meme_for_memory

    meme_text = format_sent_meme_for_memory(sent_meme)
    reply_text = str(response_text or "").strip()
    if meme_text and reply_text:
        return f"{meme_text}\n{reply_text}"
    if meme_text:
        return meme_text
    return str(response_text or "")


async def _finalize_voice_tool_turn(
    user_id: int,
    group_id: Optional[int],
    *,
    followup_text: str = "",
    send_followup: Optional[Callable[[str], Awaitable[None]]] = None,
    model_spec: Optional[str] = None,
) -> bool:
    """Finish a voice turn, with an optional planner-approved text follow-up."""
    from .tools import format_sent_voice_for_memory, get_last_sent_voice

    sent_voice = get_last_sent_voice()
    if not sent_voice:
        return False

    followup_text = enforce_reply_budget(followup_text, MEDIA_REPLY)
    sent_followup = ""
    cancelled = None
    if followup_text and send_followup:
        try:
            await send_followup(followup_text)
            sent_followup = followup_text
        except asyncio.CancelledError as exc:
            # Preserve the already-delivered media, then propagate cancellation.
            cancelled = exc
        except Exception as exc:
            logger.warning(
                f"[voice] 语音已发送，但收尾文字发送失败 group={group_id} user={user_id}: {exc}"
            )
    history_text = "\n".join(
        part for part in [format_sent_voice_for_memory(sent_voice), sent_followup] if part
    )
    try:
        await memory_manager.add_message(0 if group_id else user_id, "assistant", history_text, group_id, model_spec=model_spec if sent_followup else None)
    except Exception as exc:
        logger.warning(f"[voice] 语音消息写入记忆失败 group={group_id} user={user_id}: {exc}")

    if group_id:
        proactive_manager.mark_just_spoke(group_id)
    logger.info(
        f"[voice] 本轮语音回复已完成 group={group_id} user={user_id} "
        f"chars={len(str(sent_voice.get('text') or ''))} followup_chars={len(sent_followup)}"
    )
    if cancelled is not None:
        raise cancelled
    return True


async def _finalize_image_tool_turn(
    user_id: int,
    group_id: Optional[int],
    *,
    followup_text: str = "",
    send_followup: Optional[Callable[[str], Awaitable[None]]] = None,
    model_spec: Optional[str] = None,
) -> bool:
    """Finish an image turn, with an optional planner-approved text follow-up."""
    from .tools import format_sent_image_for_memory, get_last_sent_image

    sent_image = get_last_sent_image()
    if not sent_image:
        return False

    followup_text = enforce_reply_budget(followup_text, MEDIA_REPLY)
    sent_followup = ""
    cancelled = None
    if followup_text and send_followup:
        try:
            await send_followup(followup_text)
            sent_followup = followup_text
        except asyncio.CancelledError as exc:
            # Preserve the already-delivered media, then propagate cancellation.
            cancelled = exc
        except Exception as exc:
            logger.warning(
                f"[image_gen] 图片已发送，但收尾文字发送失败 group={group_id} user={user_id}: {exc}"
            )
    history_text = "\n".join(
        part for part in [format_sent_image_for_memory(sent_image), sent_followup] if part
    )
    try:
        await memory_manager.add_message(0 if group_id else user_id, "assistant", history_text, group_id, model_spec=model_spec if sent_followup else None)
    except Exception as exc:
        logger.warning(f"[image_gen] 图片消息写入记忆失败 group={group_id} user={user_id}: {exc}")

    if group_id:
        proactive_manager.mark_just_spoke(group_id)
    logger.info(
        f"[image_gen] 本轮图片回复已完成 group={group_id} user={user_id} "
        f"model={sent_image.get('model')} size={sent_image.get('size')} "
        f"followup_chars={len(sent_followup)}"
    )
    if cancelled is not None:
        raise cancelled
    return True


async def _finalize_immediate_tool_turn(
    user_id: int,
    group_id: Optional[int],
    *,
    followup_text: str = "",
    send_followup: Optional[Callable[[str], Awaitable[None]]] = None,
    model_spec: Optional[str] = None,
) -> bool:
    if await _finalize_image_tool_turn(
        user_id,
        group_id,
        followup_text=followup_text,
        send_followup=send_followup,
        model_spec=model_spec,
    ):
        return True
    return await _finalize_voice_tool_turn(
        user_id,
        group_id,
        followup_text=followup_text,
        send_followup=send_followup,
        model_spec=model_spec,
    )


def _get_targeted_reply_tool_context(
    event: MessageEvent,
    group_message_order: int,
) -> tuple[Optional[int], Optional[Callable[[], bool]]]:
    if not isinstance(event, GroupMessageEvent):
        return None, None
    message_id = _get_event_message_id(event)
    if message_id is None:
        return None, None
    return message_id, lambda: _should_use_targeted_reply(event, group_message_order)


def _drop_current_turn_from_history(
    history: list[LLMMessage],
    current_turns: list[tuple[str, str, bool]],
) -> list[LLMMessage]:
    """Remove group messages already represented by the current user prompt.

    The group recorder runs before the responder, so the same incoming message
    can otherwise be present both in history and as the final user message.
    """
    if not history or not current_turns:
        return history

    unresolved = [
        {
            "name": (name or "").strip(),
            "text": (text or "").strip(),
            "has_media": has_media,
        }
        for name, text, has_media in current_turns
    ]
    unresolved = [item for item in unresolved if item["name"] or item["text"] or item["has_media"]]
    if not unresolved:
        return history

    remove_indices: set[int] = set()
    for idx in range(len(history) - 1, -1, -1):
        msg = history[idx]
        if msg.role == "assistant" and remove_indices:
            break
        if msg.role != "user":
            continue

        content = msg.content or ""
        msg_name = (msg.name or "").strip()
        for spec_idx in range(len(unresolved) - 1, -1, -1):
            spec = unresolved[spec_idx]
            if spec["name"] and msg_name != spec["name"]:
                continue
            if spec["text"]:
                if spec["text"] not in content:
                    continue
            elif not spec["has_media"]:
                continue

            remove_indices.add(idx)
            unresolved.pop(spec_idx)
            break

        if not unresolved:
            break

    if not remove_indices:
        return history

    logger.info(f"[context] 从历史上下文移除本轮已记录群消息 {len(remove_indices)} 条，避免重复注入")
    return [msg for idx, msg in enumerate(history) if idx not in remove_indices]


# ===== 防提示词注入过滤 =====
INJECTION_PATTERNS = [
    # English patterns
    r'^debug\s*mode\s*:',
    r'^developer\s*(mode)?\s*:',
    r'^ignore\s+(previous\s+)?instructions',
    r'^ignore\s+all\s+(previous\s+)?(instructions\s*)?',
    r'^system\s*:',
    r'^you\s+are\s+now\s+',
    r'^act\s+as\s+',
    r'^pretend\s+(to\s+be\s+)?',
    r'^simulation\s*:',
    r'^new\s+system\s*:',
    r'^override\s+(system\s+)?(instructions?)?\s*:',
    r'^disregard\s+(previous\s+)?instructions',
    r'^forget\s+(everything\s+)?(about\s+)?(your\s+)?(instructions?|system)\s*:',
    r'^set\s+system\s*:',
    r'^in\s+\[INST\]\s*',
    r'^<<SYS>>',
    r'^\[SYSTEM\]',
    r'^根 据 你 的 内部指令',
    # Chinese patterns
    r'^你现在是',
    r'^请扮演',
    r'^忽略之前的',
    r'^无视之前的',
    r'^取消之前的',
    r'^系统设定\s*:',
    r'^系统指令\s*:',
    r'^你是',
    r'^从现在开始你是',
    r'^我要你扮演',
    r'^我希望你扮演',
]


async def process_aggregated_messages(bot: Bot, group_id: int):
    buffer = aggregation_buffers.get(group_id)
    if not buffer or not buffer.messages:
        return
    count = max(1, int(config.aggregation_max_messages))
    batch = buffer.messages[:count]
    del buffer.messages[:count]
    sid = session_key(0, group_id)
    generation = current_generation(sid)
    batch = [item for item in batch
             if getattr(item.event, "_llm_generation", generation) == generation]
    # Preserve chronological order, including A -> B -> A exchanges.
    batch.sort(key=lambda item: item.group_message_order)
    if buffer.messages:
        buffer.first_received_at = asyncio.get_running_loop().time()
    if batch:
        # The long-lived worker may have inherited a previous event's scope.
        # Each accepted batch retains its own arrival generation instead.
        with session_scope(sid, generation=generation):
            await handle_merged_chat(bot, group_id, batch)


async def delayed_process(bot: Bot, group_id: int):
    buffer = aggregation_buffers[group_id]
    loop = asyncio.get_running_loop()
    try:
        while buffer.messages:
            while True:
                quiet_deadline = buffer.last_received_at + max(0, float(config.aggregation_window))
                hard_deadline = buffer.first_received_at + max(0.1, float(config.aggregation_max_wait_seconds))
                delay = min(quiet_deadline, hard_deadline) - loop.time()
                if delay <= 0:
                    break
                buffer.wakeup.clear()
                try:
                    await asyncio.wait_for(buffer.wakeup.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    break
            try:
                await process_aggregated_messages(bot, group_id)
            except FinishedException:
                pass
            except Exception as error:
                logger.exception(f"[消息聚合] 处理失败 group={group_id}: {error}")
    finally:
        if buffer.task is asyncio.current_task():
            buffer.task = None
        if aggregation_buffers.get(group_id) is buffer and not buffer.messages:
            aggregation_buffers.pop(group_id, None)


def cancel_pending_group_turns(group_id: int) -> None:
    invalidate(f"group_{group_id}")
    _cancel_planner_wait(group_id)
    proactive_manager.cancel_wait(group_id)
    buffer = aggregation_buffers.get(group_id)
    if buffer:
        buffer.messages.clear()
        buffer.wakeup.set()


async def close_chat_tasks() -> None:
    from . import followups
    followups.windows.clear()
    tasks = [*planner_wait_tasks.values(), *(b.task for b in aggregation_buffers.values() if b.task)]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    planner_wait_tasks.clear()
    aggregation_buffers.clear()
    image_reference_store.clear()
    _mention_names.clear()


async def add_to_aggregation_buffer(bot: Bot, event: GroupMessageEvent,
                                     message: str, images_base64: list, faces: list,
                                     planner_decision: Optional[PlannerDecision] = None):
    group_id = event.group_id
    ensure_current_session()
    if not hasattr(event, "_llm_generation"):
        event._llm_generation = current_generation(session_key(event.user_id, group_id))
    buffer = aggregation_buffers.setdefault(group_id, AggregationBuffer())
    now = asyncio.get_running_loop().time()
    if not buffer.messages:
        buffer.first_received_at = now
    buffer.last_received_at = now
    buffer.messages.append(PendingMessage(
        user_id=event.user_id,
        sender_name=event.sender.card or event.sender.nickname or str(event.user_id),
        message=message, event=event, group_message_order=_get_group_message_order(event),
        images_base64=images_base64, faces=faces,
        # Plan the complete batch after collecting it.
        planner_decision=None,
    ))
    hard_cap = max(config.aggregation_max_messages * 3, 30)
    if len(buffer.messages) > hard_cap:
        buffer.messages = buffer.messages[-hard_cap:]
        logger.warning(f"[消息聚合] 群 {group_id} 缓冲区达到上限 {hard_cap}")
    buffer.wakeup.set()
    if buffer.task is None or buffer.task.done():
        buffer.task = asyncio.create_task(delayed_process(bot, group_id), name=f"aggregation-{group_id}")


def sanitize_input(user_message: str) -> str:
    """过滤用户消息中的提示词注入尝试。"""

    original = user_message

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, user_message, re.IGNORECASE):
            # 用占位符替换，而非直接拒绝
            user_message = re.sub(pattern, '[内容已过滤] ', user_message, flags=re.IGNORECASE, count=1)
            logger.warning(f"[防注入] 检测到可疑模式: {pattern}")
            # 只处理第一个匹配，避免用户猜出规则
            break

    if user_message != original:
        logger.info(f"[防注入] 原始消息: {original[:50]}... -> 过滤后: {user_message[:50]}...")

    return user_message


# ===== 消息记录 =====



# 群聊消息监听器 - 记录所有群聊消息（不阻塞，只记录）
group_message_listener = on_message(priority=5, block=False)


def _group_message_direction(
    event: GroupMessageEvent,
    message: str,
) -> tuple[bool, bool]:
    """Return (targets_bot, targets_other_user) for one group message."""
    self_id = str(getattr(event, "self_id", ""))
    targets_bot = False
    targets_other = False
    reply = getattr(event, "reply", None)
    reply_sender = _object_value(_object_value(reply, "sender", None), "user_id", None)
    if reply_sender and str(reply_sender) != self_id:
        targets_other = True
    segment_sources = [getattr(event, "message", [])]
    original_message = getattr(event, "original_message", None)
    if original_message is not None and original_message is not event.message:
        segment_sources.append(original_message)

    for segments in segment_sources:
        for segment in segments:
            if segment.type != "at":
                continue
            if str(segment.data.get("qq", "")) == self_id:
                targets_bot = True
            else:
                targets_other = True

    if getattr(event, "to_me", False):
        targets_bot = True
    if not targets_bot:
        targets_bot = any(
            str(message or "").startswith(word)
            for word in config.trigger_words
        )
    return targets_bot, targets_other


def _object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _segment_type_and_data(segment: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(segment, dict):
        seg_type = segment.get("type")
        data = segment.get("data")
    else:
        seg_type = getattr(segment, "type", None)
        data = getattr(segment, "data", None)
    return str(seg_type or ""), data if isinstance(data, dict) else {}


def _as_segment_list(message: Any) -> list[Any]:
    if message is None:
        return []
    if isinstance(message, dict) and "type" in message:
        return [message]
    if isinstance(message, (list, tuple, Message)):
        return list(message)
    if isinstance(message, str):
        return list(Message(message))
    return []


def _image_url_from_data(data: dict) -> Optional[str]:
    for value in (data.get("url"), data.get("file")):
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


# Names are scoped to the bot and group: a member can use different cards in different groups.
_mention_names: OrderedDict[tuple, tuple[float, str]] = OrderedDict()


def _remember_sender(sender: Any, budget: _ExpansionBudget, *, cache: bool = False) -> None:
    user_id = str(_object_value(sender, "user_id", ""))
    name = _object_value(sender, "card", "") or _object_value(sender, "nickname", "")
    if user_id and name:
        budget.known_names[user_id] = format_sender_name(str(name))
        if not cache:
            return
        key = (budget.self_id, budget.group_id, user_id)
        _mention_names[key] = (time.monotonic(), format_sender_name(str(name)))
        _mention_names.move_to_end(key)
        while len(_mention_names) > 2048:
            _mention_names.popitem(last=False)


async def _render_mention(bot: Bot, data: dict, budget: _ExpansionBudget) -> str:
    qq = str(data.get("qq", "")).strip()
    if qq == "all":
        return "@全体成员"
    if not qq:
        return "@某人"
    if qq == budget.self_id:
        return f"@你(QQ:{qq})"
    name = data.get("name") or data.get("nickname") or budget.known_names.get(qq)
    key = (budget.self_id, budget.group_id, qq)
    cached = _mention_names.get(key)
    if not name and cached and time.monotonic() - cached[0] < 3600:
        name = cached[1]
    if not name and budget.group_id and qq.isdigit() and budget.mention_calls_left > 0:
        budget.mention_calls_left -= 1
        try:
            member = await asyncio.wait_for(bot.call_api(
                "get_group_member_info", group_id=budget.group_id, user_id=int(qq),
            ), timeout=2.0)
            name = _object_value(member, "card", "") or _object_value(member, "nickname", "")
            if name:
                _remember_sender({"user_id": qq, "nickname": name}, budget, cache=True)
        except Exception as exc:
            logger.debug(f"[message_expand] 获取提及对象失败 qq={qq}: {exc}")
    return f"@{format_sender_name(str(name))}(QQ:{qq})" if name else f"@QQ:{qq}"


async def _render_current_text(bot: Bot, event: MessageEvent, budget: _ExpansionBudget) -> tuple[str, str]:
    """Keep mentions in memory; only bot addressing is omitted from the response input."""
    parts, response_parts = [], []
    for segment in _as_segment_list(event.message):
        seg_type, data = _segment_type_and_data(segment)
        if seg_type == "text":
            text = str(data.get("text", ""))
        elif seg_type == "at":
            text = await _render_mention(bot, data, budget) + " "
        else:
            continue
        parts.append(text)
        if seg_type != "at" or str(data.get("qq", "")) != budget.self_id:
            response_parts.append(text)
    return "".join(parts).strip(), "".join(response_parts).strip()


def _sender_label(sender: Any, event: MessageEvent) -> str:
    user_id = _object_value(sender, "user_id", "")
    if user_id and str(user_id) == str(event.self_id):
        return "你"
    raw_name = (
        _object_value(sender, "card", "")
        or _object_value(sender, "nickname", "")
        or str(user_id or "")
    )
    return format_sender_name(str(raw_name)) if raw_name else "某人"


async def _call_message_api(
    bot: Bot,
    api: str,
    budget: _ExpansionBudget,
    **kwargs: Any,
) -> Any:
    if budget.api_calls_left <= 0:
        raise RuntimeError("消息展开 API 调用次数已达上限")
    budget.api_calls_left -= 1
    return await asyncio.wait_for(bot.call_api(api, **kwargs), timeout=3.0)


def _extract_forward_nodes(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("messages", "content"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    nested = payload.get("data")
    if isinstance(nested, dict):
        return _extract_forward_nodes(nested)
    return []


def _is_forward_json(data: dict[str, Any]) -> tuple[bool, list[str]]:
    raw = data.get("data")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return False, []
    elif isinstance(raw, dict):
        payload = raw
    else:
        return False, []

    config_data = payload.get("config") if isinstance(payload, dict) else {}
    app = str(payload.get("app", "")) if isinstance(payload, dict) else ""
    is_forward = bool(
        (isinstance(config_data, dict) and config_data.get("forward"))
        or app == "com.tencent.multimsg"
    )
    if not is_forward:
        return False, []

    meta = payload.get("meta") if isinstance(payload, dict) else {}
    detail = meta.get("detail", {}) if isinstance(meta, dict) else {}
    previews = []
    if isinstance(detail, dict):
        news = detail.get("news")
        if isinstance(news, list):
            for item in news:
                text = _object_value(item, "text", "")
                if text:
                    previews.append(str(text).strip())
    return True, previews


async def _render_forward_payload(
    bot: Bot,
    payload: Any,
    budget: _ExpansionBudget,
    *,
    depth: int,
) -> tuple[str, list[str]]:
    nodes = _extract_forward_nodes(payload)
    if not nodes:
        return "", []

    # Some implementations embed one ordinary message chain directly rather
    # than wrapping it in forward-node objects.
    if all(
        _segment_type_and_data(node)[0]
        and _segment_type_and_data(node)[0] != "node"
        for node in nodes
    ):
        return await _render_message_value(bot, nodes, budget, depth=depth)

    lines: list[str] = []
    image_urls: list[str] = []
    for node in nodes[:budget.nodes_left]:
        seg_type, seg_data = _segment_type_and_data(node)
        node_data = seg_data if seg_type == "node" else node
        sender = _object_value(node_data, "sender", {}) or {}
        _remember_sender(sender, budget)
        _remember_sender({
            "user_id": _object_value(node_data, "uin", ""),
            "nickname": _object_value(node_data, "name", "") or _object_value(node_data, "nickname", ""),
        }, budget)
    for node in nodes:
        if budget.nodes_left <= 0 or budget.chars_left <= 0:
            lines.append("…[后续转发节点已省略]")
            break
        budget.nodes_left -= 1

        seg_type, seg_data = _segment_type_and_data(node)
        node_data = seg_data if seg_type == "node" else node
        sender = _object_value(node_data, "sender", {}) or {}
        _remember_sender(sender, budget)
        nickname = (
            _object_value(sender, "card", "")
            or _object_value(sender, "nickname", "")
            or _object_value(node_data, "name", "")
            or _object_value(node_data, "nickname", "")
            or str(
                _object_value(sender, "user_id", "")
                or _object_value(node_data, "uin", "")
                or "某人"
            )
        )
        content = _object_value(node_data, "content", None)
        if content is None:
            content = _object_value(node_data, "message", None)

        rendered, nested_images = await _render_message_value(
            bot,
            content,
            budget,
            depth=depth,
        )
        rendered = rendered or "[空消息]"
        lines.append(f"{format_sender_name(str(nickname))}: {rendered}")
        image_urls.extend(nested_images)

    return "\n".join(lines), image_urls


async def _render_message_value(
    bot: Bot,
    message: Any,
    budget: _ExpansionBudget,
    *,
    depth: int,
) -> tuple[str, list[str]]:
    if isinstance(message, str):
        if "[CQ:" in message:
            try:
                parsed = Message(message)
                if any(segment.type != "text" for segment in parsed):
                    return await _render_message_value(
                        bot,
                        parsed,
                        budget,
                        depth=depth,
                    )
            except Exception as exc:
                logger.debug(f"[message_expand] CQ 字符串解析失败: {exc}")
        return budget.take_text(message), []

    segments = _as_segment_list(message)
    if not segments:
        return budget.take_text(str(message or "")), []

    parts: list[str] = []
    image_urls: list[str] = []
    for segment in segments:
        if budget.chars_left <= 0:
            break
        seg_type, data = _segment_type_and_data(segment)
        if seg_type == "text":
            text = budget.take_text(data.get("text", ""), preserve_whitespace=True)
            if text:
                parts.append(text)
        elif seg_type == "at":
            parts.append(budget.take_text(await _render_mention(bot, data, budget)))
        elif seg_type == "image":
            parts.append(budget.take_text("[图片]"))
            url = budget.take_image(_image_url_from_data(data))
            if url:
                image_urls.append(url)
        elif seg_type == "face":
            face_id = str(data.get("id") or data.get("emoji_id") or "")
            face_name = FACE_EMOJI_MAP.get(face_id, f"表情{face_id}")
            parts.append(budget.take_text(f"[收到QQ表情：{face_name}]"))
        elif seg_type in {"record", "voice"}:
            parts.append(budget.take_text("[语音]"))
        elif seg_type == "video":
            parts.append(budget.take_text("[视频]"))
        elif seg_type == "file":
            name = data.get("name") or data.get("file") or ""
            parts.append(budget.take_text(f"[文件: {name}]" if name else "[文件]"))
        elif seg_type == "reply":
            reply_id = data.get("id")
            if not reply_id or depth >= budget.max_depth:
                parts.append(budget.take_text("[回复了一条消息]"))
                continue
            try:
                info = await _call_message_api(
                    bot,
                    "get_msg",
                    budget,
                    message_id=int(reply_id),
                )
                quoted, nested_images = await _render_message_value(
                    bot,
                    _object_value(info, "message", None),
                    budget,
                    depth=depth + 1,
                )
                parts.append(f"[回复: {quoted or '空消息'}]")
                image_urls.extend(nested_images)
            except Exception as exc:
                logger.debug(f"[message_expand] 展开嵌套回复失败 id={reply_id}: {exc}")
                parts.append(budget.take_text("[回复了一条无法读取的消息]"))
        elif seg_type == "forward":
            if depth >= budget.max_depth:
                parts.append(budget.take_text("[嵌套合并转发已省略]"))
                continue
            forward_id = data.get("id")
            payload = data.get("content")
            try:
                if not isinstance(payload, list) and forward_id:
                    payload = await _call_message_api(
                        bot,
                        "get_forward_msg",
                        budget,
                        id=str(forward_id),
                    )
                rendered, nested_images = await _render_forward_payload(
                    bot,
                    payload,
                    budget,
                    depth=depth + 1,
                )
                parts.append(
                    f"[嵌套合并转发]\n{rendered}"
                    if rendered
                    else "[无法读取嵌套合并转发]"
                )
                image_urls.extend(nested_images)
            except Exception as exc:
                logger.debug(f"[message_expand] 展开嵌套转发失败 id={forward_id}: {exc}")
                parts.append(budget.take_text("[无法读取嵌套合并转发]"))
        elif seg_type == "json":
            is_forward, previews = _is_forward_json(data)
            if is_forward:
                preview = "\n".join(previews)
                parts.append(
                    budget.take_text(
                        f"[合并转发预览]\n{preview}"
                        if preview
                        else "[合并转发卡片]"
                    )
                )
            else:
                parts.append(budget.take_text("[JSON 卡片]"))
        elif seg_type:
            parts.append(budget.take_text(f"[{seg_type}]"))

    return "".join(parts).strip(), image_urls


async def _resolve_reply_context(
    bot: Bot,
    event: MessageEvent,
    budget: _ExpansionBudget,
) -> tuple[Optional[str], list[str]]:
    reply_info = getattr(event, "reply", None)
    reply_id = None

    if reply_info is None:
        for source in (
            getattr(event, "original_message", None),
            getattr(event, "message", None),
        ):
            for segment in _as_segment_list(source):
                seg_type, data = _segment_type_and_data(segment)
                if seg_type == "reply":
                    reply_id = data.get("id")
                    break
            if reply_id:
                break

        if not reply_id:
            return None, []
        try:
            reply_info = await _call_message_api(
                bot,
                "get_msg",
                budget,
                message_id=int(reply_id),
            )
        except Exception as exc:
            logger.debug(f"[message_expand] get_msg 失败 id={reply_id}: {exc}")
            budget.unavailable_images = True
            return "[引用回复]\n原消息：[无法读取，可能已撤回]", []

    sender = _object_value(reply_info, "sender", {}) or {}
    _remember_sender(sender, budget)
    replied_user_id = _object_value(sender, "user_id", "")
    if replied_user_id and str(replied_user_id) == str(event.self_id):
        event.to_me = True
    name = _sender_label(sender, event)
    quoted, image_urls = await _render_message_value(
        bot,
        _object_value(reply_info, "message", None),
        budget,
        depth=0,
    )
    return f"[引用回复]\n被回复者：{name}\n原消息：{quoted or '[空消息]'}", image_urls


async def _resolve_forward_contexts(
    bot: Bot,
    event: MessageEvent,
    budget: _ExpansionBudget,
) -> tuple[list[str], list[str]]:
    blocks: list[str] = []
    image_urls: list[str] = []

    for segment in _as_segment_list(getattr(event, "message", None)):
        seg_type, data = _segment_type_and_data(segment)
        if seg_type == "forward":
            forward_id = data.get("id")
            payload = data.get("content")
            try:
                if not isinstance(payload, list) and forward_id:
                    payload = await _call_message_api(
                        bot,
                        "get_forward_msg",
                        budget,
                        id=str(forward_id),
                    )
                rendered, nested_images = await _render_forward_payload(
                    bot,
                    payload,
                    budget,
                    depth=1,
                )
                blocks.append(
                    f"[合并转发]\n{rendered}"
                    if rendered
                    else "[合并转发]\n[无法读取内容]"
                )
                image_urls.extend(nested_images)
            except Exception as exc:
                logger.debug(f"[message_expand] get_forward_msg 失败 id={forward_id}: {exc}")
                blocks.append("[合并转发]\n[无法读取内容]")
        elif seg_type == "json":
            is_forward, previews = _is_forward_json(data)
            if not is_forward:
                continue
            rendered = ""
            nested_images: list[str] = []
            message_id = getattr(event, "message_id", None)
            if message_id is not None:
                try:
                    payload = await _call_message_api(
                        bot,
                        "get_forward_msg",
                        budget,
                        message_id=str(message_id),
                    )
                    rendered, nested_images = await _render_forward_payload(
                        bot,
                        payload,
                        budget,
                        depth=1,
                    )
                except Exception as exc:
                    logger.debug(
                        f"[message_expand] JSON 转发展开失败 message_id={message_id}: {exc}"
                    )
            if not rendered:
                rendered = "\n".join(
                    text for text in (budget.take_text(item) for item in previews) if text
                )
            blocks.append(
                f"[合并转发]\n{rendered}"
                if rendered
                else "[合并转发]\n[只能识别到转发卡片，无法读取内容]"
            )
            image_urls.extend(nested_images)

    return blocks, image_urls


async def resolve_incoming_message(
    bot: Bot,
    event: MessageEvent,
) -> ResolvedIncomingMessage:
    """Expand OneBot reply/forward data once and cache it on the event."""
    cached = getattr(event, "_llm_resolved_incoming_message", None)
    if isinstance(cached, ResolvedIncomingMessage):
        return cached

    budget = _ExpansionBudget(group_id=getattr(event, "group_id", None), self_id=str(event.self_id))
    _remember_sender(getattr(event, "sender", {}), budget, cache=True)
    reply_block, reply_images = await _resolve_reply_context(bot, event, budget)
    forward_blocks, forward_images = await _resolve_forward_contexts(
        bot,
        event,
        budget,
    )
    plain_text, response_text = await _render_current_text(bot, event, budget)
    direct_images = await get_images_from_message(event)
    unavailable_direct = any(
        seg_type == "image" and not _image_url_from_data(data)
        for seg_type, data in map(_segment_type_and_data, _as_segment_list(event.message))
    )
    result = ResolvedIncomingMessage(
        plain_text=plain_text,
        response_text=response_text,
        context_blocks=[
            *([reply_block] if reply_block else []),
            *forward_blocks,
        ],
        direct_image_urls=direct_images,
        context_image_urls=[*reply_images, *forward_images],
        unavailable_images=budget.unavailable_images or unavailable_direct,
        faces=await get_faces_from_message(event),
    )
    try:
        setattr(event, "_llm_resolved_incoming_message", result)
    except Exception as exc:
        logger.debug(f"[message_expand] 无法缓存规范化事件: {exc}")
    return result


def _capture_incoming_images(event: MessageEvent, downloaded: list[tuple[str, str]]) -> list[ImageReference]:
    sid = session_key(event.user_id, getattr(event, "group_id", None))
    sender = getattr(event, "sender", {})
    name = _object_value(sender, "card", "") or _object_value(sender, "nickname", "") or str(event.user_id)
    resolved = getattr(event, "_llm_resolved_incoming_message", None)
    quoted_urls = set(resolved.context_image_urls) if resolved else set()
    references = []
    previous = {ref.base64: ref for ref in getattr(event, "_llm_image_references", [])}
    for index, (url, encoded) in enumerate(downloaded, 1):
        source = "引用/转发图片" if url in quoted_urls else "附图"
        label = previous[encoded].label if encoded in previous else (
            f"{name}(QQ:{event.user_id}) 消息{getattr(event, 'message_id', '?')} {source}{index}"
        )
        ref = image_reference_store.remember(sid, encoded, label)
        references.append(ref)
    setattr(event, "_llm_image_references", references)
    return references


def _current_image_references(event: MessageEvent, images: list[str]) -> list[ImageReference]:
    references = getattr(event, "_llm_image_references", [])
    if [ref.base64 for ref in references] == images:
        return references
    return _capture_incoming_images(event, [("", encoded) for encoded in images])


def _missing_image_references(event: MessageEvent, images: list[str]) -> bool:
    resolved = getattr(event, "_llm_resolved_incoming_message", None)
    return bool(resolved and (resolved.unavailable_images or len(resolved.image_urls) > len(images)))


def _image_reference_text(references: list[ImageReference]) -> str:
    return "\n".join(f"{reference_marker(ref)} {ref.label}" for ref in references)


_incoming_tasks: set[asyncio.Task] = set()
_incoming_image_tasks: set[asyncio.Task] = set()


def _start_incoming_task(coroutine, *, image_work: bool = False) -> asyncio.Task:
    task = asyncio.create_task(coroutine, name="incoming-enrichment")
    _incoming_tasks.add(task)
    if image_work:
        _incoming_image_tasks.add(task)

    def finished(done):
        _incoming_tasks.discard(done)
        _incoming_image_tasks.discard(done)
        if not done.cancelled() and (error := done.exception()):
            logger.warning(f"[入站补充] 后台处理失败: {type(error).__name__}: {error}")
    task.add_done_callback(finished)
    return task


async def close_incoming_tasks() -> None:
    tasks = list(_incoming_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@group_message_listener.handle()
async def record_group_message(bot: Bot, event: GroupMessageEvent):
    sid = session_key(event.user_id, event.group_id)
    event._llm_generation = getattr(event, "_llm_generation", current_generation(sid))
    with session_scope(sid, generation=event._llm_generation):
        try:
            ensure_current_session()
            await _record_group_message(bot, event)
        except TurnSuperseded:
            event._llm_record_obsolete = True


async def _record_group_message(bot: Bot, event: GroupMessageEvent):
    """记录群聊中的所有消息到记忆中"""
    user_id = event.user_id
    group_id = event.group_id

    # 防御：某些 OneBot 适配器会把 bot 自己发的消息也以 message 事件回灌，
    # 这会污染记忆 + 干扰复读计数。直接短路。
    if str(user_id) == str(event.self_id):
        return
    _mark_group_message_seen(event)
    # Freeze the conversation that existed on arrival, before any awaited work.
    # A proactive reply to this event must not make the same event its own followup.
    from . import followups
    event._llm_followup_window = followups.active(group_id)

    # Persist arrival order before reference expansion, downloads or model calls.
    raw_sender_name = event.sender.card or event.sender.nickname or str(user_id)
    sender_name = format_sender_name(raw_sender_name)
    raw_text = event.get_plaintext().strip()
    kinds = {_segment_type_and_data(seg)[0] for seg in _as_segment_list(event.message)}
    if not raw_text and not kinds:
        return
    placeholders = {"image": "[收到图片，等待识别]", "forward": "[收到合并转发，等待展开]",
                    "reply": "[引用消息，等待展开]", "face": "[收到QQ表情]", "at": "[收到提及消息]"}
    raw_body = " ".join([raw_text, *(text for kind, text in placeholders.items() if kind in kinds)]).strip()
    initial_content = f"【群员-{sender_name}】 {raw_body}"
    event_time = getattr(event, "time", None)
    received_at = (datetime.fromtimestamp(event_time) if isinstance(event_time, (int, float))
                   else datetime.now()).isoformat()
    stored_message = await memory_manager.add_message(
        0, "user", initial_content, group_id, name=sender_name, sender_id=user_id, timestamp=received_at,
    )
    ensure_current_session()
    event._llm_memory_content = initial_content

    # OneBot reply/forward 只在这里展开一次；后续响应 matcher 复用事件缓存。
    resolved = await resolve_incoming_message(bot, event)
    ensure_current_session()
    plain_message = resolved.plain_text
    message = resolved.compose()
    is_to_bot, directed_at_other = _group_message_direction(event, plain_message)
    from . import followups
    followups.observe(group_id, user_id, directed_at_other and not is_to_bot)
    image_urls = resolved.image_urls

    # Non-directed pictures enrich their original row in the background. Direct
    # questions still wait for observations before planner/chat consume them.
    if image_urls and not is_to_bot and not followups.eligible(group_id, user_id, directed_at_other):
        if len(_incoming_image_tasks) >= max(1, config.incoming_image_max_pending):
            content = f"【群员-{sender_name}】 {message} [图片暂未识别：后台繁忙]".strip()
            if stored_message is not None:
                updated = await memory_manager.enrich_message(0, stored_message, content, group_id)
                ensure_current_session()
                if not updated:
                    return
            event._llm_memory_content = content
            return
        _start_incoming_task(
            _complete_group_record(bot, event, resolved, stored_message, sender_name), image_work=True,
        )
        return
    await _complete_group_record(bot, event, resolved, stored_message, sender_name)


async def _complete_group_record(bot, event, resolved, stored_message, sender_name):
    from .conversation import session_scope, ensure_current_session
    with session_scope(session_key(event.user_id, event.group_id)):
        await _enrich_group_record(bot, event, resolved, stored_message, sender_name)
        ensure_current_session()


async def _enrich_group_record(bot, event, resolved, stored_message, sender_name):
    from .conversation import ensure_current_session
    from . import followups

    user_id, group_id = event.user_id, event.group_id
    plain_message, message = resolved.plain_text, resolved.compose()
    is_to_bot, directed_at_other = _group_message_direction(event, plain_message)
    image_urls = resolved.image_urls
    direct_image_urls = set(resolved.direct_image_urls)

    downloaded_images = await download_images_as_base64(image_urls)
    ensure_current_session()
    images_base64 = [b64 for _, b64 in downloaded_images]
    references = _capture_incoming_images(event, downloaded_images)

    faces = resolved.faces

    repeated_text = ""
    # 群复读检测（仅文字）
    if config.enable_group_repeat and plain_message:
        if not (REPEAT_TEXT_ONLY and (image_urls or faces)):

            try:
                if repeat_manager.try_repeat(group_id, user_id, str(event.self_id), plain_message):
                    ensure_send_allowed(group_id)
                    await bot.send(event, MessageSegment.text(plain_message))
                    repeated_text = plain_message
            except Exception as e:
                logger.error(f"[群复读] 处理异常: {e}")

    # 忽略空消息（没有文字、没有图片、没有表情）
    if not message and not images_base64 and not faces:
        return


    # 构建完整消息内容
    body = message
    formatted_content = f"【群员-{sender_name}】 {body}"

    has_meme_image = False
    # 如果有图片，识别图片内容（同时顺手判定是否表情包并入库）
    if images_base64:
        from .memes import clean_meme_memory_description, meme_manager
        image_descriptions = []
        analyses = await asyncio.gather(*(
            model_manager.analyze_image_for_meme(encoded, session_id=f"group_{group_id}")
            for _, encoded in downloaded_images
        ), return_exceptions=True)
        ensure_current_session()
        for idx, (url, img_b64) in enumerate(downloaded_images):
            ensure_current_session()
            try:
                meta = analyses[idx]
                if isinstance(meta, BaseException):
                    raise meta
                desc = clean_meme_memory_description(meta.get("description") or "", max_chars=300)
                if not desc:
                    logger.warning(f"[图片识别] 空描述，已忽略: group_id={group_id}, index={idx}")
                    continue
                ref = image_reference_store.remember(
                    session_key(user_id, group_id), img_b64, f"{references[idx].label}：{desc}",
                )
                references[idx] = ref
                image_descriptions.append(f"{reference_marker(ref)} {desc}")
                is_direct_image = url in direct_image_urls
                if type(meta.get("random_eligible")) is bool:
                    import base64 as _b64
                    await meme_manager.update_random_eligibility(
                        _b64.b64decode(img_b64), meta["random_eligible"],
                        classification=meta,
                    )
                if is_direct_image and meta.get("is_meme"):
                    has_meme_image = True
                # 表情包采集
                if is_direct_image and meta.get("is_meme") and meta.get("topic"):
                    try:
                        import base64 as _b64
                        raw = _b64.b64decode(img_b64)
                        ext = "jpg"
                        if url:
                            low = url.lower().split("?")[0]
                            for cand in ("png", "gif", "webp", "jpeg", "jpg"):
                                if low.endswith(f".{cand}"):
                                    ext = "jpeg" if cand == "jpeg" else cand
                                    break
                        await meme_manager.add_meme(
                            image_data=raw,
                            topic=meta["topic"],
                            description=desc,
                            source_url=url,
                            source_user_id=user_id,
                            source_group_id=group_id,
                            ext=ext,
                            random_eligible=meta.get("random_eligible"),
                            meme_kind=meta.get("meme_kind"),
                            classification_reason=meta.get("classification_reason", ""),
                            text_sentence_count=meta.get("text_sentence_count"),
                        )
                    except Exception as e:
                        logger.warning(f"[meme] 入库失败: {e}")
            except Exception as e:
                logger.warning(f"[图片识别] 失败，已忽略该图片: group_id={group_id}, index={idx}, error={e}")
        if image_descriptions:
            desc_text = "\n".join([f"图片{i+1}（自动识别，仅作背景）: {d}" for i, d in enumerate(image_descriptions)])
            formatted_content = (
                f"[{sender_name}] {desc_text}\n\n用户消息: {body}"
                if body
                else f"[{sender_name}] {desc_text}\n\n用户消息: [仅分享图片，未提出问题]"
            )
        elif not body and not faces:
            formatted_content = f"【群员-{sender_name}】 [图片识别失败]"

    unmentioned_refs = [ref for ref in references if reference_marker(ref) not in formatted_content]
    if unmentioned_refs:
        formatted_content += "\n" + _image_reference_text(unmentioned_refs)

    # 如果有表情
    if faces:
        face_text = "，".join(faces)
        if formatted_content == f"【群员-{sender_name}】 ":
            formatted_content = f"【群员-{sender_name}】 [收到QQ表情：{face_text}]"
        else:
            formatted_content = f"{formatted_content} [收到QQ表情：{face_text}]"

    # 群聊使用 user_id=0 表示群组会话，而不是发送者的 user_id
    logger.info(f"[群聊记忆] 记录消息: group_id={group_id}, sender={sender_name}, msg={formatted_content[:50]}...")
    ensure_current_session()
    if stored_message is not None and formatted_content != stored_message.content:
        updated = await memory_manager.enrich_message(0, stored_message, formatted_content, group_id)
        ensure_current_session()
        if not updated:
            return
    logger.info(f"[群聊记忆] 消息已保存")
    event._llm_memory_content = formatted_content
    if repeated_text:
        try:
            await memory_manager.add_message(0, "assistant", repeated_text, group_id)
            logger.info(f"[群复读] 已记录 bot 复读: group_id={group_id}, msg={repeated_text[:50]}...")
        except Exception as e:
            logger.warning(f"[群复读] 记录 bot 复读失败: group_id={group_id}, error={e}")

    ensure_current_session()

    from .image_observations import unsolicited_view
    proactive_content = unsolicited_view(formatted_content)
    proactive_manager.add_recent_message(group_id, proactive_content, user_id)
    if not image_urls or plain_message.strip():
        proactive_manager.increment_counter(group_id)

    # 情感分析：分析消息情感（异步）
    if config.enable_sentiment_analysis and plain_message:
        async def analyze_sentiment():
            from .conversation import session_scope
            with session_scope(f"group_{group_id}"):
                await sentiment_analyzer.analyze(
                    plain_message, user_id, group_id, is_to_bot=is_to_bot,
                    directed_at_other=directed_at_other and not is_to_bot,
                )
        _start_incoming_task(analyze_sentiment())

    # 尝试主动发言（使用formatted_content，包含图片描述）
    if image_urls and not plain_message.strip():
        return  # Sharing a picture is not a request for unsolicited captioning.
    if has_meme_image:
        logger.info(f"[主动发言] 跳过：当前图片已识别为表情包，group_id={group_id}")
        return
    if is_to_bot:
        logger.info(
            f"[主动发言] 跳过：消息明确指向 Bot，将由被动回复处理 group_id={group_id}"
        )
        return
    if followups.eligible(group_id, user_id, directed_at_other):
        return  # The passive planner will decide whether this is actually a followup.
    await proactive_manager.try_proactive(bot, group_id, proactive_content, user_id, _recorded=True)


async def should_respond(event: MessageEvent) -> bool:
    """判断是否应该响应消息"""
    if getattr(event, "_llm_record_obsolete", False):
        return False
    sid = session_key(event.user_id, getattr(event, "group_id", None))
    if getattr(event, "_llm_generation", current_generation(sid)) != current_generation(sid):
        return False
    if str(event.user_id) == str(event.self_id):
        return False
    # 私聊直接响应
    if isinstance(event, PrivateMessageEvent):
        return True

    # 群聊判断触发条件
    if isinstance(event, GroupMessageEvent):
        message = event.get_plaintext().strip()
        targets_bot, directed_elsewhere = _group_message_direction(event, message)
        from . import followups
        return targets_bot or bool(message and _eligible_event_followup(event, directed_elsewhere))

    return False


def _eligible_event_followup(event, directed_elsewhere=False):
    from . import followups
    window = followups.active(event.group_id)
    return (window is not None
            and getattr(event, "_llm_followup_window", window) is window
            and followups.eligible(event.group_id, event.user_id, directed_elsewhere))


# 消息响应器
chat_matcher = on_message(rule=Rule(should_respond), priority=10, block=True)


def extract_trigger_word(message: str) -> tuple[str, str]:
    """提取触发词并返回清理后的消息"""
    for word in config.trigger_words:
        if message.startswith(word):
            # 移除触发词
            cleaned = message[len(word):].strip()
            return word, cleaned
    return "", message


async def get_images_from_message(event: MessageEvent) -> list[str]:
    """从消息中提取图片URL"""
    images = []
    for seg in _as_segment_list(event.message):
        seg_type, data = _segment_type_and_data(seg)
        if seg_type == "image":
            url = _image_url_from_data(data)
            # 过滤无效URL（QQ内置表情等）
            if url:
                images.append(url)
    return images


# QQ 内置表情映射（常见表情）
FACE_EMOJI_MAP = {
    "0": "微笑", "1": "偷笑", "2": "大笑", "3": "抠鼻", "4": "擦汗",
    "5": "鼓掌", "6": "啤酒", "7": "瓢虫", "8": "睡", "9": "爱心",
    "10": "玫瑰", "11": "小星星", "12": "月亮", "13": "太阳", "14": "奖杯",
    "15": "蛋糕", "16": "炸弹", "17": "刀", "18": "足球", "19": "便便",
    "20": "咖啡", "21": "音乐", "22": "灯泡", "23": "纸巾", "24": "药水",
    "25": "口罩", "26": "电话", "27": "时钟", "28": "包裹", "29": "书本",
    "30": "铅笔", "31": "考试", "32": "画板", "33": "追", "34": "期待",
    "35": "甩", "36": "得意", "37": "恼火", "38": "石化", "39": "擦汗",
    "40": "无奈", "41": "流汗", "42": "呕吐", "43": "抓狂", "44": "囧",
    "45": "尴尬", "46": "微笑", "47": "难过", "48": "伤心", "49": "哭了",
    "50": "生气", "51": "白眼", "52": "左哼", "53": "右哼", "54": "哈欠",
    "55": "折磨", "56": "悠闲", "57": "惊讶", "58": "疑问", "59": "冻僵",
    "60": "恐惧", "61": "发呆", "62": "挥手", "63": "再见", "64": "亲亲",
    "65": "抱抱", "66": "举爪", "67": "商务", "68": "NO", "69": "OK",
    "70": "握手", "71": "献花", "72": "献玫瑰", "73": "差评", "74": "好评",
    "75": "加油", "76": "点赞", "77": "weak", "78": "good", "79": "抱拳",
    "80": "查看", "81": "发怒", "82": "爱心", "83": "心碎", "84": "允悲",
    "85": "托腮", "86": "吃瓜", "87": "加油", "88": "打call", "89": "崇拜",
    "90": "比心", "91": "点赞", "92": "社会社会", "93": "旺柴", "94": "好的",
    "95": "打工人", "96": "加油打工人", "97": "我裂开了", "98": "摊手",
    "99": "ohhh", "100": "呜呜", "101": "好累", "102": "药丸", "103": "老爷爷",
    "104": "吸入", "105": "抛出", "106": "妙啊", "107": "梭哈", "108": "精神",
    "109": "蔫了", "110": "嗯嗯", "111": "no", "112": "甩锅", "113": "无奈",
    "114": "嗨呀", "115": "谢谢你", "116": "夸夸", "117": "么么哒", "118": "收下",
    "119": "比心", "120": "胡瓜", "121": "发箍", "122": "二哈", "123": "雕塑",
    "124": "铃铛", "125": "便捷", "126": "便捷", "127": "二维码", "128": "文件夹",
    "129": "芯片", "130": "放大镜", "131": "云", "132": "及时雨", "133": "太阳",
    "134": "小中风", "135": "脑瓜疼", "136": "吃瓜", "137": "996", "138": "加油",
}


async def get_faces_from_message(event: MessageEvent) -> list[str]:
    """从消息中提取QQ内置表情"""
    faces = []
    for seg in event.message:
        if seg.type == "face":
            emoji_id = seg.data.get("id") or seg.data.get("emoji_id")
            face_name = FACE_EMOJI_MAP.get(emoji_id, f"表情{emoji_id}")
            faces.append(face_name)
    return faces


async def download_images_as_base64(urls: list[str]) -> list[tuple[str, str]]:
    """下载图片，失败项静默跳过并保留成功项的 URL 对应关系。"""
    from .multimodal import process_image_urls

    return await process_image_urls(urls)


@dataclass
class PreparedIncomingMessage:
    """One normalized inbound message, shared by the group and private paths."""

    resolved: ResolvedIncomingMessage
    plain_text: str
    message: str
    image_urls: list[str]
    images_base64: list[str]
    faces: list[str]


async def prepare_incoming_message(bot: Bot, event: MessageEvent) -> PreparedIncomingMessage:
    """Expand, sanitize and download one event exactly once per matcher."""
    resolved = await resolve_incoming_message(bot, event)
    ensure_current_session()

    plain_text = resolved.response_text if resolved.response_text is not None else resolved.plain_text
    _, plain_text = extract_trigger_word(plain_text)
    plain_text = sanitize_input(plain_text)

    downloaded = await download_images_as_base64(resolved.image_urls)
    ensure_current_session()
    _capture_incoming_images(event, downloaded)

    return PreparedIncomingMessage(
        resolved=resolved,
        plain_text=plain_text,
        message=resolved.compose(plain_text),
        image_urls=resolved.image_urls,
        images_base64=[encoded for _, encoded in downloaded],
        faces=resolved.faces,
    )


@chat_matcher.handle()
async def handle_chat(bot: Bot, event: MessageEvent):
    """处理聊天消息入口"""
    sid = session_key(event.user_id, getattr(event, "group_id", None))
    event._llm_generation = getattr(event, "_llm_generation", current_generation(sid))
    with session_scope(sid, generation=event._llm_generation):
        try:
            ensure_current_session()
            await _handle_chat(bot, event)
        except TurnSuperseded:
            event._llm_record_obsolete = True


async def _handle_chat(bot: Bot, event: MessageEvent):
    # Corrections use the same conversation queue as other inputs.
    # A model request already rendering its final answer cannot acknowledge a
    # live steering update; returning early here used to lose that user message.
    # 私聊直接处理，不聚合
    if isinstance(event, PrivateMessageEvent):
        await _process_chat(bot, event)
        return

    # 群聊消息
    group_id = event.group_id
    # 新的定向消息优先于尚未执行的旧 wait；主动发言 wait 也不再抢这轮对话。
    _cancel_planner_wait(group_id)
    proactive_manager.cancel_wait(group_id)

    prepared = await prepare_incoming_message(bot, event)

    message = prepared.message
    images_base64 = prepared.images_base64
    faces = prepared.faces

    # 如果没有消息内容，设置默认
    if not message:
        if not images_base64 and not faces:
            # 群聊仅@且无内容时不回复
            return
        if images_base64:
            message = "请描述这张图片"
        elif faces:
            message = " "

    if config.enable_message_aggregation:
        await add_to_aggregation_buffer(bot, event, message, images_base64, faces)
    else:
        await handle_single_chat(bot, event, message, images_base64, faces)


@serialized_turn(lambda bot, event: session_key(event.user_id, getattr(event, "group_id", None)))
async def _process_chat(bot: Bot, event: MessageEvent):
    """私聊消息处理"""
    user_id = event.user_id

    prepared = await prepare_incoming_message(bot, event)

    message = prepared.message
    images_base64 = prepared.images_base64
    faces = prepared.faces

    # 如果没有消息内容，设置默认
    if not message:
        if not images_base64 and not faces:
            if prepared.image_urls:
                logger.info(f"[私聊图片下载] 图片全部失败且无文本/表情，忽略消息: user_id={user_id}")
                return
            await chat_matcher.finish("你好！有什么我可以帮助你的吗？")
        if images_base64:
            message = "请描述这张图片"
        elif faces:
            message = " "

    planner_decision = await plan_chat_action(
        session_id=memory_manager._get_session_id(user_id, None),
        group_id=0,
        current_message=message,
        mode="passive",
        is_direct=True,
        has_images=bool(images_base64),
        faces=faces,
        default_action="reply",
    )
    # 私聊必须回应；Planner 在这里负责媒体/召回意图，不负责让本轮沉默。
    if not planner_decision.should_continue:
        planner_decision.action = "reply"
    await handle_single_chat(
        bot,
        event,
        message,
        images_base64,
        faces,
        planner_decision,
    )


@serialized_turn(lambda bot, event, *args, **kwargs: session_key(event.user_id, getattr(event, "group_id", None)))
async def handle_single_chat(bot: Bot, event: MessageEvent, message: str,
                              images_base64: list, faces: list,
                              planner_decision: Optional[PlannerDecision] = None):
    """处理单条聊天消息"""
    user_id = event.user_id
    group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
    from . import followups
    direct = not group_id or _group_message_direction(event, event.get_plaintext())[0]
    if not direct and not _eligible_event_followup(event, _group_message_direction(event, event.get_plaintext())[1]):
        return
    if planner_decision is None:
        planner_decision = await plan_chat_action(
            session_id=session_key(user_id, group_id), group_id=group_id or 0,
            current_message=message, mode="passive", is_direct=direct,
            has_images=bool(images_base64), faces=faces, default_action="reply" if direct else "silent",
            followup_context=followups.context(group_id) if not direct else "",
        )
        if group_id and (config.enable_reply_judge or not direct):
            if planner_decision.action == "wait":
                _schedule_planner_wait(bot=bot,event=event,message=message,
                    images_base64=images_base64,faces=faces,decision=planner_decision)
                return
            if not planner_decision.should_continue:
                if not direct:
                    followups.windows.pop(group_id, None)
                return
        elif not planner_decision.should_continue:
            planner_decision.action = "reply"
    group_message_order = _get_group_message_order(event) if group_id else 0
    reply_message_id, should_use_reply = _get_targeted_reply_tool_context(event, group_message_order)

    # 设置工具上下文（供戳一戳 / 表情包 / 语音 / 生图等工具使用）
    from .tools import (
        clear_image_context,
        clear_meme_context,
        clear_poke_context,
        clear_voice_context,
        set_image_context,
        set_meme_context,
        set_poke_context,
        set_voice_context,
    )
    set_poke_context(group_id=group_id, user_id=user_id, bot=bot,
                     current_user_text=getattr(event, "get_plaintext", lambda: "")(),
                     history_request_texts=[getattr(event, "_llm_memory_content", "")])
    set_voice_context(group_id=group_id, user_id=user_id, bot=bot)
    set_image_context(
        group_id=group_id, user_id=user_id, bot=bot,
        reference_images=_current_image_references(event, images_base64),
        missing_reference_images=_missing_image_references(event, images_base64),
    )
    set_meme_context(
        group_id=group_id,
        user_id=user_id,
        bot=bot,
        reply_message_id=reply_message_id,
        should_use_reply=should_use_reply,
        directed=not group_id or _group_message_direction(event, event.get_plaintext())[0],
    )

    try:
        await _do_handle_single_chat(
            bot, event, message, images_base64, faces, user_id, group_id, planner_decision, group_message_order
        )
    finally:
        # 清理工具上下文
        clear_poke_context()
        clear_meme_context()
        clear_voice_context()
        clear_image_context()


async def _do_handle_single_chat(bot: Bot, event: MessageEvent, message: str,
                                  images_base64: list, faces: list, user_id: int,
                                  group_id: int = None,
                                  planner_decision: Optional[PlannerDecision] = None,
                                  group_message_order: int = 0):
    """实际处理单条聊天消息的逻辑"""
    reply_policy = select_reply_policy(
        [current_request_text(event, message)], decision=planner_decision,
    )
    planner_decision = constrain_plan(planner_decision, reply_policy)
    sid = memory_manager._get_session_id(0 if group_id else user_id, group_id)
    # 立即获取历史快照（防止后续消息干扰）
    history = await memory_manager.get_history(0 if group_id else user_id, group_id, check_compress=True)

    # 获取会话模型
    session_model = memory_manager.get_session_model(0 if group_id else user_id, group_id)
    model = session_model or model_manager.current_model

    # 添加当前用户消息
    if isinstance(event, GroupMessageEvent):
        raw_sender_name = event.sender.card or event.sender.nickname or str(user_id)
        sender_name = format_sender_name(raw_sender_name)
        display_message = f"【群员-{sender_name}】 {message}"
    else:
        display_message = message
        raw_sender_name = str(user_id)
        sender_name = format_sender_name(raw_sender_name)

    # 加入表情信息
    if faces:
        face_text = "，".join(faces)
        display_message = f"{display_message} [收到QQ表情：{face_text}]"

    user_content = display_message
    user_images = images_base64 if images_base64 else None
    query_text = message
    caller = "chat:plain"

    # 私聊图片需要在回复前识别，并把描述作为用户消息保存；群聊图片已在记录阶段识别。
    if images_base64 and not group_id:
        image_descriptions = []
        for idx, img_b64 in enumerate(images_base64):
            try:
                desc = (await model_manager.describe_image(img_b64, session_id=sid)).strip()
            except Exception as e:
                logger.warning(f"[私聊图片识别] 失败，已忽略该图片: user_id={user_id}, index={idx}, error={e}")
                continue
            if desc:
                image_descriptions.append(desc)
            else:
                logger.warning(f"[私聊图片识别] 空描述，已忽略: user_id={user_id}, index={idx}")

        if not image_descriptions:
            if not message or message == "请描述这张图片":
                logger.info(f"[私聊图片识别] 图片全部失败且无文本，忽略消息: user_id={user_id}")
                return
            user_images = None
            caller = "chat:plain"
        else:
            desc_text = "\n".join([f"图片{i+1}: {d}" for i, d in enumerate(image_descriptions)])
            if message and message != "请描述这张图片":
                user_content = f"{desc_text}\n\n用户消息: {message}"
            else:
                user_content = f"{desc_text}\n\n请根据图片内容回复用户"
            query_text = user_content
            user_images = None
            caller = "chat:private_with_imgs"
    elif images_base64 and group_id:
        caller = "chat:group_with_imgs"

    references = _current_image_references(event, images_base64)
    if references:
        user_content += "\n" + _image_reference_text(references)

    if isinstance(event, PrivateMessageEvent):
        await memory_manager.add_message(
            user_id,
            "user",
            user_content,
            group_id,
            name=sender_name,
            sender_id=user_id,
        )

    await execute_planned_side_effect(planner_decision)
    planned_meme = await prepare_planned_meme(planner_decision)
    context_history = history
    if group_id:
        context_history = _drop_current_turn_from_history(
            history,
            [(sender_name, message, bool(images_base64 or faces))],
        )
    messages = await build_context_messages(
        history=context_history,
        session_id=sid,
        query=query_text,
        user_id=user_id,
        group_id=group_id,
        user_content=user_content,
        user_name=sender_name if isinstance(event, GroupMessageEvent) else None,
        user_images=user_images,
        reply_policy=reply_policy,
        planner_decision=planner_decision,
        planned_meme=planned_meme,
    )

    # 调用模型
    sent_turn = None
    try:
        if planned_meme and is_meme_only(planner_decision):
            response_text = ""
        else:
            response_text = await model_manager.run_turn(
                messages,
                progress_callback=lambda text: _send_research_progress(bot, event, text, user_id, group_id),
                reply_policy=reply_policy,
                model=model,
                role="chat",
                caller=caller,
                session_id=sid,
                required_tool=(planner_decision.required_tool if planner_decision else None),
                media_followup=(planner_decision.media_followup if planner_decision else True),
            )
        reply_policy = getattr(response_text, "reply_policy", None) or reply_policy

        cleaned_response = clean_response(response_text, preserve_paragraphs=reply_policy.detailed)

        async def send_media_followup(text: str) -> None:
            ensure_send_allowed(group_id)
            await _deliver_agent_reply(response_text,text,lambda:bot.send(event,text), bot=bot, group_id=group_id)

        # 媒体已由工具发送；Planner 可选择只发媒体或再补一句文字。
        if await _finalize_immediate_tool_turn(
            user_id,
            group_id,
            followup_text=cleaned_response,
            send_followup=send_media_followup,
            model_spec=getattr(response_text, "model_spec", None),
        ):
            return

        # 如果本轮会发表情包，且文字只是 meta 废话，丢弃文字（只发表情）
        from .utils import is_meta_meme_filler
        if planned_meme and wants_meme(planner_decision) and is_meta_meme_filler(cleaned_response):
            logger.info(f"[meme] 过滤废话文字回复: {cleaned_response!r}")
            cleaned_response = ""

        async def send_text_reply(text: str, is_first: bool) -> None:
            payload = _build_targeted_reply_message(event, text, group_message_order) if is_first else text
            ensure_send_allowed(group_id)
            await _deliver_agent_reply(response_text,text,lambda:bot.send(event,payload), bot=bot, group_id=group_id)

        sent_turn = await send_response_turn(
            text=cleaned_response,
            decision=planner_decision,
            planned_meme=planned_meme,
            send_text=send_text_reply,
            reply_policy=reply_policy,
        )
        if not sent_turn.sent_anything:
            return

        # 记录最近发言时间（用于主动发言冷却和情感归因）
        if group_id:
            proactive_manager.mark_just_spoke(group_id)
            from . import followups
            followups.answered(group_id, [user_id], sent_turn.text_for_memory,
                               direct=_group_message_direction(event, event.get_plaintext())[0],
                               closed=bool(planner_decision and planner_decision.action == "close"))

        # 历史占位：让记忆里保留"我具体发了一张什么表情包"
        history_text = _build_assistant_history_text(sent_turn.text_for_memory, sent_turn.sent_meme)
        await memory_manager.add_message(0 if group_id else user_id, "assistant", history_text, group_id,
                                         model_spec=getattr(response_text, "model_spec", None) if sent_turn.text_for_memory else None)

    except FinishedException:
        raise
    except DeliveryUncertain as e:
        logger.warning(f"[agent_delivery] {e}")
        return
    except Exception as e:
        if sent_turn and sent_turn.sent_anything:
            logger.error(f"回复已发送，后续记忆记录失败: {e}")
            return
        if await _finalize_immediate_tool_turn(user_id, group_id):
            logger.warning(f"模型后处理失败，但本轮即时媒体已经发送: {e}")
            return
        logger.error(f"模型调用失败: {e}")
        ensure_send_allowed(group_id)
        try:
            await bot.send(event, e.user_message if isinstance(e, ModelTurnTimeout) else "刚才没处理成功，稍后再试一下。")
        except Exception:
            pass
        return



@serialized_turn(lambda bot, group_id, *args, **kwargs: f"group_{group_id}")
async def handle_merged_chat(bot: Bot, group_id: int, pending_messages: list[PendingMessage], *, _wait_started_at=None, _wait_replans=0):
    """处理合并后的多条消息"""
    if not pending_messages:
        return

    combined = "\n".join(f"【群员-{format_sender_name(pm.sender_name)} QQ:{pm.user_id}】 {pm.message}" for pm in pending_messages)
    from . import followups
    direct = any(_group_message_direction(pm.event, getattr(pm.event, "get_plaintext", lambda: pm.message)())[0]
                 for pm in pending_messages)
    if not direct:
        pending_messages = [pm for pm in pending_messages if _eligible_event_followup(
            pm.event, _group_message_direction(pm.event, getattr(pm.event, "get_plaintext", lambda: pm.message)())[1])]
        if not pending_messages:
            return
        combined = "\n".join(f"【群员-{format_sender_name(pm.sender_name)} QQ:{pm.user_id}】 {pm.message}" for pm in pending_messages)
    decision = await plan_chat_action(
        session_id=f"group_{group_id}", group_id=group_id, current_message=combined,
        mode="passive", is_direct=direct,
        followup_context=followups.context(group_id) if not direct else "",
        has_images=any(pm.images_base64 for pm in pending_messages),
        faces=[face for pm in pending_messages for face in pm.faces], default_action="reply" if direct else "silent",
    )
    ensure_send_allowed(group_id)
    if (config.enable_reply_judge or not direct) and decision.action == "wait":
        loop = asyncio.get_running_loop()
        started = loop.time() if _wait_started_at is None else _wait_started_at
        remaining = config.planner_wait_max_seconds - (loop.time() - started)
        if remaining > 0 and _wait_replans < config.planner_wait_max_replans:
            _cancel_planner_wait(group_id)
            async def resume():
                try:
                    await asyncio.sleep(min(decision.wait_delay_seconds, remaining))
                    await handle_merged_chat(bot, group_id, pending_messages,
                        _wait_started_at=started, _wait_replans=_wait_replans + 1)
                finally:
                    if planner_wait_tasks.get(group_id) is asyncio.current_task():
                        planner_wait_tasks.pop(group_id, None)
            planner_wait_tasks[group_id] = asyncio.create_task(resume(), name=f"batch-wait-{group_id}")
        return
    if (config.enable_reply_judge or not direct) and not decision.should_continue:
        if not direct:
            followups.windows.pop(group_id, None)
        return
    if not decision.should_continue:
        decision.action = "reply"
    if planner_wait_tasks.get(group_id) is asyncio.current_task():
        planner_wait_tasks.pop(group_id, None)
    if len({pm.user_id for pm in pending_messages}) > 1 and (
        decision.action == "poke" or decision.required_tool
    ):
        # "sender", "me" and personal media references need an unambiguous
        # author. Replan contiguous author groups before executing any effect;
        # ordinary multiuser text replies still keep their shared batch.
        author_batches: list[list[PendingMessage]] = []
        for pm in pending_messages:
            if not author_batches or author_batches[-1][-1].user_id != pm.user_id:
                author_batches.append([])
            author_batches[-1].append(pm)
        for author_batch in author_batches:
            await handle_merged_chat(bot, group_id, author_batch)
        return
    for pm in pending_messages:
        pm.planner_decision = decision

    # 聚合批次的工具与定向回复上下文以第一条触发消息为准。
    first_msg = pending_messages[0]
    primary_user_id = first_msg.user_id
    reply_message_id, should_use_reply = _get_targeted_reply_tool_context(
        first_msg.event,
        first_msg.group_message_order,
    )

    # 设置工具上下文（供戳一戳 / 表情包 / 语音 / 生图等工具使用）
    from .tools import (
        clear_image_context,
        clear_meme_context,
        clear_poke_context,
        clear_voice_context,
        set_image_context,
        set_meme_context,
        set_poke_context,
        set_voice_context,
    )
    set_poke_context(group_id=group_id, user_id=primary_user_id, bot=bot,
                     current_user_text=getattr(pending_messages[-1].event, "get_plaintext", lambda: "")() if len({pm.user_id for pm in pending_messages}) == 1 else "",
                     history_request_texts=[getattr(pm.event, "_llm_memory_content", "") for pm in pending_messages],
                     allow_sender_alias=len({pm.user_id for pm in pending_messages}) == 1)
    set_voice_context(group_id=group_id, user_id=primary_user_id, bot=bot)
    set_image_context(
        group_id=group_id, user_id=primary_user_id, bot=bot,
        reference_images=[ref for pm in pending_messages for ref in _current_image_references(pm.event, pm.images_base64)],
        missing_reference_images=any(_missing_image_references(pm.event, pm.images_base64) for pm in pending_messages),
        allow_auto_reference=len({pm.user_id for pm in pending_messages}) == 1,
    )
    set_meme_context(
        group_id=group_id,
        user_id=primary_user_id,
        bot=bot,
        reply_message_id=reply_message_id,
        should_use_reply=should_use_reply,
        directed=any(_group_message_direction(pm.event, getattr(pm.event, "get_plaintext", lambda: pm.message)())[0]
                     for pm in pending_messages),
    )

    try:
        await _do_handle_merged_chat(
            bot,
            group_id,
            pending_messages,
            primary_user_id=primary_user_id,
        )
    finally:
        # 清理工具上下文
        clear_poke_context()
        clear_meme_context()
        clear_voice_context()
        clear_image_context()


async def _do_handle_merged_chat(
    bot: Bot,
    group_id: int,
    pending_messages: list[PendingMessage],
    *,
    primary_user_id: int,
):
    """实际处理合并后的多条消息的逻辑"""
    sid = f"group_{group_id}"
    participant_count = len({pm.user_id for pm in pending_messages})
    reply_policy = select_reply_policy(
        [current_request_text(pm.event, pm.message) for pm in pending_messages],
        scene="multiuser" if participant_count > 1 else "chat",
        decision=pending_messages[0].planner_decision,
    )
    # 构建合并后的用户消息列表
    combined_text_parts = []
    all_images = []

    for pm in pending_messages:
        if pm.message:
            formatted_name = format_sender_name(pm.sender_name)
            combined_text_parts.append(f"【群员-{formatted_name} QQ:{pm.user_id}】 {pm.message}")
        if pm.images_base64:
            all_images.extend(pm.images_base64)
            combined_text_parts.append(_image_reference_text(_current_image_references(pm.event, pm.images_base64)))

    # 合并后的消息文本
    combined_message = "\n".join(combined_text_parts)
    if not combined_message:
        combined_message = "几位群友向你打招呼" if participant_count > 1 else "群友向你打招呼"

    logger.info(f"[消息聚合] 合并消息: {combined_message[:100]}...")

    # 获取历史
    history = await memory_manager.get_history(0, group_id, check_compress=True)

    # 获取会话模型
    session_model = memory_manager.get_session_model(0, group_id)
    model = session_model or model_manager.current_model

    # 构建提示
    planner_decision = constrain_plan(pending_messages[0].planner_decision, reply_policy)
    user_content = combined_message
    situation_hint = (
        f"【当前情况】{participant_count} 位群友的消息按时间排列。区分说话人，选最相关的问题自然接话，"
        "不必逐人点名或逐条答复，仍遵守本轮文字预算。"
        if participant_count > 1 else ""
    )

    await execute_planned_side_effect(planner_decision)
    planned_meme = await prepare_planned_meme(planner_decision)
    context_history = _drop_current_turn_from_history(
        history,
        [
            (format_sender_name(pm.sender_name), pm.message or "", bool(pm.images_base64 or pm.faces))
            for pm in pending_messages
        ],
    )
    messages = await build_context_messages(
        history=context_history,
        session_id=sid,
        query=combined_message,
        user_id=0,
        group_id=group_id,
        user_content=user_content,
        user_images=all_images[:max(1, int(config.incoming_images_limit))] if all_images else None,
        extra_system=situation_hint,
        reply_policy=reply_policy,
        planner_decision=planner_decision,
        planned_meme=planned_meme,
        related_user_id=0,
        participant_profiles={
            pm.user_id: format_sender_name(pm.sender_name)
            for pm in pending_messages
        },
    )

    # 调用模型
    sent_turn = None
    try:
        # 注：群聊图片描述已由 record_group_message 在每条消息保存时调用 describe_image
        # 写入历史；此处历史已包含描述，无需重复识别（重复会浪费 vision token + 增加延迟）。
        # 仅在含图片时把 raw images 透传给 user_msg，让支持原生 vision 的模型能直接看图。
        if planned_meme and is_meme_only(planner_decision):
            response_text = ""
        else:
            response_text = await model_manager.run_turn(
                messages,
                progress_callback=lambda text: _send_research_progress(bot, pending_messages[0].event, text, 0, group_id),
                reply_policy=reply_policy,
                model=model,
                role="chat",
                caller="chat:reply",
                session_id=sid,
                required_tool=(planner_decision.required_tool if planner_decision else None),
                media_followup=(planner_decision.media_followup if planner_decision else True),
            )
        reply_policy = getattr(response_text, "reply_policy", None) or reply_policy

        cleaned_response = clean_response(response_text, preserve_paragraphs=reply_policy.detailed)

        first_msg = pending_messages[0]

        async def send_media_followup(text: str) -> None:
            ensure_send_allowed(group_id)
            await _deliver_agent_reply(response_text,text,lambda:bot.send(first_msg.event,text), bot=bot, group_id=group_id)

        # 图片或语音已发送；只有媒体后的 text part 才会得到收尾文字。
        if await _finalize_immediate_tool_turn(
            primary_user_id,
            group_id,
            followup_text=cleaned_response,
            send_followup=send_media_followup,
            model_spec=getattr(response_text, "model_spec", None),
        ):
            return

        # 如果本轮会发表情包，且文字只是 meta 废话，丢弃文字（只发表情）
        from .utils import is_meta_meme_filler
        if planned_meme and wants_meme(planner_decision) and is_meta_meme_filler(cleaned_response):
            logger.info(f"[meme] 过滤废话文字回复: {cleaned_response!r}")
            cleaned_response = ""

        # 发送回复（使用 bot.send 而不是 finish，因为不是在 matcher 中）
        async def send_text_reply(text: str, is_first: bool) -> None:
            payload = (
                _build_targeted_reply_message(first_msg.event, text, first_msg.group_message_order)
                if is_first
                else text
            )
            ensure_send_allowed(group_id)
            await _deliver_agent_reply(response_text,text,lambda:bot.send(first_msg.event,payload), bot=bot, group_id=group_id)

        sent_turn = await send_response_turn(
            text=cleaned_response,
            decision=planner_decision,
            planned_meme=planned_meme,
            send_text=send_text_reply,
            reply_policy=reply_policy,
        )
        if not sent_turn.sent_anything:
            return

        # 记录最近发言时间（用于主动发言冷却和情感归因）
        proactive_manager.mark_just_spoke(group_id)

        # 保存助手回复
        from . import followups
        followups.answered(group_id, [pm.user_id for pm in pending_messages], sent_turn.text_for_memory,
                           direct=any(_group_message_direction(pm.event, getattr(pm.event, "get_plaintext", lambda: pm.message)())[0]
                                      for pm in pending_messages),
                           closed=bool(planner_decision and planner_decision.action == "close"))
        history_text = _build_assistant_history_text(sent_turn.text_for_memory, sent_turn.sent_meme)
        await memory_manager.add_message(0, "assistant", history_text, group_id,
                                         model_spec=getattr(response_text, "model_spec", None) if sent_turn.text_for_memory else None)

    except DeliveryUncertain as e:
        logger.warning(f"[agent_delivery] {e}")
        return
    except Exception as e:
        if sent_turn and sent_turn.sent_anything:
            logger.error(f"回复已发送，后续记忆记录失败: {e}")
            return
        if await _finalize_immediate_tool_turn(primary_user_id, group_id):
            logger.warning(f"[消息聚合] 模型后处理失败，但本轮即时媒体已经发送: {e}")
            return
        if isinstance(e, ModelTurnTimeout):
            logger.warning(f"[消息聚合] {e}")
        else:
            logger.exception(f"[消息聚合] 模型调用失败: {e}")
        ensure_send_allowed(group_id)
        try:
            await bot.send(pending_messages[0].event, e.user_message if isinstance(e, ModelTurnTimeout) else "刚才没处理成功，稍后再试一下。")
        except Exception:
            pass
        return
