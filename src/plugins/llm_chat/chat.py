"""聊天处理核心逻辑"""
import asyncio
from collections import OrderedDict
from typing import Awaitable, Callable, Optional
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
from .qq_addressing import MessageTarget, QQAddressing
from .proactive import proactive_manager
from .repeat import repeat_manager
from .turns import is_meme_only, prepare_planned_meme, send_response_turn
from .utils import clean_response
from .conversation import (
    TurnSuperseded, current_generation, ensure_current_session, ensure_send_allowed,
    serialized_turn, session_key, session_scope,
)
from .agent_runner import DeliveryUncertain, deliver_reply as _deliver_agent_reply
from .incoming import (
    ResolvedIncomingMessage, format_sender_name, resolve_incoming_message,
    object_value, segment_type_and_data, as_segment_list, clear_mention_cache,
)
from .image_references import ImageReference, image_reference_store, reference_marker


REQUEST_FAILURE = "刚才没处理成功，可以再发一次。"
IMAGE_FAILURE = "这次没能读到图片，可以重新发一下。"
MISSING_IMAGE_GUIDANCE = "按每条消息的原始图片编号回答可见内容，简短指出需要重发的图片编号。"


async def _send_request_notice(bot, event, group_id, text):
    ensure_send_allowed(group_id)
    try:
        await bot.send(event, text)
    except Exception as error:
        logger.warning(f"[聊天反馈] 发送失败 group={group_id}: {type(error).__name__}")


async def _plan_chat_request(bot, event, *, group_id, is_direct, **kwargs) -> Optional[PlannerDecision]:
    try:
        return await plan_chat_action(group_id=group_id, is_direct=is_direct, **kwargs)
    except Exception as error:
        logger.warning(f"[聊天规划] 本轮结束 group={group_id}: {type(error).__name__}")
        if is_direct:
            await _send_request_notice(bot, event, group_id, REQUEST_FAILURE)
        return None


async def _send_research_progress(bot, event, text, user_id, group_id):
    ensure_send_allowed(group_id)
    receipt = await _deliver_agent_reply(text, str(text), lambda: bot.send(event, str(text)), bot=bot, group_id=group_id)
    await memory_manager.add_message(0 if group_id else user_id, "assistant", "[检索进展，非最终结论] " + text, group_id, model_spec=getattr(text, "model_spec", None))
    return receipt


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

    @property
    def directed_to_bot(self) -> bool:
        text = getattr(self.event, "get_plaintext", lambda: self.message)()
        return _group_message_direction(self.event, text)[0]


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
            decision = await _plan_chat_request(
                bot, event,
                session_id=f"group_{group_id}",
                group_id=group_id,
                current_message=message,
                current_targets=_event_message_targets(event, message),
                bot_self_id=str(getattr(bot, "self_id", "")),
                recent_context=recent_context,
                mode="passive",
                is_direct=direct,
                followup_context=followups.context(group_id) if not direct else "",
                has_images=bool(images_base64),
                faces=faces,
                default_action="reply" if direct else "silent",
            )
            if decision is None:
                return
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


def _event_message_targets(event: MessageEvent, message: str, *, user_id: int | None = None) -> tuple[MessageTarget, ...]:
    sender = getattr(event, "sender", None)
    names = tuple(name for key in ("card", "nickname") if (name := object_value(sender, key, "")))
    current = MessageTarget(_get_event_message_id(event), user_id if user_id is not None else event.user_id,
                            current_request_text(event, message), names, "current")
    targets = [current]
    for segment in getattr(event, "original_message", None) or getattr(event, "message", ()):
        if segment.type == "at":
            raw_id = str(segment.data.get("qq", ""))
            if raw_id.isdecimal() and int(raw_id) > 0:
                targets.append(MessageTarget(None, int(raw_id), "本轮消息原生 @ 的对象", origin="mentioned"))
    reply = getattr(event, "reply", None)
    if reply is not None:
        message_id = _get_event_message_id(reply)
        sender_id = getattr(getattr(reply, "sender", None), "user_id", None)
        if message_id is not None and isinstance(sender_id, int):
            quoted = getattr(reply, "message", None)
            excerpt = quoted.extract_plain_text() if isinstance(quoted, Message) else ""
            targets.append(MessageTarget(message_id, sender_id, excerpt or "本轮所引用的消息", origin="quoted"))
    return tuple(targets)


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


async def _finish_response_turn(bot, event, user_id, group_id, *, direct, response,
                                reply_policy, planner_decision, planned_meme):
    """Deliver the planned reply and record whichever parts were sent."""
    reply_policy = getattr(response, "reply_policy", None) or reply_policy
    text = clean_response(response, preserve_paragraphs=reply_policy.detailed)
    model_spec = getattr(response, "model_spec", None)

    async def send_text(value, payload):
        ensure_send_allowed(group_id)
        await _deliver_agent_reply(response, value, lambda: bot.send(event, payload),
                                   bot=bot, group_id=group_id)

    async def send_media_followup(value):
        await send_text(value, QQAddressing().text(value))
        if group_id:
            from . import followups
            followups.answered(group_id, value, direct=direct,
                               closed=bool(planner_decision and planner_decision.action == "close"))

    if await _finalize_immediate_tool_turn(user_id, group_id, followup_text=text,
                                           send_followup=send_media_followup, model_spec=model_spec):
        return

    sent_turn = await send_response_turn(text=text, decision=planner_decision, planned_meme=planned_meme,
                                         send_text=send_text, reply_policy=reply_policy, group_chat=bool(group_id))
    if not sent_turn.sent_anything:
        return
    try:
        if group_id:
            from . import followups
            proactive_manager.mark_just_spoke(group_id)
            followups.answered(group_id, sent_turn.text_for_memory, direct=direct,
                               closed=bool(planner_decision and planner_decision.action == "close"))
        history_text = _build_assistant_history_text(sent_turn.text_for_memory, sent_turn.sent_meme)
        await memory_manager.add_message(0 if group_id else user_id, "assistant", history_text, group_id,
                                         model_spec=model_spec if sent_turn.text_for_memory else None)
    except Exception as error:
        logger.error(f"回复已发送，后续记忆记录失败: {error}")


async def _report_response_failure(bot, event, user_id, group_id, *, direct, error):
    if isinstance(error, DeliveryUncertain):
        logger.warning(f"[agent_delivery] {error}")
        return
    if await _finalize_immediate_tool_turn(user_id, group_id):
        logger.warning(f"模型后处理失败，但本轮即时媒体已经发送: {error}")
        return
    logger.error(f"模型调用或投递失败: {error}")
    if direct:
        notice = error.user_message if isinstance(error, ModelTurnTimeout) else REQUEST_FAILURE
        await _send_request_notice(bot, event, group_id, notice)


def _drop_current_turn_from_history(
    history: list[LLMMessage],
    events,
) -> list[LLMMessage]:
    """Only remove records known to be represented in the current prompt."""
    ids = {getattr(event, "_llm_memory_source_id", None) for event in events}
    ids.discard(None)
    return [message for message in history if message.source_id not in ids]


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
    clear_mention_cache()


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
    reply_sender = object_value(object_value(reply, "sender", None), "user_id", None)
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


def _image_positions(resolved: ResolvedIncomingMessage) -> dict[str, int]:
    positions = {}
    for index, url in enumerate(resolved.image_slots, 1):
        if url:
            positions.setdefault(url, index)
    return positions


def _capture_incoming_images(event: MessageEvent, downloaded: list[tuple[str, str]]) -> list[ImageReference]:
    sid = session_key(event.user_id, getattr(event, "group_id", None))
    sender = getattr(event, "sender", {})
    name = object_value(sender, "card", "") or object_value(sender, "nickname", "") or str(event.user_id)
    resolved = getattr(event, "_llm_resolved_incoming_message", None)
    quoted_urls = set(resolved.context_image_urls) if resolved else set()
    positions = _image_positions(resolved) if resolved else {}
    references = []
    previous = {ref.base64: ref for ref in getattr(event, "_llm_image_references", [])}
    for index, (url, encoded) in enumerate(downloaded, 1):
        index = positions.get(url, index)
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


def _group_image_context(event: MessageEvent, images: list[str]) -> str:
    references = _current_image_references(event, images)
    slots = getattr(event, "_llm_image_slots", None)
    if not slots:
        return _image_reference_text(references)
    by_image = {ref.base64: ref for ref in references}
    return "\n".join(
        f"图片{index}: {reference_marker(by_image[encoded])} {by_image[encoded].label}"
        if encoded in by_image else f"图片{index}: 未能读取"
        for index, encoded in enumerate(slots, 1)
    )


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
    kinds = {segment_type_and_data(seg)[0] for seg in as_segment_list(event.message)}
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
    event._llm_memory_source_id = getattr(stored_message, "source_id", None)
    event._llm_memory_content = initial_content

    # OneBot reply/forward 只在这里展开一次；后续响应 matcher 复用事件缓存。
    resolved = await resolve_incoming_message(bot, event)
    ensure_current_session()
    plain_message = resolved.plain_text
    message = resolved.compose()
    is_to_bot, directed_at_other = _group_message_direction(event, plain_message)
    from . import followups
    if directed_at_other and not is_to_bot and _eligible_event_followup(event):
        followups.skip(group_id)
    image_urls = resolved.image_urls

    # Non-directed pictures enrich their original row in the background. Direct
    # questions still wait for observations before planner/chat consume them.
    if image_urls and not is_to_bot and not (plain_message.strip() and _eligible_event_followup(event, directed_at_other)):
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

    # 忽略空消息（没有文字、没有图片、没有表情）
    if not message and not resolved.image_slots and not faces:
        return


    # 构建完整消息内容
    body = message
    formatted_content = f"【群员-{sender_name}】 {body}"
    if resolved.image_slots and not images_base64 and not body:
        formatted_content = f"【群员-{sender_name}】 [图片下载失败]"

    has_meme_image = False
    # 如果有图片，识别图片内容（同时顺手判定是否表情包并入库）
    if images_base64:
        from .memes import clean_meme_memory_description, meme_manager
        image_descriptions = {}
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
                image_descriptions[url] = f"{reference_marker(ref)} {desc}"
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
            desc_text = "\n".join(
                f"图片{index}（自动识别，仅作背景）: {image_descriptions[url]}"
                for index, url in enumerate(resolved.image_slots, 1) if url in image_descriptions
            )
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
    ensure_current_session()

    from .image_observations import unsolicited_view
    proactive_content = unsolicited_view(formatted_content)
    proactive_manager.add_recent_message(group_id, proactive_content, user_id, _get_event_message_id(event),
                                         sender_name=sender_name)
    if not resolved.image_slots or plain_message.strip():
        proactive_manager.increment_counter(group_id)

    # 尝试主动发言（使用formatted_content，包含图片描述）
    if resolved.image_slots and not plain_message.strip():
        return  # Sharing a picture is not a request for unsolicited captioning.
    if has_meme_image:
        logger.info(f"[主动发言] 跳过：当前图片已识别为表情包，group_id={group_id}")
        return
    if is_to_bot:
        logger.info(
            f"[主动发言] 跳过：消息明确指向 Bot，将由被动回复处理 group_id={group_id}"
        )
        return
    if plain_message.strip() and _eligible_event_followup(event, directed_at_other):
        return  # The passive planner will decide whether this is actually a followup.
    repeat_candidate = ""
    if not (directed_at_other or image_urls or faces) and repeat_manager.is_candidate(group_id, plain_message):
        repeat_candidate = plain_message
    kwargs = {"repeat_candidate": repeat_candidate} if repeat_candidate else {}
    await proactive_manager.try_proactive(
        bot, group_id, proactive_content, user_id, request_text=plain_message, _recorded=True, **kwargs,
    )


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
        return targets_bot or bool(message and _eligible_event_followup(event, directed_elsewhere))

    return False


def _eligible_event_followup(event, directed_elsewhere=False):
    from . import followups
    window = followups.active(event.group_id)
    return (not directed_elsewhere and window is not None
            and getattr(event, "_llm_followup_window", window) is window)


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
    image_slots: list[str | None]
    images_base64: list[str]
    faces: list[str]


async def prepare_incoming_message(bot: Bot, event: MessageEvent) -> PreparedIncomingMessage:
    """Expand and download one event exactly once per matcher."""
    resolved = await resolve_incoming_message(bot, event)
    ensure_current_session()

    plain_text = resolved.response_text if resolved.response_text is not None else resolved.plain_text
    _, plain_text = extract_trigger_word(plain_text)

    downloaded = await download_images_as_base64(resolved.image_urls)
    ensure_current_session()
    _capture_incoming_images(event, downloaded)

    encoded_by_url = dict(downloaded)
    image_slots = [encoded_by_url.get(url) for url in resolved.image_slots]
    event._llm_image_slots = image_slots
    return PreparedIncomingMessage(
        resolved=resolved,
        plain_text=plain_text,
        message=resolved.compose(plain_text),
        image_slots=image_slots,
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
    direct = _group_message_direction(event, event.get_plaintext())[0]
    # Only a new request to the bot supersedes existing wait plans.
    if direct:
        _cancel_planner_wait(group_id)
        proactive_manager.cancel_wait(group_id)

    prepared = await prepare_incoming_message(bot, event)

    message = prepared.message
    images_base64 = prepared.images_base64
    faces = prepared.faces

    # 如果没有消息内容，设置默认
    if not message:
        if not images_base64 and not faces:
            if getattr(event, "_llm_image_slots", ()) and _group_message_direction(event, event.get_plaintext())[0]:
                await _send_request_notice(bot, event, group_id, IMAGE_FAILURE)
            return
        if images_base64:
            message = "请描述这张图片"
        elif faces:
            message = " "

    await add_to_aggregation_buffer(bot, event, message, images_base64, faces)


@serialized_turn(lambda bot, event: session_key(event.user_id, getattr(event, "group_id", None)))
async def _process_chat(bot: Bot, event: MessageEvent):
    """私聊消息处理"""
    user_id = event.user_id

    prepared = await prepare_incoming_message(bot, event)

    message = prepared.message
    images_base64 = prepared.images_base64
    faces = prepared.faces

    if prepared.image_slots and not images_base64:
        await _send_request_notice(bot, event, None, IMAGE_FAILURE)
        return

    # 如果没有消息内容，设置默认
    if not message:
        if not images_base64 and not faces:
            await chat_matcher.finish("你好！有什么我可以帮助你的吗？")
        if images_base64:
            message = "请描述收到的图片"
        elif faces:
            message = " "

    planner_decision = await _plan_chat_request(
        bot, event,
        session_id=memory_manager._get_session_id(user_id, None),
        group_id=0,
        current_message=message,
        mode="passive",
        is_direct=True,
        has_images=bool(images_base64),
        faces=faces,
        default_action="reply",
    )
    if planner_decision is None:
        return
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
        image_slots=prepared.image_slots or None,
    )


@serialized_turn(lambda bot, event, *args, **kwargs: session_key(event.user_id, getattr(event, "group_id", None)))
async def handle_single_chat(bot: Bot, event: MessageEvent, message: str,
                              images_base64: list, faces: list,
                              planner_decision: Optional[PlannerDecision] = None, *,
                              image_slots: list[str | None] | None = None):
    """处理单条聊天消息"""
    user_id = event.user_id
    group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
    from . import followups
    direct = not group_id or _group_message_direction(event, event.get_plaintext())[0]
    if not direct and not _eligible_event_followup(event, _group_message_direction(event, event.get_plaintext())[1]):
        return
    if planner_decision is None:
        planner_decision = await _plan_chat_request(
            bot, event,
            session_id=session_key(user_id, group_id), group_id=group_id or 0,
            current_message=message, mode="passive", is_direct=direct,
            current_targets=_event_message_targets(event, message) if group_id else (),
            bot_self_id=str(getattr(bot, "self_id", "")),
            has_images=bool(images_base64), faces=faces, default_action="reply" if direct else "silent",
            followup_context=followups.context(group_id) if not direct else "",
        )
        if planner_decision is None:
            return
        if group_id and (config.enable_reply_judge or not direct):
            if planner_decision.action == "wait":
                _schedule_planner_wait(bot=bot,event=event,message=message,
                    images_base64=images_base64,faces=faces,decision=planner_decision)
                return
            if not planner_decision.should_continue:
                if not direct:
                    followups.skip(group_id)
                return
        elif not planner_decision.should_continue:
            planner_decision.action = "reply"
    addressing = planner_decision.addressing if group_id else QQAddressing()

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
    set_voice_context(group_id=group_id, user_id=user_id, bot=bot, addressing=addressing)
    set_image_context(
        group_id=group_id, user_id=user_id, bot=bot,
        addressing=addressing,
        reference_images=_current_image_references(event, images_base64),
        missing_reference_images=_missing_image_references(event, images_base64),
    )
    set_meme_context(
        group_id=group_id,
        user_id=user_id,
        bot=bot,
        addressing=addressing,
        directed=not group_id or _group_message_direction(event, event.get_plaintext())[0],
    )

    try:
        await _do_handle_single_chat(
            bot, event, message, images_base64, faces, user_id, group_id, planner_decision,
            image_slots=image_slots,
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
                                  planner_decision: Optional[PlannerDecision] = None, *,
                                  image_slots: list[str | None] | None = None):
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
    image_guidance = ""

    # 私聊图片需要在回复前识别，并把描述作为用户消息保存；群聊图片已在记录阶段识别。
    if images_base64 and not group_id:
        image_descriptions = []
        failed = []
        descriptions_by_image = {}
        references = {ref.base64: ref for ref in _current_image_references(event, images_base64)}
        for index, img_b64 in enumerate(image_slots if image_slots is not None else images_base64, 1):
            if img_b64 and img_b64 not in descriptions_by_image:
                try:
                    descriptions_by_image[img_b64] = (await model_manager.describe_image(img_b64, session_id=sid)).strip()
                except Exception as error:
                    logger.warning(f"[私聊图片识别] user_id={user_id} image={index}: {type(error).__name__}")
                    descriptions_by_image[img_b64] = ""
            desc = descriptions_by_image.get(img_b64)
            if desc:
                marker = reference_marker(references[img_b64]) if img_b64 in references else ""
                image_descriptions.append(f"图片{index}: {marker} {desc}".strip())
            else:
                failed.append(index)
                image_descriptions.append(f"图片{index}: 未能读取")

        if len(failed) == len(image_descriptions):
            await _send_request_notice(bot, event, None, IMAGE_FAILURE)
            return
        if failed:
            image_guidance = MISSING_IMAGE_GUIDANCE
        user_content = "\n".join(image_descriptions) + f"\n\n用户消息: {display_message}"
        query_text = user_content
        user_images = None
        caller = "chat:private_with_imgs"
    elif images_base64 and group_id:
        caller = "chat:group_with_imgs"

    if group_id:
        image_context = _group_image_context(event, images_base64)
        if image_context:
            user_content += "\n" + image_context
        if None in getattr(event, "_llm_image_slots", ()):
            image_guidance = MISSING_IMAGE_GUIDANCE

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
            [event],
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
        extra_system=image_guidance,
        reply_policy=reply_policy,
        planner_decision=planner_decision,
        planned_meme=planned_meme,
    )

    direct = not group_id or _group_message_direction(event, event.get_plaintext())[0]
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
        await _finish_response_turn(
            bot, event, user_id, group_id, direct=direct, response=response_text,
            reply_policy=reply_policy, planner_decision=planner_decision, planned_meme=planned_meme,
        )
    except FinishedException:
        raise
    except Exception as error:
        await _report_response_failure(bot, event, user_id, group_id, direct=direct, error=error)


@serialized_turn(lambda bot, group_id, *args, **kwargs: f"group_{group_id}")
async def handle_merged_chat(bot: Bot, group_id: int, pending_messages: list[PendingMessage], *, _wait_started_at=None, _wait_replans=0):
    """处理合并后的多条消息"""
    if not pending_messages:
        return

    from . import followups
    directed = [pm for pm in pending_messages if pm.directed_to_bot]
    direct = bool(directed)
    if direct:
        # Other nearby lines remain in group history. Only explicit requests
        # in this batch should be treated as instructions to the bot.
        pending_messages = directed
    else:
        pending_messages = [pm for pm in pending_messages if _eligible_event_followup(
            pm.event, _group_message_direction(pm.event, getattr(pm.event, "get_plaintext", lambda: pm.message)())[1])]
        if not pending_messages:
            return
    combined = "\n".join(f"【群员-{format_sender_name(pm.sender_name)} QQ:{pm.user_id}】 {pm.message}" for pm in pending_messages)
    decision = await _plan_chat_request(
        bot, pending_messages[0].event,
        session_id=f"group_{group_id}", group_id=group_id, current_message=combined,
        mode="passive", is_direct=direct,
        current_targets=tuple(target for pm in pending_messages
                              for target in _event_message_targets(pm.event, pm.message, user_id=pm.user_id)),
        bot_self_id=str(getattr(bot, "self_id", "")),
        followup_context=followups.context(group_id) if not direct else "",
        has_images=any(pm.images_base64 for pm in pending_messages),
        faces=[face for pm in pending_messages for face in pm.faces], default_action="reply" if direct else "silent",
    )
    if decision is None:
        return
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
            followups.skip(group_id)
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

    # 工具调用者沿用首位发言者，消息引用与 @ 由 planner 选择。
    first_msg = pending_messages[0]
    primary_user_id = first_msg.user_id
    addressing = decision.addressing

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
    set_voice_context(group_id=group_id, user_id=primary_user_id, bot=bot, addressing=addressing)
    set_image_context(
        group_id=group_id, user_id=primary_user_id, bot=bot,
        addressing=addressing,
        reference_images=[ref for pm in pending_messages for ref in _current_image_references(pm.event, pm.images_base64)],
        missing_reference_images=any(_missing_image_references(pm.event, pm.images_base64) for pm in pending_messages),
        allow_auto_reference=len({pm.user_id for pm in pending_messages}) == 1,
    )
    set_meme_context(
        group_id=group_id,
        user_id=primary_user_id,
        bot=bot,
        addressing=addressing,
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
        image_context = _group_image_context(pm.event, pm.images_base64)
        if image_context:
            combined_text_parts.append(image_context)

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
    if any(None in getattr(pm.event, "_llm_image_slots", ()) for pm in pending_messages):
        situation_hint = "\n".join(part for part in (situation_hint, MISSING_IMAGE_GUIDANCE) if part)

    await execute_planned_side_effect(planner_decision)
    planned_meme = await prepare_planned_meme(planner_decision)
    context_history = _drop_current_turn_from_history(
        history,
        [pm.event for pm in pending_messages],
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
    )

    event = pending_messages[0].event
    direct = any(pm.directed_to_bot for pm in pending_messages)
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
        await _finish_response_turn(
            bot, event, primary_user_id, group_id, direct=direct, response=response_text,
            reply_policy=reply_policy, planner_decision=planner_decision, planned_meme=planned_meme,
        )
    except Exception as error:
        await _report_response_failure(bot, event, primary_user_id, group_id, direct=direct, error=error)
