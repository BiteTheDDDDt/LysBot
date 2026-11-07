"""事件处理器"""
from .conversation import ensure_send_allowed, serialized_turn, session_key

from nonebot import get_driver, on_request, on_notice, logger
from nonebot.adapters.onebot.v11 import (
    Bot,
    FriendRequestEvent,
    GroupRequestEvent,
    PokeNotifyEvent,
    MessageSegment,
)

logger.info("[events] 模块已加载")


def _is_superuser(user_id: int | str) -> bool:
    superusers = getattr(get_driver().config, "superusers", set()) or set()
    return str(user_id) in {str(uid) for uid in superusers}


def _format_poke_sender_name(name: str, user_id: int) -> str:
    """格式化戳一戳发送者名称"""
    if not name:
        return str(user_id)
    name = name.strip()
    if len(name) > 8:
        return name[:8] + "..."
    return name

# 群请求处理器（包括群邀请和入群申请）
group_req = on_request()


@group_req.handle()
async def handle_group_request(bot: Bot, event: GroupRequestEvent):
    logger.info(f"[events] 收到群请求: type={event.sub_type}, group={event.group_id}")
    approved = _is_superuser(event.user_id)
    await bot.set_group_add_request(
        flag=event.flag,
        sub_type=event.sub_type,
        approve=approved
    )
    if approved:
        logger.info(f"[events] 已同意超级用户群请求: user={event.user_id}, group={event.group_id}")
    else:
        logger.warning(f"[events] 已拒绝非超级用户群请求: user={event.user_id}, group={event.group_id}")


# 好友请求处理器
friend_req = on_request()


@friend_req.handle()
async def handle_friend_request(bot: Bot, event: FriendRequestEvent):
    logger.info(f"[events] 收到好友请求: {event.user_id}")
    approved = _is_superuser(event.user_id)
    await bot.set_friend_add_request(
        flag=event.flag,
        approve=approved
    )
    if approved:
        logger.info(f"[events] 已同意超级用户好友请求: user={event.user_id}")
    else:
        logger.warning(f"[events] 已拒绝非超级用户好友请求: user={event.user_id}")


# 戳一戳事件处理器
poke_notice = on_notice()
_poke_inflight: set[tuple[str, str]] = set()


@poke_notice.handle()
async def handle_poke(bot: Bot, event: PokeNotifyEvent):
    """处理戳一戳事件（包括群聊和私聊戳一戳）"""
    # 只处理别人戳机器人的情况
    if event.target_id != event.self_id:
        return

    user_id = event.user_id
    group_id = event.group_id if hasattr(event, 'group_id') else None

    logger.info(f"[poke] 收到戳一戳: user={user_id}, group={group_id}")
    key = (str(event.self_id), session_key(user_id, group_id))
    # Admit before the serialized conversation queue: bursts must not queue up
    # and produce delayed replies one by one. Group members share this limit.
    if key in _poke_inflight:
        logger.info(f"[poke] 忽略连续戳一戳: user={user_id}, group={group_id}")
        return
    _poke_inflight.add(key)

    # 设置工具上下文
    from .tools import set_meme_context, set_poke_context, set_image_context, set_voice_context
    set_poke_context(group_id=group_id, user_id=user_id, bot=bot)
    set_meme_context(group_id=group_id, user_id=user_id, bot=bot)
    set_image_context(group_id=group_id, user_id=user_id, bot=bot)
    set_voice_context(group_id=group_id, user_id=user_id, bot=bot)

    try:
        # 调用聊天处理逻辑响应戳一戳
        await _handle_poke_chat(bot, user_id, group_id)
    finally:
        # 清理上下文
        from .tools import clear_meme_context, clear_poke_context, clear_image_context, clear_voice_context
        clear_poke_context()
        clear_meme_context()
        clear_image_context()
        clear_voice_context()
        _poke_inflight.discard(key)


@serialized_turn(lambda bot, user_id, group_id=None: session_key(user_id, group_id))
async def _handle_poke_chat(bot: Bot, user_id: int, group_id: int = None):
    """处理戳一戳触发的聊天"""
    from .context_builder import build_context_messages
    from .reply_policy import POKE_REPLY, enforce_reply_budget
    from .models import model_manager
    from .memory import memory_manager
    from .utils import clean_response

    sender_name = str(user_id)
    if group_id:
        try:
            member_info = await bot.get_group_member_info(group_id=group_id, user_id=user_id, no_cache=True)
            raw_name = member_info.get("card") or member_info.get("nickname") or str(user_id)
            sender_name = _format_poke_sender_name(raw_name, user_id)
        except Exception as e:
            logger.warning(f"[poke] 获取群成员信息失败: {e}")

    # 获取历史
    history = await memory_manager.get_history(0 if group_id else user_id, group_id, check_compress=True)

    session_id = memory_manager._get_session_id(0 if group_id else user_id, group_id)
    poke_message = f"【群员-{sender_name}】 [戳了戳你]" if group_id else "[戳了戳你]"
    from .planner import plan_chat_action, execute_planned_side_effect
    decision = await plan_chat_action(
        session_id=session_id, group_id=group_id or 0,
        current_message=poke_message, mode="passive", is_direct=True,
        default_action="silent", trigger_kind="poke",
        bot_self_id=str(getattr(bot, "self_id", "")),
    )
    # Preserve ignored events as context for later participation decisions.
    await memory_manager.add_message(
        0 if group_id else user_id, "user", poke_message, group_id,
        name=sender_name if group_id else "系统", sender_id=user_id,
    )
    if decision.action in {"silent", "wait"}:
        logger.info(f"[poke] planner 保持安静: group={group_id} reason={decision.reason}")
        return
    if decision.action == "poke":
        if not group_id:
            return
        decision.poke_target_id = decision.poke_target_id or "sender"
        result = await execute_planned_side_effect(decision)
        if result.startswith("戳一戳已发送给 "):
            target = user_id if decision.poke_target_id == "sender" else decision.poke_target_id
            await memory_manager.add_message(0, "assistant", f"[已戳一戳 QQ:{target}]", group_id)
            from .proactive import proactive_manager
            proactive_manager.mark_just_spoke(group_id)
        if not decision.parts:
            return
    from .turns import prepare_planned_meme, is_meme_only, send_response_turn
    planned_meme = await prepare_planned_meme(decision)
    messages = await build_context_messages(
        history=history,
        session_id=session_id,
        query="戳一戳",
        user_id=user_id,
        group_id=group_id,
        user_content=poke_message,
        user_name=sender_name,
        related_user_id=user_id,
        reply_policy=POKE_REPLY,
        planner_decision=decision,
        planned_meme=planned_meme,
    )

    # 调用模型
    try:
        response_text = "" if is_meme_only(decision) else await model_manager.run_turn(
            messages, model=memory_manager.get_session_model(user_id, group_id) or None,
            role="chat", caller="events:trigger", session_id=session_id,
            reply_policy=POKE_REPLY, required_tool=decision.required_tool,
            media_followup=decision.media_followup,
        )
        cleaned_response = enforce_reply_budget(clean_response(response_text), POKE_REPLY)
        from .tools import (
            _meme_context,
            format_sent_meme_for_memory,
            meme_was_sent_this_round,
        )
        from .utils import is_meta_meme_filler
        if meme_was_sent_this_round() and is_meta_meme_filler(cleaned_response):
            logger.info(f"[meme] 戳一戳过滤废话文字回复: {cleaned_response!r}")
            cleaned_response = ""

        from .agent_runner import deliver_reply as _deliver_agent_reply
        async def send_text(text, first):
            ensure_send_allowed(group_id)
            if group_id:
                await _deliver_agent_reply(response_text, text, lambda: bot.send_group_msg(
                    group_id=group_id, message=MessageSegment.text(text)
                ), bot=bot, group_id=group_id)
            else:
                await _deliver_agent_reply(response_text, text, lambda: bot.send_private_msg(user_id=user_id, message=text))

        sent_turn = await send_response_turn(
            text=cleaned_response, decision=decision, planned_meme=planned_meme,
            reply_policy=POKE_REPLY, send_text=send_text,
        )
        cleaned_response = sent_turn.text_for_memory
        history_text = cleaned_response
        if meme_was_sent_this_round():
            sent = (_meme_context.get() or {}).get("last_sent") or {}
            meme_memory = format_sent_meme_for_memory(sent)
            history_text = f"{meme_memory} {cleaned_response}".strip() if meme_memory else cleaned_response

        from .tools import (format_sent_image_for_memory, get_last_sent_image,
                            format_sent_voice_for_memory, get_last_sent_voice)
        history_text = "\n".join(p for p in (
            format_sent_image_for_memory(get_last_sent_image()),
            format_sent_voice_for_memory(get_last_sent_voice()), history_text,
        ) if p)
        # Only confirmed deliveries enter memory.
        if not history_text:
            return
        await memory_manager.add_message(0 if group_id else user_id, "assistant", history_text, group_id, model_spec=getattr(response_text, "model_spec", None) if cleaned_response else None)
        if group_id:
            from .proactive import proactive_manager
            proactive_manager.mark_just_spoke(group_id)

        logger.info(f"[poke] 已响应戳一戳: user={user_id}, group={group_id}")

    except Exception as e:
        logger.error(f"[poke] 处理戳一戳失败: {e}")
