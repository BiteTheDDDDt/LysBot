"""Conversation deadlines and planner-to-renderer handoff.

The caller supplies the model manager; this module does not construct clients or
choose providers. All candidate requests still go through manager._chat_impl.
"""
import asyncio
import json
from dataclasses import replace
from typing import List, Optional

from .config import config
from .logging_setup import general_logger
from .response_safety import is_tool_call_text
from .model_types import (
    Message, ChatResponse, ModelTurnTimeout, current_session_id, current_user_message,
    _attach_response_context, _add_usage,
)


async def execute_model_call(
    manager,
    messages: List[Message],
    model: Optional[str] = None,
    role: Optional[str] = None,
    caller: str = "?",
    session_id: Optional[str] = None,
    allow_fallback: bool = True,
    required_tool: Optional[str] = None,
    media_followup: bool = True,
    _orchestrate: bool = False,
    progress_callback=None,
    reply_policy=None,
) -> str:
    # 解析最终 session_id：显式 > contextvar > _unknown_
    sid = session_id or current_session_id.get() or "_unknown_"
    token = current_session_id.set(sid)
    run_token = None
    runner = None
    try:
        effective_role = role or "chat"
        budget = {
            "planner": config.planner_total_timeout,
            "summary": config.summary_total_timeout,
            "chat": config.chat_turn_timeout,
        }.get(effective_role, config.background_llm_total_timeout)
        from .capabilities import MEDIA_TOOLS
        if effective_role == "chat" and required_tool in MEDIA_TOOLS:
            budget = max(float(budget), float(config.chat_media_turn_timeout))
        budget = max(0.1, float(budget))
        if _orchestrate:
            from .agent_runner import AgentRunner, current_run
            runner = AgentRunner(sid,budget)
            runner.progress_callback = progress_callback
            runner.reply_policy = reply_policy
            run_token = current_run.set(runner)
            await runner.start(messages)
        # This deadline includes model planning/tool rounds, media generation,
        # the QQ send and output description. HTTP timeouts remain per request.
        async def execute_turn():
            from .conversation import activity_scope
            from .network_retry import retry_scope
            with activity_scope(caller or effective_role, kind="turn", session_id=sid,
                                run_id=runner.id if runner else None), retry_scope(budget):
                return await (manager._run_stages if _orchestrate else manager._chat_impl)(
                    messages, model=model, role=role, caller=caller, session_id=sid,
                    allow_fallback=allow_fallback, required_tool=required_tool,
                    media_followup=media_followup,
                )
        task = asyncio.create_task(execute_turn())
        try:
            async with asyncio.timeout(max(0.01, runner.remaining()) if runner else budget) as turn_timeout:
                if runner:
                    runner.timeout_scope = turn_timeout
                response = await task
            if runner:
                if not isinstance(response,ChatResponse):
                    response=ChatResponse(str(response or ""))
                response.agent_run=runner
                response.reply_policy=runner.reply_policy
                await runner.finish("complete",response)
            return response
        except asyncio.TimeoutError as error:
            if not turn_timeout.expired():
                raise  # Preserve a nested timeout; it did not exhaust this turn.
            if runner and runner.render_fallback is not None:
                await runner.validate_scope()
                response = runner.render_fallback
                response.agent_run = runner
                response.reply_policy = runner.reply_policy
                general_logger.warning(f"[render_fallback] session={sid} caller={caller} turn deadline; deliver prepared answer")
                await runner.finish("complete", response)
                return response
            timeout_error = ModelTurnTimeout(
                role=effective_role, required_tool=required_tool,
                timeout=(runner.deadline - runner.started_at if runner else budget),
            )
            general_logger.error(f"[turn_timeout] session={sid} caller={caller} {timeout_error}")
            raise timeout_error from error
    finally:
        try:
            if run_token is not None:
                try:
                    if runner.status=="running":
                        await runner.finish("interrupted")
                finally:
                    current_run.reset(run_token)
        finally:
            current_session_id.reset(token)


def _planning_background(messages):
    """Encode prior turns as records, without giving them live chat roles."""
    entries = []
    images = []
    for message in messages:
        entry = {
            "role": "reference" if message.is_reference or message.is_summary else message.role,
            "content": message.content,
        }
        if message.name:
            entry["name"] = message.name
        if message.sender_id is not None:
            entry["sender_id"] = message.sender_id
        if message.timestamp:
            entry["timestamp"] = message.timestamp
        if message.images:
            entry["image_indices"] = list(range(len(images) + 1, len(images) + len(message.images) + 1))
            images.extend(message.images)
        entries.append(entry)
    return Message(
        role="user", is_reference=True, images=images or None,
        content="【背景资料】按需参考历史对话和检索线索。本轮触发消息单列在后。\n"
                + "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries),
    )


async def run_stages(manager, messages, *, model, role, caller, session_id,
                      allow_fallback, required_tool, media_followup):
    from .agent_runner import current_run
    from .conversation_style import PLANNING_STYLE, RENDERING_STYLE, build_rendering_handoff
    from .qq_addressing import ADDRESSING_GUIDE

    runner = current_run.get()
    runner.revise_output = True
    request = current_user_message(messages)
    instructions = [m for m in messages if m.role == "system"]
    background = [m for m in messages if m is not request and m.role != "system" and not m.is_thinking]
    optional_participation = runner.reply_policy and runner.reply_policy.optional_participation
    addressing = getattr(runner.reply_policy, "addressing", None)
    if addressing is not None and addressing.candidates:
        planning_address_message = Message(role="system", content=(
            "最终 JSON 的 addressing 用于指定送达。暂定送达符合本轮原请求时可以沿用，空数组或 null 清除对应选择。"
            + ADDRESSING_GUIDE
            + "若本轮最终回复确实请一位群友补图，在最终 JSON 顶层填 requested_image_from 为该人的候选 QQ 号，收到图时可以接回本轮。其余情况省略。"
            + "转交任务时，content 写直接对收话人说的请求，无需先代他们完成任务。"
            "媒体工具沿用参与规划的送达，媒体已经发送后本轮送达保持不变。引用从以下消息候选选；收话人和 @ 也可以用本轮 get_group_facts(scope=member) 确认在本群的成员。以下是真实事件候选ID。\n"
            + json.dumps([row.as_candidate(allow_reply=addressing.allow_reply)
                          for row in addressing.candidates], ensure_ascii=False)
        ))
    planning_messages = [*instructions, Message(role="system", content=(
        PLANNING_STYLE + "\n较长任务可用 report_progress 告知已有发现或实际阻碍，不为进展拖延任务。"
    ))]
    if runner.reply_policy and runner.reply_policy.scene == "fact_check":
        planning_messages.append(Message(role="system", content=(
            "按指定主题或引用直接查证。只有调查对象还不明确时，用 start_fact_check 回看对话，挑一句值得查的。"
            "同一批资料读过即可，后续查询针对尚有疑问的部分。"
            "直接说查到什么，顺手带上依据。"
        )))
    if addressing is not None:
        planning_messages.append(Message(role="system", content=addressing.system_hint()))
        if addressing.candidates:
            planning_messages.append(planning_address_message)
        if addressing.handoff:
            planning_messages.append(Message(role="system", content=(
                "本轮送达已选择转交任务。用一句话直接交代原请求，按需确认收话人和任务内容。"
            )))
    if background:
        planning_messages.append(_planning_background(background))
    if runner.working_context:
        planning_messages.append(Message(role="user", content=runner.working_context, is_reference=True))
    if request is not None:
        label = "【本轮群聊消息】\n" if optional_participation else "【本轮请求】\n"
        planning_messages.append(replace(request, content=label + request.content))
    plan = await manager._chat_impl(
        planning_messages, model=None, role="planner", caller=caller + ":plan",
        session_id=session_id, allow_fallback=allow_fallback,
        required_tool=required_tool, media_followup=media_followup, tools_enabled=True,
    )
    if runner.media_delivered:
        followup = media_followup if runner.text_parts is None else runner.text_parts > 0
        if not followup:
            return ChatResponse("", usage=getattr(plan, "usage", None))
    if not str(plan).strip():
        return plan
    runner.planning_result = str(plan)[:4000]
    await runner.checkpoint()
    await runner.validate_scope()
    if optional_participation and not runner.should_reply:
        general_logger.info("[群聊参与] 执行规划选择旁听，跳过本轮文字生成")
        return ChatResponse("", usage=getattr(plan, "usage", None))
    # The execution planner has the latest context and can shorten or expand
    # ordinary replies. An earlier participation decision is only provisional.
    policy = runner.reply_policy
    if policy and policy.scene in {"chat", "multiuser", "close", "proactive", "poke"} and not required_tool and runner.reply_mode:
        runner.reply_policy = replace(policy, detailed=runner.reply_mode == "detailed" and policy.scene in {"chat", "multiuser"}, tiny=runner.reply_mode == "tiny")
    if runner.reply_policy and runner.text_parts:
        policy_for_delivery = runner.reply_policy
        count = min(runner.text_parts, policy_for_delivery.max_text_parts)
        if required_tool or policy_for_delivery.scene in {"close", "jrrp"}:
            count = 1
        runner.reply_policy = replace(policy_for_delivery, text_parts=count)
    draft = str(plan) if runner.identity_verdict else runner.answer_draft
    if runner.response_mode == "factual" and draft and not is_tool_call_text(draft):
        runner.render_fallback = _attach_response_context(
            ChatResponse(draft, usage=getattr(plan, "usage", None)),
            model_spec=getattr(plan, "model_spec", None), primary_spec=getattr(plan, "primary_spec", None),
            fallback_attempt=getattr(plan, "fallback_attempt", 0),
            role="chat", caller=caller + ":render_fallback", session_id=session_id,
        )
    rendering_messages = instructions.copy()
    if addressing is not None:
        rendering_messages.append(Message(role="system", content=addressing.rendering_hint(allow_inline=not runner.media_delivered)))
    if runner.reply_policy:
        old_hint, final_hint = policy.system_hint(), runner.reply_policy.system_hint()
        if any(old_hint in m.content for m in rendering_messages):
            rendering_messages = [replace(m, content=m.content.replace(old_hint, final_hint)) for m in rendering_messages]
        else:
            rendering_messages.append(Message(role="system", content=final_hint))
    if runner.response_mode in {"social", "mixed"}:
        from .image_observations import unsolicited_view

        recent = [m for m in background if m.role in {"user", "assistant"} and not m.is_summary and not m.is_reference][-6:]
        if recent:
            rendering_messages.append(Message(role="user", is_reference=True,
                content="最近对话，用于理解指代、语气和梗；事实依据见本轮交接。\n" +
                        "\n".join(f"{m.role}: {unsolicited_view(m.content)[:400]}" for m in recent)))
    rendering_messages.extend([
        Message(role="system", content=RENDERING_STYLE),
        Message(role="system", content=build_rendering_handoff(
            str(plan), response_mode=runner.response_mode,
            requirements=runner.reply_requirements, answer_draft=draft,
            identity=bool(runner.identity_verdict),
        )),
    ])
    if request is not None:
        label = "群聊触发消息：" if optional_participation else "用户本轮问题："
        rendering_messages.append(Message(role="user", content=label + request.content))
    try:
        result = await manager._chat_impl(
            rendering_messages, model=model, role="chat", caller=caller + ":render",
            session_id=session_id, allow_fallback=allow_fallback,
            required_tool=None, media_followup=False,
        )
    except Exception:
        if runner.render_fallback is None:
            raise
        await runner.validate_scope()
        general_logger.warning(f"[render_fallback] session={session_id} caller={caller} rendering failed; deliver prepared answer")
        return runner.render_fallback
    # Identity is a verified account mapping, not an invitation to reinterpret it.
    if runner.identity_verdict and str(result) != str(plan):
        result = _attach_response_context(
            ChatResponse(str(plan), usage=result.usage), model_spec=result.model_spec,
            primary_spec=result.primary_spec, fallback_attempt=result.fallback_attempt,
            role="chat", caller=caller + ":render", session_id=session_id,
        )
    result.usage = _add_usage(getattr(plan, "usage", None) or {}, getattr(result, "usage", None) or {})
    return result
