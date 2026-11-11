import asyncio
import base64
import json
import logging
import time
from copy import deepcopy
from inspect import signature
from contextvars import ContextVar
from typing import Dict, Any, List, Optional
from functools import wraps
from .conversation import ensure_send_allowed, session_key, activity_scope
from .capabilities import (CAPABILITIES, READ_TOOLS, BASE_CHAT_TOOLS, MEDIA_CHAT_TOOLS,
                           AVAILABLE_TOOLS as AVAILABLE_TOOLS, bind_handlers)
# Preserve existing imports while the declaration source lives in one module.
from .capabilities import (
    SEARCH_WEB_TOOL as SEARCH_WEB_TOOL, POKE_TOOL as POKE_TOOL,
    VOICE_TOOL as VOICE_TOOL, IMAGE_GENERATION_TOOL as IMAGE_GENERATION_TOOL,
    RECALL_HISTORY_TOOL as RECALL_HISTORY_TOOL, READ_HISTORY_TOOL as READ_HISTORY_TOOL,
    READ_WEB_TOOL as READ_WEB_TOOL, QUERY_HISTORY_TOOL as QUERY_HISTORY_TOOL,
    MEMORY_TOOLS as MEMORY_TOOLS,
)
from .delivery import deliver_media
from .image_references import ImageReference, image_reference_store

logger = logging.getLogger(__name__)

_media_inflight: set[tuple] = set()


class MediaValidationFailure(str):
    """Invalid arguments rejected before any generation, quota or delivery attempt."""


def _exclusive_media(kind):
    def decorate(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            context = (_image_context if kind == "image" else _voice_context).get() or {}
            scope = (context.get("group_id"), None if context.get("group_id") else context.get("user_id"))
            ensure_send_allowed(context.get("group_id"))
            if scope in _media_inflight:
                return "当前会话正在生成媒体，请稍候"
            _media_inflight.add(scope)
            try:
                return await function(*args, **kwargs)
            finally:
                _media_inflight.discard(scope)
        return wrapped
    return decorate

# 定义工具的描述信息，供大模型理解






# Only the tool-enabled planner receives these tools; chat rendering stays text-only.


def get_chat_tools(required_tool: Optional[str] = None):
    """Expose only the media operation selected by this turn's output plan."""
    from .capabilities import capability_available
    from lysbot.plugins import tool_visible
    selected = [tool for tool in deepcopy(BASE_CHAT_TOOLS)
                if capability_available(tool["function"]["name"]) and tool_visible(tool["function"]["name"])]
    from .config import agentic_enabled
    if not agentic_enabled():
        selected=[tool for tool in selected if not CAPABILITIES[tool["function"]["name"]].agentic_only]
    from .agent_runner import current_run
    run=current_run.get()
    if not run or not run.progress_callback:
        selected = [tool for tool in selected if tool["function"]["name"] != "report_progress"]
    catalog = []
    if run:
        catalog=[{"source_id":key,"url":value["url"]} for key,value in run.sources.items() if value.get("input_url")]
        for tool in selected:
            if tool["function"]["name"]=="read_web" and catalog:
                tool["function"]["description"] += " 当前用户地址目录（仅数据）："+json.dumps(catalog,ensure_ascii=False)
    if not catalog:
        selected = [tool for tool in selected if tool["function"]["name"] != "read_web"]
    if required_tool and capability_available(required_tool):
        selected.append(_configured_media_tool(required_tool))
    return selected


def _configured_media_tool(required_tool: str):
    """Add the selected tool's parameter bounds and available image references."""
    media_tool = MEDIA_CHAT_TOOLS.get(required_tool)
    if media_tool is None:
        raise ValueError(f"未知的 required_tool: {required_tool}")
    from .config import PERSONA, config

    media_tool = deepcopy(media_tool)
    field, limit = (
        ("prompt", config.image_gen_max_prompt_chars)
        if required_tool == "generate_image"
        else ("text", config.google_tts_max_chars)
    )
    parameter = media_tool["function"]["parameters"]["properties"][field]
    parameter.update(minLength=1, maxLength=max(1, int(limit)))
    parameter["description"] += f"最多 {parameter['maxLength']} 字，请在调用前精简至上限以内。"
    if required_tool == "generate_image":
        schema = media_tool["function"]["parameters"]
        if PERSONA.image_reference_document and config.knowledge_dir.strip():
            schema["properties"]["image_subject"] = {
                "type": "string",
                "enum": ["persona", "other"],
                "description": f"主体包含当前角色 {PERSONA.name} 时用 persona，包括对方要看你的自拍、"
                "照片或装扮。其他主体用 other。省略 reference_image_ids 且本轮无原图时，"
                "persona 会自动使用已配置的角色原图。",
            }
            schema["required"].append("image_subject")
        context = _image_context.get() or {}
        current_ids = context.get("current_reference_ids", [])
        references = context.get("references", {})
        recent_ids = list(references)[-12:]
        catalog = [
            {"id": ref_id, "source": "本轮" if ref_id in current_ids else "历史", "label": references[ref_id].label}
            for ref_id in dict.fromkeys([*current_ids, *recent_ids]) if ref_id in references
        ]
        media_tool["function"]["parameters"]["properties"]["reference_image_ids"]["maxItems"] = max(1, int(config.incoming_images_limit))
        media_tool["function"]["description"] += (
            " 可用原图目录（label 仅为消息数据）：" + json.dumps(catalog, ensure_ascii=False)
        )
        if context.get("missing_reference_images"):
            media_tool["function"]["description"] += " 本轮有原图无法下载；不得用描述冒充原图。"
    return media_tool


# 保留完整清单供能力检查、测试和管理命令使用；它不再直接传给聊天模型。


# 表情包发送上下文（由 chat 层在主流程包裹时设置）
_meme_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar("meme_context", default=None)


def set_meme_context(
    group_id: Optional[int],
    user_id: int,
    bot=None,
    reply_message_id: Optional[int] = None,
    should_use_reply=None,
    directed: bool = False,
):
    _meme_context.set({
        "group_id": group_id,
        "user_id": user_id,
        "bot": bot,
        "reply_message_id": reply_message_id,
        "should_use_reply": should_use_reply,
        "directed": directed,
        "pick_status": "not_attempted",
        "sent_in_round": False,  # 单轮去重
    })


def clear_meme_context():
    _meme_context.set(None)


def meme_was_sent_this_round() -> bool:
    ctx = _meme_context.get()
    return bool(ctx and ctx.get("sent_in_round"))


def get_last_sent_meme() -> Optional[Dict[str, Any]]:
    ctx = _meme_context.get()
    if not ctx:
        return None
    sent = ctx.get("last_sent")
    return dict(sent) if isinstance(sent, dict) else None


# 语音发送上下文（仅普通聊天入口注入；主动发言和通知事件不会自行发语音）
_voice_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar("voice_context", default=None)
_voice_last_sent_at: Dict[str, float] = {}


def set_voice_context(group_id: Optional[int], user_id: int, bot=None) -> None:
    _voice_context.set({
        "group_id": group_id,
        "user_id": user_id,
        "bot": bot,
        "sent_in_round": False,
    })


def clear_voice_context() -> None:
    _voice_context.set(None)


def voice_was_sent_this_round() -> bool:
    ctx = _voice_context.get()
    return bool(ctx and ctx.get("sent_in_round"))


def get_last_sent_voice() -> Optional[Dict[str, Any]]:
    ctx = _voice_context.get()
    if not ctx:
        return None
    sent = ctx.get("last_sent")
    return dict(sent) if isinstance(sent, dict) else None


def format_sent_voice_for_memory(sent: Optional[Dict[str, Any]]) -> str:
    if not sent:
        return ""
    transcript = " ".join(str(sent.get("text") or "").split()).strip()
    if len(transcript) > 1000:
        transcript = transcript[:1000] + "…"
    emotion = str(sent.get("emotion") or "neutral")
    voice = str(sent.get("voice") or "?")
    model = str(sent.get("model") or "?")
    status = "回执未确认，不可断言对方已收到；" if sent.get("delivery_status") == "unconfirmed" else ""
    return (
        f"【生成语音 model={model} voice={voice} emotion={emotion}】"
        f"{status}实际台词：{transcript}"
    )


# AI 图片发送上下文（只在普通聊天入口注入）。
_image_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar("image_context", default=None)
_image_last_sent_at: Dict[str, float] = {}


def set_image_context(
    group_id: Optional[int], user_id: int, bot=None, *,
    reference_images: Optional[list[ImageReference]] = None,
    missing_reference_images: bool = False,
    allow_auto_reference: bool = True,
) -> None:
    references = image_reference_store.snapshot(session_key(user_id, group_id))
    current = {ref.id: ref for ref in (reference_images or [])}
    references.update(current)
    _image_context.set({
        "group_id": group_id,
        "user_id": user_id,
        "bot": bot,
        "sent_in_round": False,
        "references": references,
        "current_reference_ids": list(current),
        "missing_reference_images": missing_reference_images,
        "allow_auto_reference": allow_auto_reference,
    })


def clear_image_context() -> None:
    _image_context.set(None)


def image_was_sent_this_round() -> bool:
    ctx = _image_context.get()
    return bool(ctx and ctx.get("sent_in_round"))


def get_last_sent_image() -> Optional[Dict[str, Any]]:
    ctx = _image_context.get()
    if not ctx:
        return None
    sent = ctx.get("last_sent")
    return dict(sent) if isinstance(sent, dict) else None


def format_sent_image_for_memory(sent: Optional[Dict[str, Any]]) -> str:
    if not sent:
        return ""
    prompt = " ".join(str(sent.get("prompt") or "").split()).strip()
    if len(prompt) > 500:
        prompt = prompt[:500] + "…"
    description = " ".join(str(sent.get("description") or "").split()).strip()
    if len(description) > 500:
        description = description[:500] + "…"
    model = str(sent.get("model") or "?")
    size = str(sent.get("size") or "?")
    parts = [f"【生成图片 model={model} size={size}】"]
    if sent.get("reference_id"):
        parts.append(f"[图片引用:{sent['reference_id']}]")
    if sent.get("delivery_status") == "unconfirmed":
        parts.append("回执未确认，不可断言对方已收到")
    if description:
        parts.append(f"实际画面：{description}")
    if prompt:
        parts.append(f"生成依据：{prompt}")
    return " ".join(parts)


def format_sent_meme_for_memory(sent: Optional[Dict[str, Any]]) -> str:
    if not sent:
        return ""
    topic = sent.get("topic") or "?"
    meme_id = sent.get("id")
    message_id = sent.get("message_id")
    description = (sent.get("description") or "").strip()
    reason = (sent.get("reason") or "").strip()

    parts = [f"topic={topic}"]
    if meme_id is not None:
        parts.append(f"id={meme_id}")
    if message_id is not None:
        parts.append(f"message_id={message_id}")
    if description:
        parts.append(f"内容={description}")
    if reason:
        parts.append(f"原因={reason}")
    return "【我发送了一张表情包】" + "，".join(parts)


def format_sent_meme_for_prompt(sent: Optional[Dict[str, Any]]) -> str:
    memory_text = format_sent_meme_for_memory(sent)
    if not memory_text:
        return ""
    return (
        "【本轮已发送表情包-内部参考】\n"
        f"{memory_text}\n"
        "这张表情包已经由你发出；如果继续文字回复，要把它当成自己的上一句动作，"
        "不要表现得不知道图片内容，也不要再解释你无法选择具体图片。"
    )


def format_meme_selection_failure() -> str:
    status = (_meme_context.get() or {}).get("pick_status", "unavailable")
    reason = {
        "cooldown": "主动表情发送处于群级冷却，不是没有合适表情",
        "already_sent": "本轮已经发送过表情，不能重复发送",
        "no_candidate": "当前分类没有可发送的合格图片",
        "selection_error": "表情选择过程出错，未取得图片",
    }.get(status, "本轮没有取得可发送表情，具体原因未确认")
    return (f"【本轮表情预选结果】status={status}；{reason}。本次预选没有发送新表情，不代表没有图库。"
            "改用文字自然接话，保留用户的玩笑或表达意图，不主动汇报选图过程。"
            "用户追问失败原因时按上述实际状态说明，不能编造图库不足或假装已发送。")


def format_planned_meme_for_prompt(planned: Optional[Dict[str, Any]]) -> str:
    if not planned:
        return ""
    topic = planned.get("topic") or "?"
    description = (planned.get("description") or "").strip()
    reason = (planned.get("reason") or "").strip()

    parts = [f"topic={topic}"]
    if description:
        parts.append(f"内容={description}")
    if reason:
        parts.append(f"原因={reason}")
    return (
        "【本轮预选表情包-内部参考】\n"
        + "，".join(parts)
        + "\n这张表情包尚未发送，但系统可能会把它作为本轮其中一句话发出。"
        "你只需要输出自然语言文字；不要解释你无法选择图片，也不要复述 JSON 或工具调用。"
    )


async def pick_meme(topic: str, reason: str = "") -> Optional[Dict[str, Any]]:
    """只选择一张表情包，不发送。用于让 LLM 先知道本轮可能发什么图。"""
    from .memes import meme_manager, ALLOWED_TOPICS

    if topic not in ALLOWED_TOPICS:
        logger.warning(f"[pick_meme] 未知 topic={topic!r}")
        return None

    ctx = _meme_context.get() or {}
    bot = ctx.get("bot")
    group_id = ctx.get("group_id")

    if not bot or not group_id:
        return None

    # 单轮去重 + 群级冷却。
    if ctx.get("sent_in_round"):
        ctx["pick_status"] = "already_sent"
        return None
    if not ctx.get("directed") and meme_manager.is_on_cooldown(group_id):
        ctx["pick_status"] = "cooldown"
        return None

    try:
        picked = await meme_manager.pick_random(topic)
    except Exception:
        ctx["pick_status"] = "selection_error"
        raise
    if not picked:
        ctx["pick_status"] = "no_candidate"
        return None
    ctx["pick_status"] = "ready"
    meme_id, file_path, description = picked
    planned = {
        "id": meme_id,
        "file_path": file_path,
        "topic": topic,
        "description": description,
        "reason": reason,
    }
    ctx["planned_meme"] = planned
    _meme_context.set(ctx)
    return dict(planned)


async def send_picked_meme(planned: Optional[Dict[str, Any]], allow_reply: bool = True) -> str:
    """发送已预选的表情包。"""
    from .memes import meme_manager

    if not planned:
        return "empty"

    ctx = _meme_context.get() or {}
    bot = ctx.get("bot")
    group_id = ctx.get("group_id")

    if not bot or not group_id:
        return "send_meme 不可用：当前不是群聊上下文"

    if ctx.get("sent_in_round"):
        return "本轮回复已发过表情包，请用文字回应"

    meme_id = planned.get("id")
    file_path = planned.get("file_path")
    topic = planned.get("topic") or "?"
    description = planned.get("description") or ""
    reason = planned.get("reason") or ""

    try:
        from pathlib import Path
        p = Path(file_path)
        if not p.exists():
            logger.warning(f"[send_meme] 文件不存在 id={meme_id} path={file_path}")
            return "empty"
        # 用 base64 内联发图，兼容跨机部署的 NapCat（file:// 只能同机）
        import base64 as _b64
        raw = await asyncio.to_thread(p.read_bytes)
        b64 = _b64.b64encode(raw).decode("ascii")
        reply_prefix = ""
        reply_message_id = ctx.get("reply_message_id")
        should_use_reply = ctx.get("should_use_reply")
        if allow_reply and reply_message_id is not None and callable(should_use_reply):
            try:
                if should_use_reply():
                    reply_prefix = f"[CQ:reply,id={reply_message_id}]"
            except Exception as e:
                logger.debug(f"[send_meme] 判断 reply 引用失败: {e}")
        ensure_send_allowed(group_id)
        send_result = await bot.send_group_msg(
            group_id=group_id,
            message=f"{reply_prefix}[CQ:image,file=base64://{b64}]",
        )
    except Exception as e:
        logger.error(f"[send_meme] 发送失败 id={meme_id}: {e}")
        return f"send_meme 发送失败: {e}"

    meme_manager.mark_sent(group_id)
    ctx["sent_in_round"] = True
    ctx["last_sent"] = {
        "id": meme_id,
        "message_id": send_result.get("message_id") if isinstance(send_result, dict) else None,
        "topic": topic,
        "description": description,
        "reason": reason,
    }
    _meme_context.set(ctx)
    try:
        await meme_manager.mark_used(meme_id)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.warning(f"[send_meme] 已发送，更新使用计数失败 id={meme_id}: {error}")
    return f"已发送 {topic} 表情包：{(description or '')[:80]}"


@_exclusive_media("voice")
async def send_voice(text: str, emotion: str = "neutral") -> str:
    """Generate a Gemini TTS clip and immediately send it as a QQ record message."""
    from .config import config
    from .tts import EMOTION_PROMPTS, TTSGenerationError, generate_speech

    ctx = _voice_context.get() or {}
    bot = ctx.get("bot")
    group_id = ctx.get("group_id")
    user_id = ctx.get("user_id")

    if not bot or not user_id:
        return "send_voice 不可用：当前没有可发送语音的聊天上下文"
    from .roles import role_available
    if not role_available("tts"):
        return "send_voice 不可用：语音生成功能已关闭"
    if ctx.get("sent_in_round"):
        return "本轮已经发送过语音，请不要重复调用"
    if image_was_sent_this_round():
        return "本轮已经发送过图片，请不要再发送语音"

    if not isinstance(text, str):
        return MediaValidationFailure("send_voice 失败：text 必须是字符串")
    transcript = text.replace("\x00", "").strip()
    if not transcript:
        return MediaValidationFailure("send_voice 失败：朗读内容不能为空")
    max_chars = max(1, int(config.google_tts_max_chars))
    if len(transcript) > max_chars:
        return MediaValidationFailure(f"send_voice 失败：朗读内容超过 {max_chars} 字（实际 {len(transcript)} 字）。请缩短到上限以内；要求原文朗读时不要擅自删改，应说明长度限制")
    if not isinstance(emotion, str) or emotion not in EMOTION_PROMPTS:
        return MediaValidationFailure(f"send_voice 失败：不支持的 emotion '{emotion}'")

    scope_key = f"group:{group_id}" if group_id else f"private:{user_id}"
    cooldown = max(0.0, float(config.google_tts_cooldown))
    remaining = cooldown - (time.monotonic() - _voice_last_sent_at.get(scope_key, 0.0))
    if remaining > 0:
        return f"send_voice 暂时冷却中，请约 {remaining:.0f} 秒后再试"

    try:
        speech = await generate_speech(transcript, emotion=emotion)
    except TTSGenerationError as exc:
        logger.warning(f"[send_voice] 生成失败 scope={scope_key}: {exc}")
        return f"send_voice 生成失败：{exc}"
    except Exception as exc:
        logger.exception(f"[send_voice] 未预期的生成错误 scope={scope_key}")
        return f"send_voice 生成失败：{exc}"

    try:
        delivery_status, send_result = await deliver_media(
            bot, group_id=group_id, user_id=user_id, media_type="voice", data=speech.wav_bytes)
    except Exception as exc:
        logger.error(f"[send_voice] QQ投递明确失败 scope={scope_key}: {exc}")
        return f"send_voice 发送失败：{exc}"

    _voice_last_sent_at[scope_key] = time.monotonic()
    if len(_voice_last_sent_at) > 1000:
        expiry = time.monotonic() - max(3600.0, cooldown)
        for key, sent_at in list(_voice_last_sent_at.items()):
            if sent_at < expiry:
                _voice_last_sent_at.pop(key, None)

    ctx["sent_in_round"] = True
    ctx["last_sent"] = {
        "delivery_status": delivery_status,
        "message_id": send_result.get("message_id") if isinstance(send_result, dict) else None,
        "text": transcript,
        "emotion": emotion,
        "voice": config.google_tts_voice,
        "model": speech.model_spec,
        "source_mime_type": speech.source_mime_type,
    }
    _voice_context.set(ctx)
    logger.info(
        f"[send_voice] 已发送 scope={scope_key} chars={len(transcript)} "
        f"emotion={emotion} voice={config.google_tts_voice}"
    )
    return json.dumps(
        {
            "status": delivery_status,
            "delivery_status": delivery_status,
            "media_type": "voice",
            "transcript": transcript,
            "emotion": emotion,
            "voice": config.google_tts_voice,
            "model": speech.model_spec,
            "instruction": (
                "语音发送已尝试，delivery_status=unconfirmed 时回执不确定，不可声称对方已收到。不要重复朗读 transcript，也不要再次调用媒体工具；"
                "是否补充文字由本轮回复规划决定。"
            ),
        },
        ensure_ascii=False,
    )


@_exclusive_media("image")
async def generate_image(
    prompt: str, orientation: str = "auto", reference_image_ids: Optional[list[str]] = None,
    image_subject: Optional[str] = None,
) -> str:
    """Generate an image through MODEL_IMAGE_GEN and immediately send it to QQ."""
    from .config import PERSONA, config
    from .knowledge import KnowledgeStore
    from .image_generation import (
        ImageGenerationError,
        generate_image as generate_image_bytes,
    )

    ctx = _image_context.get() or {}
    bot = ctx.get("bot")
    group_id = ctx.get("group_id")
    user_id = ctx.get("user_id")

    if not bot or not user_id:
        return "generate_image 不可用：当前没有可发送图片的聊天上下文"
    from .roles import role_available
    if not role_available("image_gen"):
        return "generate_image 不可用：图片生成功能已关闭"
    if ctx.get("sent_in_round"):
        return "本轮已经发送过生成图片，请不要重复调用"
    if voice_was_sent_this_round():
        return "本轮已经发送过语音，请不要再生成图片"

    if not isinstance(prompt, str):
        return MediaValidationFailure("generate_image 失败：prompt 必须是字符串")
    clean_prompt = prompt.replace("\x00", "").strip()
    if not clean_prompt:
        return MediaValidationFailure("generate_image 失败：图片描述不能为空")
    max_chars = max(1, int(config.image_gen_max_prompt_chars))
    if len(clean_prompt) > max_chars:
        return MediaValidationFailure(f"generate_image 失败：图片描述超过 {max_chars} 字，请精简后再试")
    if orientation not in ("auto", "square", "landscape", "portrait"):
        return MediaValidationFailure("generate_image 失败：orientation 必须是 auto、square、landscape 或 portrait")

    persona_document = PERSONA.image_reference_document if config.knowledge_dir.strip() else ""
    if image_subject not in (None, "persona", "other") or (
        persona_document and image_subject is None and reference_image_ids is None
    ):
        return MediaValidationFailure("请填写 image_subject：画当前角色用 persona，其他主体用 other。")
    max_refs = max(1, int(config.incoming_images_limit))
    if reference_image_ids is None:
        if ctx.get("missing_reference_images"):
            return MediaValidationFailure("参考图片下载不完整，请用户重新发送或明确选择可用原图；不能仅凭文字描述重画。")
        reference_image_ids = ctx.get("current_reference_ids", [])
        if reference_image_ids and not ctx.get("allow_auto_reference", True):
            return MediaValidationFailure("本轮有多位用户的图片，请明确填写 reference_image_ids，避免选错原图。")
        if not reference_image_ids and image_subject == "persona" and persona_document:
            store = KnowledgeStore(config.knowledge_dir)
            try:
                refs = await asyncio.to_thread(store.image_refs, persona_document)
            except (OSError, ValueError):
                refs = []
            if not refs:
                return MediaValidationFailure("已配置的角色参考图不可用，请重新选择可用的图片引用。")
            reference_image_ids = [ref["id"] for ref in refs[:min(2, max_refs)]]
    if not isinstance(reference_image_ids, list) or any(not isinstance(item, str) for item in reference_image_ids):
        return MediaValidationFailure("reference_image_ids 必须是图片引用 ID 的数组")
    reference_image_ids = list(dict.fromkeys(reference_image_ids))
    if len(reference_image_ids) > max_refs:
        return MediaValidationFailure("参考图片过多，请明确选择需要的原图")
    references = ctx.get("references", {})
    originals = []
    for ref_id in reference_image_ids:
        if ref_id in references:
            originals.append(references[ref_id].base64)
        elif ref_id.startswith("ki:") and config.knowledge_dir.strip():
            try:
                original = await asyncio.to_thread(KnowledgeStore(config.knowledge_dir).image_base64, ref_id)
            except (OSError, ValueError):
                return MediaValidationFailure("知识库参考图不可用，请重新读取角色资料并选用有效的图片引用。")
            originals.append(original)
        else:
            return MediaValidationFailure("所选原图已过期或不属于当前会话，请用户引用或重发图片；不能仅凭文字描述重画。")

    scope_key = f"group:{group_id}" if group_id else f"private:{user_id}"
    cooldown = max(0.0, float(config.image_gen_cooldown))
    remaining = cooldown - (time.monotonic() - _image_last_sent_at.get(scope_key, 0.0))
    if remaining > 0:
        return f"generate_image 暂时冷却中，请约 {remaining:.0f} 秒后再试"

    try:
        reference_args = {"reference_images": originals} if originals else {}
        generated = await generate_image_bytes(clean_prompt, orientation=orientation, **reference_args)
    except ImageGenerationError as exc:
        logger.warning(f"[generate_image] 生成失败 scope={scope_key}: {exc}")
        return f"generate_image 生成失败：{exc}"
    except Exception as exc:
        logger.exception(f"[generate_image] 未预期的生成错误 scope={scope_key}")
        return f"generate_image 生成失败：{exc}"

    try:
        delivery_status, send_result = await deliver_media(
            bot, group_id=group_id, user_id=user_id, media_type="image", data=generated.image_bytes)
    except Exception as exc:
        logger.error(f"[generate_image] QQ投递明确失败 scope={scope_key}: {exc}")
        return f"generate_image 发送失败：{exc}"

    _image_last_sent_at[scope_key] = time.monotonic()
    if len(_image_last_sent_at) > 1000:
        expiry = time.monotonic() - max(3600.0, cooldown)
        for key, sent_at in list(_image_last_sent_at.items()):
            if sent_at < expiry:
                _image_last_sent_at.pop(key, None)

    ctx["sent_in_round"] = True
    output_reference = image_reference_store.remember(
        session_key(user_id, group_id), base64.b64encode(generated.image_bytes).decode("ascii"),
        f"bot 生成：{clean_prompt}",
    )
    ctx["last_sent"] = {
        "reference_id": output_reference.id,
        "source_reference_ids": reference_image_ids,
        "delivery_status": delivery_status,
        "message_id": send_result.get("message_id") if isinstance(send_result, dict) else None,
        "prompt": clean_prompt,
        "orientation": orientation,
        "model": generated.model_spec,
        "size": generated.size,
        "mime_type": generated.mime_type,
        "revised_prompt": generated.revised_prompt,
        "description": "",
    }
    _image_context.set(ctx)

    if config.image_gen_describe_output:
        try:
            import base64 as _b64

            from .models import current_session_id, model_manager

            image_b64 = _b64.b64encode(generated.image_bytes).decode("ascii")
            description = (
                await model_manager.describe_image(
                    image_b64,
                    session_id=current_session_id.get(),
                )
            ).strip()
            ctx["last_sent"]["description"] = description
            if description:
                image_reference_store.remember(
                    session_key(user_id, group_id), output_reference.base64, f"bot 生成：{description}",
                )
            _image_context.set(ctx)
            logger.info(
                f"[generate_image] 实际画面描述完成 scope={scope_key} "
                f"description_chars={len(description)}"
            )
        except Exception as exc:
            logger.warning(
                f"[generate_image] 实际画面描述失败，记忆将保留生成依据 "
                f"scope={scope_key}: {exc}"
            )
    logger.info(
        f"[generate_image] 已发送 scope={scope_key} model={generated.model_spec} "
        f"size={generated.size} prompt_chars={len(clean_prompt)}"
    )
    return json.dumps(
        {
            "status": delivery_status,
            "delivery_status": delivery_status,
            "media_type": "image",
            "reference_id": output_reference.id,
            "prompt": clean_prompt,
            "actual_description": str(ctx["last_sent"].get("description") or ""),
            "model": generated.model_spec,
            "size": generated.size,
            "instruction": (
                "图片发送已尝试，delivery_status=unconfirmed 时回执不确定，不可声称对方已收到。不要完整复述 prompt 或画面描述，也不要再次调用媒体工具；"
                "是否补充文字由本轮回复规划决定。"
            ),
        },
        ensure_ascii=False,
    )


# TOOL_FUNCTIONS 在文件下方定义，这里 deferred 注册（见文件末尾）

async def search_web(query: str) -> str:
    """
    网页搜索：优先走 MODEL_SEARCH 路由（GPT Responses 原生搜索或 DashScope enable_search），
    失败则回退到本地代理 + DuckDuckGo。
    """
    from .logging_setup import general_logger
    from .models import model_manager, Message

    log = general_logger.bind(category="llm")
    log.info(f"[search_web] role=search 联网: {query}")
    try:
        prompt = (
            f"请使用联网搜索功能，查找关于 '{query}' 的最新客观信息并总结。"
            "直接输出事实，包含具体数值，不要多余的寒暄。"
        )
        search_result = await model_manager.chat(
            messages=[Message(role="user", content=prompt)],
            role="search",
            caller="tools:search_web",
        )
        log.info(f"[search_web] role=search 成功 preview={str(search_result)[:50]}...")
        return f"搜索结果摘要如下:\n{search_result}"

    except Exception as qwen_e:
        log.warning(f"[search_web] role=search 失败 -> DDG: {qwen_e}")

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            # The fallback is optional; a missing package must not break the tool.
            log.warning("[search_web] duckduckgo-search 未安装，跳过 DDG 兜底")
            return f"搜索失败：模型路由不可用，且未安装 duckduckgo-search 兜底 ({qwen_e})"

        from .providers import get_provider_proxy
        proxy = get_provider_proxy("google")
        options = {"proxy": proxy} if proxy else {}

        def ddg_search():
            with DDGS(timeout=10, **options) as ddgs:
                return list(ddgs.text(query, max_results=8))

        try:
            results = await asyncio.to_thread(ddg_search)
        except Exception as e:
            log.error(f"[search_web] DDG 也挂了: {e}")
            return f"搜索失败：无法访问搜索引擎兜底 ({str(e)})"

        if not results:
            return "未能找到相关的搜索结果摘要，请尝试更换搜索词。"

        result_texts = []
        for i, r in enumerate(results):
            title = r.get('title', '')
            body = r.get('body', '')
            result_texts.append(f"{i+1}. 标题: {title}\n摘要: {body}")

        final_result = "\n\n".join(result_texts)
        log.info(f"[search_web] DDG 成功 preview={final_result[:50]}...")
        return f"搜索结果摘要如下:\n{final_result}"

# 工具路由字典

# 戳一戳操作的上下文存储（由事件处理器设置）
_poke_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar("poke_context", default=None)

def set_poke_context(
    group_id: int, user_id: int, bot=None, history_request_texts=None, *, allow_sender_alias: bool = True,
    current_user_text: str = "",
):
    """设置戳一戳的上下文信息"""
    _poke_context.set({
        "group_id": group_id,
        "user_id": user_id,
        "bot": bot,
        "history_request_texts": history_request_texts or [],
        "allow_sender_alias": allow_sender_alias,
        "current_user_text": current_user_text,
        "poke_results": {},
    })

def clear_poke_context():
    """清除戳一戳上下文"""
    _poke_context.set(None)

async def send_poke(target_id: str, reason: str = "") -> str:
    """One attempt per target in the current turn, including direct callers."""
    context = _poke_context.get()
    if not context:
        return "戳一戳失败：无法获取机器人实例"
    use_sender = str(target_id).lower() == "sender"
    if use_sender and not context.get("allow_sender_alias", True):
        return "戳一戳未执行：本轮有多位发言者，请根据当前请求明确指定目标QQ号，不能使用 sender。"
    target = str(context.get("user_id")) if use_sender else str(target_id)
    try:
        target = str(int(target))
    except ValueError:
        return "戳一戳失败：无效的目标QQ号"
    ensure_send_allowed(context.get("group_id"))
    attempts = context.setdefault("poke_results", {})
    if target in attempts:
        return attempts[target]
    # Set before awaiting so concurrent calls and cancellation cannot resend.
    attempts[target] = "戳一戳发送结果未知，已有执行尝试，不再重复发送"
    result = await _send_poke_once(target, reason)
    attempts[target] = result
    return result


async def _send_poke_once(target_id: str, reason: str = "") -> str:
    """
    发送戳一戳
    """
    context = (_poke_context.get() or {}).copy()

    if not context.get("bot"):
        return "戳一戳失败：无法获取机器人实例"

    bot = context["bot"]
    group_id = context.get("group_id")
    sender_id = context.get("user_id")
    ensure_send_allowed(group_id)

    # 解析目标ID
    if target_id.lower() == "sender":
        target_qq = sender_id
    else:
        try:
            target_qq = int(target_id)
        except ValueError:
            return f"戳一戳失败：无效的目标QQ号 '{target_id}'"

    if not group_id:
        return "戳一戳失败：只能在群聊中使用戳一戳"

    if not target_qq:
        return "戳一戳失败：无法确定目标用户"

    api_errors = []
    try:
        # NapCat / OneBot 实现的 poke action 命名不完全统一，优先尝试 API。
        for action_name in ("send_poke", "group_poke", "send_group_poke"):
            try:
                receipt=await bot.call_api(action_name, group_id=group_id, user_id=target_qq)
                from .agent_runner import record_tool_delivery
                await record_tool_delivery("sent",receipt)
                reason_hint = f"（原因：{reason}）" if reason else ""
                logger.info(f"[send_poke] action={action_name} group={group_id} target={target_qq}")
                return f"戳一戳已发送给 {target_qq}{reason_hint}"
            except Exception as e:
                api_errors.append(f"{action_name}: {e}")
                logger.debug(f"[send_poke] action={action_name} 失败: {e}")
                info=getattr(e,"info",{})
                unsupported=isinstance(info,dict) and (info.get("retcode") in {404,1404} or
                    any(term in str(info.get("wording",info.get("message",""))).lower()
                        for term in ("unknown action","unsupported action","not implemented","不支持的api")))
                if not unsupported:
                    return "戳一戳发送结果未知，已停止自动重试，不能确认对方已收到。"

        # 兜底：部分实现接受 CQ:poke。若平台忽略该 CQ 码，日志里会保留 API 失败信息。
        await bot.send_group_msg(
            group_id=group_id,
            message=f"[CQ:poke,qq={target_qq}]"
        )

        reason_hint = f"（原因：{reason}）" if reason else ""
        logger.info(f"[send_poke] fallback=CQ group={group_id} target={target_qq} api_errors={api_errors[:2]}")
        return (
            f"戳一戳已尝试通过 CQ 码发送给 {target_qq}{reason_hint}，"
            "但当前平台可能不支持，请不要承诺已经戳到"
        )
    except Exception as e:
        detail = "; ".join(api_errors[-3:])
        suffix = f"；API 失败：{detail}" if detail else ""
        return f"戳一戳发送失败: {str(e)}{suffix}"

# 注册戳一戳函数


def _parse_since(since: str) -> Optional[str]:
    """把相对时间 ('24h'/'7d'/'30m') 或绝对 ISO 转成 ISO 时间戳字符串。失败返回 None。"""
    from datetime import datetime, timedelta
    if not since:
        return None
    s = since.strip().lower()
    try:
        if s.endswith("h"):
            return (datetime.now() - timedelta(hours=int(s[:-1]))).isoformat()
        if s.endswith("d"):
            return (datetime.now() - timedelta(days=int(s[:-1]))).isoformat()
        if s.endswith("m") and not s.endswith("month"):
            return (datetime.now() - timedelta(minutes=int(s[:-1]))).isoformat()
        # 当成 ISO 解析
        return datetime.fromisoformat(s).isoformat()
    except Exception:
        return None


async def recall_history(
    query: str = "",
    sources: Optional[List[str]] = None,
    sender_ids: Optional[List[str]] = None,
    sender_names: Optional[List[str]] = None,
    since: str = "",
    until: str = "",
    roles: Optional[List[str]] = None,
    match_mode: str = "all",
    sort: str = "relevance",
    limit: int = 20,
    context_before: int = 0,
    context_after: int = 0,
    # 兼容旧版模型可能仍缓存着的参数名。
    keyword: str = "",
    sender_name: str = "",
    role: str = "all",
    keyword_mode: str = "",
) -> str:
    """让 Chat LLM 自主检索当前会话的多层 memory。"""
    from .models import current_session_id
    from .memory import memory_manager

    sid = current_session_id.get() or "_unknown_"
    if sid == "_unknown_":
        return "召回失败：当前会话上下文未确定，无法检索历史。"

    since_iso = _parse_since(since) if since else None
    if since and not since_iso:
        return f"召回失败：since 参数 '{since}' 无法解析。请用 '24h' / '7d' / '2025-01-01' 这类格式。"
    until_iso = _parse_since(until) if until else None
    if until and not until_iso:
        return f"召回失败：until 参数 '{until}' 无法解析。请用 '24h' / '7d' / '2025-01-01' 这类格式。"

    normalized_query = (query or keyword or "").strip()
    normalized_sender_names = list(sender_names or [])
    if sender_name and sender_name.strip():
        normalized_sender_names.append(sender_name.strip())

    normalized_roles = list(roles or [])
    legacy_role = (role or "all").strip().lower()
    if not normalized_roles and legacy_role in {"user", "assistant", "system"}:
        normalized_roles = [legacy_role]
    normalized_match_mode = (match_mode or keyword_mode or "all").strip().lower()
    if normalized_match_mode not in {"all", "any"}:
        normalized_match_mode = "all"

    try:
        result = await memory_manager.recall_memory(
            session_id=sid,
            query=normalized_query,
            sources=sources,
            sender_ids=sender_ids,
            sender_names=normalized_sender_names,
            since=since_iso,
            until=until_iso,
            roles=normalized_roles,
            match_mode=normalized_match_mode,
            sort=sort,
            limit=limit if isinstance(limit, int) else 20,
            context_before=context_before if isinstance(context_before, int) else 0,
            context_after=context_after if isinstance(context_after, int) else 0,
        )
    except Exception as e:
        logger.warning(f"[recall_history] 检索失败 sid={sid}: {e}")
        return f"召回失败：{e}"

    if not result.get("hits"):
        return "未在历史中找到匹配的消息。"

    # 保留结构化 ID / 来源 / 用户 / 时间，供模型按结果继续缩小检索；
    # 同时限制单条长度，避免一次工具结果挤爆后续对话上下文。
    for hit in result["hits"]:
        content = str(hit.get("content") or "").strip()
        if len(content) > 500:
            hit["content"] = content[:500] + "…"
        for context_item in hit.get("context") or []:
            context_content = str(context_item.get("content") or "").strip()
            if len(context_content) > 300:
                context_item["content"] = context_content[:300] + "…"
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))



async def execute_tool(func_name: str, args: Dict[str, Any]) -> str:
    """执行指定的工具函数"""
    from .capabilities import capability_available
    if not capability_available(func_name):
        return json.dumps({"status": "unavailable", "error": {"code": "capability_unconfigured"}})
    read_only = func_name in READ_TOOLS
    if func_name in TOOL_FUNCTIONS:
        func = TOOL_FUNCTIONS[func_name]
        if func is None:
            return f"工具 {func_name} 尚未初始化"
        if read_only:
            try:
                if not isinstance(args,dict):
                    raise TypeError("arguments_must_be_object")
                signature(func).bind(**args)
            except TypeError as error:
                from .retrieval import failure
                return json.dumps(failure(str(error),invalid=True),ensure_ascii=False)
        if func_name in MEDIA_CHAT_TOOLS:
            # Validate the call signature before entering the function. Do not
            # classify exceptions raised during generation as safe to retry.
            try:
                signature(func).bind(**args)
            except TypeError as error:
                return MediaValidationFailure(f"{func_name} 参数校验失败：{error}")
        try:
            from .agent_runner import current_run
            run = current_run.get()
            with activity_scope(func_name, kind="capability", run_id=run.id if run else None,
                                session_id=run.session_id if run else None):
                return await func(**args)
        except Exception as e:
            if read_only:
                from .retrieval import failure
                from .diagnostics import api_failure
                payload = failure(type(e).__name__,retryable=True)
                payload["error"] = api_failure(e, operation=func_name)
                return json.dumps(payload,ensure_ascii=False)
            return f"执行工具 {func_name} 时发生错误: {str(e)}"
    return f"找不到工具: {func_name}"


# Structured search tools share scoped sources; legacy parameter spellings remain
# accepted for in-flight conversations, but new prompts expose the explicit API.
_legacy_recall_history = recall_history
_legacy_search_web = search_web


async def recall_history(query="", mode="hybrid", sources=None, author_ids=None, subject_ids=None,
                         sender_ids=None, sender_names=None, since="", until="", roles=None,
                         exact_phrase="", match_mode="any", sort="relevance", limit=8,
                         cursor=None, group_by=None, **legacy):
    from .config import agentic_enabled
    if not agentic_enabled():
        return await _legacy_recall_history(query=query,sources=sources,sender_ids=sender_ids or author_ids,
            sender_names=sender_names,since=since,until=until,roles=roles,match_mode=match_mode,
            sort=sort,limit=limit,**legacy)
    from .memory import memory_manager
    from .models import current_session_id
    from .retrieval import HistorySearch,failure
    allowed={"keyword","sender_name","role","keyword_mode","context_before","context_after"}
    if set(legacy)-allowed:
        return json.dumps(failure("unknown_arguments",invalid=True),ensure_ascii=False)
    query=query or legacy.get("keyword","")
    if legacy.get("sender_name"):
        sender_names=[*(sender_names or []),legacy["sender_name"]]
    if not roles and legacy.get("role") in {"user","assistant","system"}:
        roles=[legacy["role"]]
    sid=current_session_id.get()
    if not sid or sid=="_unknown_":
        return json.dumps(failure("missing_session",invalid=True),ensure_ascii=False)
    await memory_manager._get_db()
    if sender_ids and not author_ids and sources and "messages" not in sources and "generated_media" not in sources:
        subject_ids = subject_ids or sender_ids
        sender_ids = None
    from .agent_runner import current_run
    run = current_run.get()
    excluded = getattr(run, "request_source_ids", []) if run and run.session_id == sid else []
    payload=await HistorySearch(memory_manager._db_path).search(sid,query=query,mode=mode,sources=sources,
        author_ids=author_ids or sender_ids,subject_ids=subject_ids,sender_names=sender_names,since=since,
        until=until,roles=roles,exact_phrase=exact_phrase,match_mode=match_mode,sort=sort,limit=limit,
        cursor=cursor,group_by=group_by, exclude_message_ids=excluded)
    payload["session_id"]=sid
    return json.dumps(payload,ensure_ascii=False,separators=(",",":"))


async def read_history(ids=None,anchor_id=None,before=3,after=3,cursor=None,limit=20):
    from .memory import memory_manager
    from .models import current_session_id
    from .retrieval import HistorySearch,failure
    if not current_session_id.get() or current_session_id.get()=="_unknown_":
        return json.dumps(failure("missing_session",invalid=True),ensure_ascii=False)
    await memory_manager._get_db()
    payload=await HistorySearch(memory_manager._db_path).read(current_session_id.get(),ids=ids,
        anchor_id=anchor_id,before=before,after=after,cursor=cursor,limit=limit)
    return json.dumps(payload,ensure_ascii=False,separators=(",",":"))


async def search_web(query,domains=None,freshness_days=None,limit=8,image_refs=None):
    from .roles import role_available
    if not role_available("search"):
        return json.dumps({"status": "unavailable", "error": {"code": "search_unconfigured"}})
    from .config import agentic_enabled
    if not agentic_enabled():
        if image_refs:
            return json.dumps({"status":"invalid","error":{"code":"image_search_requires_agentic_mode"}})
        return await _legacy_search_web(query)
    from .web_search import search
    return json.dumps(await search(query,domains=domains,freshness_days=freshness_days,limit=limit,image_refs=image_refs),ensure_ascii=False)


async def read_web(source_id,focus="",cursor=None):
    from .web_search import read
    return json.dumps(await read(source_id,focus=focus,cursor=cursor),ensure_ascii=False)





async def query_history(sql="", parameters=None, limit=30):
    import re

    from .config import agentic_enabled
    from .history_sql import query_history_sql
    from .memory import memory_manager
    from .models import current_session_id
    from .retrieval import failure

    if not agentic_enabled():
        return json.dumps(failure("agentic_search_disabled"), ensure_ascii=False)
    sid = current_session_id.get()
    if not sid or sid == "_unknown_":
        return json.dumps(failure("missing_session", invalid=True), ensure_ascii=False)
    await memory_manager._get_db()
    payload = await query_history_sql(memory_manager._db_path, sid, sql, parameters, limit)
    # SQL expressions are arbitrary. Never promote their text to evidence:
    # independently reread only explicit original-message source handles.
    source_columns = [i for i, column in enumerate(payload.get("columns", []))
                      if column == "source_id" or column.endswith("_source_id")]
    refs = list(dict.fromkeys(
        value for row in payload.get("rows", []) for i in source_columns
        if i < len(row) and isinstance(value := row[i], str)
        and re.fullmatch(r"h:\d+:messages:\d+", value)
    ))[:8]
    if refs:
        from .retrieval import HistorySearch
        verified = await HistorySearch(memory_manager._db_path).read(sid, ids=refs, limit=8)
        if verified.get("status") == "ok":
            items = verified.get("items", [])
            for item in items:
                content = item.get("content", "")
                item.update(content=content[:700], excerpt=content[:700],
                            truncated=item.get("truncated", False) or len(content) > 700)
                item["location"] = {"start": 0, "end": len(item["content"])}
            payload["items"] = items
            payload["read_hint"] = (
                "items 是按 source_id 独立回读核验的原消息，不是 SQL 计算文本；仅附前 8 个来源及每条前 700 字。"
                "具体断言以这些原文为依据。需要完整原文、更多来源或相邻对话时再调用 read_history。"
            )
        else:
            payload["warnings"].append("sql_source_verification_failed_read_history_required")
            payload["status"] = "partial"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))




async def inspect_memory(owner_id=None, run_id=None):
    from .agent_runner import current_run
    from .memory_changes import inspect
    run = current_run.get()
    if not run or not run.memory_path:
        return json.dumps({"status": "unavailable"})
    if not owner_id and not run_id:
        run_id = next(iter(run.context_run_ids), None)
        if not run_id:
            owner_id = run.requester_id
    payload = await asyncio.to_thread(inspect, run.memory_path, run.session_id, owner_id=owner_id, run_id=run_id)
    return json.dumps(payload, ensure_ascii=False)


async def revise_memory(action, ref="", content=None, owner_id=None, assertion=None, reason="", operation_id=""):
    from nonebot import get_driver
    from .agent_runner import current_run
    from .memory_changes import change, correction_epoch
    from .memory_index import connect
    run = current_run.get()
    context = _poke_context.get() or {}
    if not run or not run.memory_path or not context.get("allow_sender_alias", False):
        return json.dumps({"status": "unavailable", "reason": "需要单一明确请求者"}, ensure_ascii=False)
    actor = context.get("user_id")
    admin = str(actor) in {str(value) for value in get_driver().config.superusers}
    try:
        payload = await asyncio.to_thread(change, run.memory_path, run.session_id, actor,
            action=action, ref=ref, content=content, owner_id=owner_id, assertion=assertion,
            reason=reason, operation_id=operation_id, confirmation=context.get("current_user_text", ""), admin=admin,
            correction_source_ids=getattr(run, "request_source_ids", ()))
    except ValueError as error:
        return json.dumps({"status": "invalid_arguments", "reason": str(error)}, ensure_ascii=False)
    if payload["status"] in {"applied", "undone"}:
        def epoch():
            with connect(run.memory_path) as db:
                return correction_epoch(db, run.session_id)
        run.memory_correction_epoch = await asyncio.to_thread(epoch)
        for item in run.injected_memories:
            item["invalidated_by"] = payload["operation_id"]
        run.evidence = {k: v for k, v in run.evidence.items() if k.startswith("w:")}
        run.cache.clear()
        run.identity_verdict = None
        payload["note"] = "变更已落库。本轮旧记忆与旧规划不再作为依据；只报告本次变更结果，需要其他事实请重新查询。"
        await run.checkpoint()
    return json.dumps(payload, ensure_ascii=False)




# Bind only after all implementations have been defined.
async def search_knowledge(query="", limit=8, offset=0):
    from .knowledge import KnowledgeStore
    from .config import config
    result = await asyncio.to_thread(KnowledgeStore(config.knowledge_dir).search, query, limit, offset)
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


async def read_knowledge(document_id, offset=0, max_chars=4000):
    from .knowledge import KnowledgeStore
    from .config import config
    result = await asyncio.to_thread(KnowledgeStore(config.knowledge_dir).read, document_id, offset, max_chars)
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


TOOL_FUNCTIONS = bind_handlers(globals())
