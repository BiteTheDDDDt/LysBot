"""Lightweight action planner for group chat turns."""
import json
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from nonebot import logger

from .config import config
from .qq_addressing import MessageTarget, QQAddressing, select_addressing
from .memory import memory_manager
from .memes import ALLOWED_TOPICS
from .models import Message as LLMMessage, model_manager
from .reply_policy import REPLY_MODE_GUIDANCE


# Triggers supply candidates. Only the chosen action becomes executable.
CHAT_ACTIONS = ("reply", "close", "silent", "wait")
NON_REPLY_ACTIONS = {"silent", "wait"}
TOPIC_STATES = {"active", "closing", "exhausted"}
REPLY_PART_TYPES = {"text", "meme", "image", "voice"}
MEDIA_PART_TYPES = {"image", "voice"}
MEDIA_PART_TO_TOOL = {
    "image": "generate_image",
    "voice": "send_voice",
}
TOOL_TO_MEDIA_PART = {
    tool_name: part_type
    for part_type, tool_name in MEDIA_PART_TO_TOOL.items()
}
LEGACY_RESPONSE_PARTS = {
    "text_only": ["text"],
    "two_texts": ["text", "text"],
    "meme_only": ["meme"],
    "meme_then_text": ["meme", "text"],
    "text_then_meme": ["text", "meme"],
    "media_only": [],
    "media_then_text": ["text"],
}


@dataclass(frozen=True)
class ReplyPart:
    """One user-visible output in a planned reply, in delivery order."""

    type: str
    topic: str = ""

    def label(self) -> str:
        if self.type == "meme" and self.topic:
            return f"meme:{self.topic}"
        return self.type


@dataclass
class PlannerDecision:
    """Planner output normalized for the rest of the chat pipeline."""

    action: str = "reply"
    reason: str = ""
    reply_hint: str = ""
    poke_target_id: str = ""
    wait_seconds: int = 0
    topic_state: str = "active"
    reply_mode: str = "sentence"
    parts: list[ReplyPart] = field(
        default_factory=lambda: [ReplyPart("text")]
    )
    addressing: QQAddressing = field(default_factory=QQAddressing)

    def __post_init__(self) -> None:
        if self.action in NON_REPLY_ACTIONS | {"repeat"}:
            self.parts = []

    @property
    def should_continue(self) -> bool:
        return self.action not in NON_REPLY_ACTIONS

    @property
    def required_tool(self) -> str:
        """Derive the required media tool from the visible output plan."""
        for part in self.parts:
            tool_name = MEDIA_PART_TO_TOOL.get(part.type)
            if tool_name:
                return tool_name
        return ""

    @property
    def media_followup(self) -> bool:
        """Whether a text part appears after the generated media."""
        media_index = next(
            (index for index, part in enumerate(self.parts) if part.type in MEDIA_PART_TYPES),
            -1,
        )
        return bool(
            media_index >= 0
            and any(part.type == "text" for part in self.parts[media_index + 1:])
        )

    @property
    def meme_part(self) -> Optional[ReplyPart]:
        return next((part for part in self.parts if part.type == "meme"), None)

    @property
    def meme_topic(self) -> str:
        part = self.meme_part
        return part.topic if part else ""

    @property
    def text_part_count(self) -> int:
        return sum(1 for part in self.parts if part.type == "text")

    @property
    def parts_label(self) -> str:
        return " -> ".join(part.label() for part in self.parts) or "none"

    @property
    def wait_delay_seconds(self) -> int:
        """Return the configured, bounded delay before the next Planner pass."""
        default_seconds = max(1, int(config.planner_wait_default_seconds))
        max_seconds = max(1, int(config.planner_wait_max_seconds))
        return max(1, min(int(self.wait_seconds or default_seconds), max_seconds))

    def system_hint(self, *, include_reasoning: bool = True) -> str:
        """Return a compact instruction block for the final chat model."""
        hints = [
            f"规划动作: {self.action}",
            f"用户可见输出顺序: {self.parts_label}",
        ]
        if include_reasoning and self.reason:
            hints.append(f"原因: {self.reason}")
        if include_reasoning and self.reply_hint:
            hints.append(f"回复侧重点: {self.reply_hint}")
        if self.addressing.reply_to is not None:
            hints.append(f"首条消息会引用 QQ 消息 {self.addressing.reply_to}，正文自然接话。")
        if self.addressing.mention is not None:
            hints.append(f"首条消息会 @ QQ {self.addressing.mention}，正文无需再写 @。")
        if self.action == "close":
            hints.append(
                "本轮是话题收束：只说一句自然、简短的结束语，不提出新问题，"
                "不继续抬杠，不留下必须让对方接话的钩子。"
            )
        if self.required_tool:
            hints.append(
                f"本轮必须通过标准 tool_calls 协议调用 {self.required_tool}；"
                "不要只用文字承诺稍后执行。"
            )
            hints.append(
                "根据实际回执判断媒体是否已完整回应，再决定要不要补一句。"
            )
        elif self.meme_part:
            hints.append(
                f"系统会按上述顺序发送 {self.meme_topic} 表情包；"
                "你只负责其中的 text 部分，不要输出 JSON，也不要解释表情包机制。"
            )
            if self.text_part_count == 0:
                hints.append("本轮没有 text 部分，无需再补文字。")
        if self.text_part_count >= 2:
            hints.append(
                "本轮包含两个 text 部分；你仍只输出自然语言。"
                "请写成两句短话，最好用换行分隔，系统会按两条消息连续发送。"
            )
        if self.action == "poke":
            target = self.poke_target_id or "sender"
            hints.append(f"已优先尝试戳一戳目标: {target}；文字回复不要再重复承诺未执行的动作。")
        return "【本轮动作规划-内部参考】\n" + "\n".join(hints)


def _extract_json_object(text: str) -> Optional[dict]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _coerce_str(data: dict, key: str, default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def _coerce_int(data: dict, key: str, default: int = 0) -> int:
    try:
        return int(data.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _one_line(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _is_current_user_message(content: str, current_message: str) -> bool:
    """Detect the recorder's formatted copy of the current group message."""
    current = _one_line(current_message)
    recorded = _one_line(content)
    if not current or not recorded:
        return False
    if recorded == current:
        return True
    # Group memory usually prefixes the sender, for example
    # ``【群员-甲】 当前消息``. Only inspect the newest user item.
    return recorded.endswith(current) or f"用户消息: {current}" in recorded


def _format_planner_history(
    messages: list[LLMMessage],
    *,
    current_message: str,
    max_chars: int,
    hide_image_observations: bool = False,
) -> str:
    """Format a bounded recent conversation block for the action planner."""
    visible = [
        message
        for message in messages
        if message.role in {"user", "assistant"}
        and not getattr(message, "is_thinking", False)
    ]
    if (
        visible
        and visible[-1].role == "user"
        and _is_current_user_message(visible[-1].content, current_message)
    ):
        visible = visible[:-1]

    budget = max(200, int(max_chars or 6000))
    newest_first: list[str] = []
    used = 0
    for message in reversed(visible):
        from .image_observations import unsolicited_view
        content = _one_line(unsolicited_view(message.content) if hide_image_observations else message.content)
        if not content:
            continue
        # One oversized message should not crowd out the entire dialogue.
        if len(content) > 1200:
            content = content[:1200] + "…"
        if message.role == "assistant":
            line = f"Bot: {content}"
        elif getattr(message, "name", None) and not content.startswith("【群员-"):
            line = f"{message.name}: {content}"
        else:
            line = content

        cost = len(line) + 1
        if newest_first and used + cost > budget:
            break
        if cost > budget:
            line = line[-budget:]
            cost = len(line)
        newest_first.append(line)
        used += cost

    return "\n".join(reversed(newest_first))


def _strip_current_from_fallback(context: str, current_message: str) -> str:
    lines = [line for line in str(context or "").splitlines() if line.strip()]
    if lines and _is_current_user_message(lines[-1], current_message):
        lines.pop()
    return "\n".join(lines).strip()


async def _resolve_planner_context(
    *,
    session_id: str,
    current_message: str,
    fallback_context: str,
    hide_image_observations: bool = False,
) -> str:
    """Prefer persisted/in-memory conversation history over caller caches."""
    try:
        limit = max(1, min(int(config.planner_context_messages), 100))
        recent_messages = await memory_manager.get_recent_session_messages(
            session_id,
            limit=limit,
        )
        history_context = _format_planner_history(
            recent_messages,
            current_message=current_message,
            max_chars=int(config.planner_context_max_chars),
            hide_image_observations=hide_image_observations,
        )
        if history_context:
            return history_context
    except Exception as exc:
        logger.warning(
            f"[planner] 读取最近会话失败，使用入口缓存 session={session_id}: {exc}"
        )
    from .image_observations import unsolicited_view
    fallback = unsolicited_view(fallback_context) if hide_image_observations else fallback_context
    return _strip_current_from_fallback(fallback, current_message)


def _fallback_parts(action: str) -> list[ReplyPart]:
    if action in NON_REPLY_ACTIONS | {"repeat"}:
        return []
    return [ReplyPart("text")]


def _legacy_parts(
    data: dict,
    *,
    legacy_meme_action: bool,
    default_topic: str,
    mode: str,
) -> list[ReplyPart]:
    """Convert the previous response_shape/required_tool output into parts."""
    required_tool = _coerce_str(data, "required_tool").lower()
    media_type = TOOL_TO_MEDIA_PART.get(required_tool)
    legacy_shape = _coerce_str(data, "response_shape").lower()
    if media_type:
        parts = [ReplyPart(media_type)]
        if legacy_shape == "media_then_text":
            parts.append(ReplyPart("text"))
        return parts

    part_types = LEGACY_RESPONSE_PARTS.get(legacy_shape)
    if part_types is None and legacy_meme_action:
        if mode == "proactive":
            part_types = random.choices(
                [["meme"], ["meme", "text"], ["text", "meme"]],
                weights=[35, 45, 20],
                k=1,
            )[0]
        else:
            part_types = random.choices(
                [["meme"], ["meme", "text"], ["text", "meme"]],
                weights=[20, 55, 25],
                k=1,
            )[0]
    if not part_types:
        return []
    return [
        ReplyPart(part_type, default_topic if part_type == "meme" else "")
        for part_type in part_types
    ]


def _parse_reply_parts(
    data: dict,
    *,
    action: str,
    legacy_meme_action: bool,
    mode: str,
) -> list[ReplyPart]:
    """Normalize the Planner's ordered visible outputs to a safe executable plan."""
    top_level_topic = _coerce_str(data, "meme_topic").lower()
    if top_level_topic not in ALLOWED_TOPICS:
        top_level_topic = "funny" if legacy_meme_action else ""

    parsed: list[ReplyPart] = []
    raw_parts = data.get("parts")
    if isinstance(raw_parts, list):
        for raw_part in raw_parts:
            if isinstance(raw_part, str):
                part_type = raw_part.strip().lower()
                topic = ""
            elif isinstance(raw_part, dict):
                part_type = str(raw_part.get("type") or "").strip().lower()
                topic = str(raw_part.get("topic") or "").strip().lower()
            else:
                continue
            if part_type not in REPLY_PART_TYPES:
                continue
            if part_type == "meme":
                if topic not in ALLOWED_TOPICS:
                    topic = top_level_topic or "funny"
            else:
                topic = ""
            parsed.append(ReplyPart(part_type, topic))
    else:
        parsed = _legacy_parts(
            data,
            legacy_meme_action=legacy_meme_action,
            default_topic=top_level_topic or "funny",
            mode=mode,
        )

    media_index = next(
        (index for index, part in enumerate(parsed) if part.type in MEDIA_PART_TYPES),
        -1,
    )
    if media_index >= 0:
        # Generated media is sent during the Chat LLM tool loop. The only
        # currently meaningful visible continuation is one text part after it.
        media_part = parsed[media_index]
        has_followup = any(
            part.type == "text"
            for part in parsed[media_index + 1:]
        )
        return [
            media_part,
            *([ReplyPart("text")] if has_followup else []),
        ]

    if action in NON_REPLY_ACTIONS:
        return []
    if action == "close":
        return [ReplyPart("text")]
    if action == "poke" and data.get("parts") == []:
        return []

    normalized: list[ReplyPart] = []
    meme_seen = False
    text_count = 0
    for part in parsed:
        if part.type == "meme":
            if meme_seen:
                continue
            meme_seen = True
        elif part.type == "text":
            if text_count >= 2:
                continue
            text_count += 1
        else:
            continue
        normalized.append(part)
        if len(normalized) >= 2:
            break

    return normalized or _fallback_parts(action)


def _normalize_decision(data: Optional[dict], default_action: str, mode: str = "passive") -> PlannerDecision:
    if not isinstance(data, dict):
        return PlannerDecision(
            action=default_action,
            parts=_fallback_parts(default_action),
        )

    raw_action = _coerce_str(data, "action", default_action).lower()
    legacy_meme_action = raw_action == "meme"
    action = "reply" if legacy_meme_action else raw_action
    if action not in (*CHAT_ACTIONS, "poke", "repeat"):
        action = default_action

    wait_seconds = max(
        0,
        min(
            _coerce_int(data, "wait_seconds", 0),
            max(1, int(config.planner_wait_max_seconds)),
        ),
    )
    topic_state = _coerce_str(data, "topic_state", "active").lower()
    if topic_state not in TOPIC_STATES:
        topic_state = "active"
    parts = [] if action == "repeat" else _parse_reply_parts(
        data,
        action=action,
        legacy_meme_action=legacy_meme_action,
        mode=mode,
    )
    has_media = any(part.type in MEDIA_PART_TYPES for part in parts)
    if has_media:
        action = "reply"
        topic_state = "active"
    elif action == "close":
        topic_state = "closing"
    elif topic_state == "exhausted" and action == "reply":
        # If the model recognizes an exhausted topic but asks for another
        # ordinary reply, end the loop. A fresh media request reopens it above.
        action = "silent"
        parts = []
    return PlannerDecision(
        action=action,
        reason=_coerce_str(data, "reason"),
        reply_hint=_coerce_str(data, "reply_hint"),
        poke_target_id=_coerce_str(data, "poke_target_id"),
        wait_seconds=wait_seconds,
        topic_state=topic_state,
        parts=parts,
        reply_mode=data.get("reply_mode") if data.get("reply_mode") in {"tiny", "sentence", "detailed"} else "sentence",
    )


async def plan_chat_action(
    *,
    session_id: str,
    group_id: int,
    current_message: str,
    recent_context: str = "",
    mode: str = "passive",
    is_direct: bool = True,
    has_images: bool = False,
    faces: Optional[list[str]] = None,
    default_action: str = "reply",
    followup_context: str = "",
    trigger_kind: str = "message",
    bot_self_id: str = "",
    current_targets: tuple[MessageTarget, ...] = (),
    repeat_candidate: str = "",
) -> PlannerDecision:
    """Choose an interaction from the actions available for this trigger."""
    idle = trigger_kind == "idle"
    proactive = mode == "proactive"
    actions = CHAT_ACTIONS + ("poke",) if group_id else CHAT_ACTIONS
    if idle:
        actions = ("silent", "reply")
    if repeat_candidate and proactive and not idle and not is_direct:
        actions += ("repeat",)
    if proactive or idle:
        default_action = "silent"

    recent_context = await _resolve_planner_context(
        session_id=session_id,
        current_message=current_message,
        fallback_context=recent_context,
        hide_image_observations=mode == "proactive",
    )
    context_lines = sum(1 for line in recent_context.splitlines() if line.strip())
    logger.info(
        f"[planner] input mode={mode} session={session_id} "
        f"context_lines={context_lines} context_chars={len(recent_context)} "
        f"current_chars={len(current_message or '')}"
    )

    from .image_observations import UNSOLICITED_IMAGE_POLICY, unsolicited_view
    from .proactive_style import PROACTIVE_VALUE_POLICY
    from .capabilities import VOICE_INTENT_POLICY
    from .roles import role_available

    allowed_media = {
        "voice": not idle and role_available("tts"),
        "image": not idle and role_available("image_gen"),
    }
    if idle:
        policy = """这是长时间无人发言后的定时机会。看最近互动决定保持安静，还是分享一张独立网络梗图。
如果最近在认真讨论、有人不欢迎 bot 或此时插话不合适，选择 silent。轻松氛围下可以 reply。
reply 只允许一张 meme，不加字，不回复旧消息，不 @。topic 可省略，表示从全部网络梗图等概率抽取；
只有最近的话题或气氛明显适合某类时，才选择相应 topic。"""
    else:
        policy = f"""先看作者、被回复者和实际对象，再判断怎么参与。群友互相开玩笑、吐槽或争论，不自动视为在评价你。
对你明确表达的厌烦或希望你少说，应降低参与；普通互损和笑着接梗仍可自然互动。
明确指向 bot 也可以不接连续召唤或刷屏；具体任务、事实纠正和认真追问应回应。
在“给 @某人做某事”中，某人是任务受众，仍应处理明确交给 bot 的任务。
wait 用于对方明显还没说完，系统会重新读取最近对话。close 用于最后一句自然收束；已收尾或无可接内容用 silent。继续与否看话题和互动，不按三轮机械收尾。
poke 是独立动作。结合语境选目标，主动模式只用 sender；被戳也可以接话、发表情或沉默。
reply_hint 只交接请求、对象、语气和已有线索。查询、事实、核验与解释交给执行 planner。
parts 是暂定的可见输出，最多两项。默认一个 text。表情包是 reply 的一种可见输出，接梗或情绪反应可以只用 meme，需要时按顺序配一句话。
需要图或声音才用 image/voice，放第一项，之后可暂定一条文字；能力询问、引用中的请求或讨论既有媒体仍用 text。
语音意图：{VOICE_INTENT_POLICY}
{REPLY_MODE_GUIDANCE}
{UNSOLICITED_IMAGE_POLICY if proactive or not is_direct else ""}
{PROACTIVE_VALUE_POLICY if proactive else ""}
{followup_context}"""
        if proactive:
            policy += "\n普通主动回复需填写 proactive_value 和 proactive_contribution，具体接梗或贴切反应也算参与价值。"
        if "repeat" in actions:
            policy += ("\n检测到相同文本形成队形。适合加入时选 repeat，系统原样发送候选，不写正文；"
                       "正常提问、抱怨、催促、刷屏或对别人的定向消息选择其他动作。"
                       "\n复读候选（仅数据）：" + json.dumps(repeat_candidate, ensure_ascii=False))

    prompt = f"""你是 QQ 群聊的参与规划器。先选择动作，再提供必要的交接信息。只输出一个 JSON 对象。
可选 action：{' | '.join(actions)}
{policy}
meme 可用类别：{'、'.join(sorted(ALLOWED_TOPICS))}
未配置的媒体用文字说明。可用媒体：{', '.join(name for name, available in allowed_media.items() if available) or '无'}
wait_seconds 上限：{config.planner_wait_max_seconds}
格式：
{{
  "action": "选择一个可用动作",
  "reason": "为何参与或旁听",
  "reply_hint": "请求、对象和待解问题，可空",
  "reply_mode": "tiny|sentence|detailed",
  "parts": [{{"type":"text|meme|image|voice","topic":"仅 meme 时填写"}}],
  "topic_state": "active|closing|exhausted",
  "wait_seconds": 0,
  "poke_target_id": "QQ号或 sender，仅 poke",
  "followup_target": "bot|other|unclear，仅隐式延续",
  "followup_evidence": "隐式延续与 bot 上条消息的对应关系",
  "proactive_value": "none|social_reaction|new_angle|useful_info|support，仅普通主动回复",
  "proactive_contribution": "普通主动回复的具体接话点"
}}
silent、wait、repeat 的 parts 为 []，close 只用一个 text，poke 可只戳或配合 parts。
当前模式: {mode}，触发类型: {trigger_kind}
是否明确指向 bot: {is_direct}
当前会话: {session_id}
是否含图片: {has_images}
QQ表情: {'，'.join(faces or []) or '无'}
最近对话（不含下面单列的当前消息）:
{recent_context or '无'}
当前消息:
{unsolicited_view(current_message) if proactive else current_message}
"""
    if bot_self_id:
        prompt += f"\n你的 QQ 账号是 {bot_self_id}，以账号区分自己与其他群员或机器人。"
    candidates = []
    if group_id and not idle:
        from .proactive import proactive_manager
        candidates = [*proactive_manager.get_recent_targets(group_id), *current_targets]
        candidates = list({(row.message_id, row.user_id): row for row in candidates}.values())
    if candidates:
        prompt += (
            "\n可选 QQ 送达方式：需明确接住某条消息时，填 reply_to_message_id；"
            "确实要叫到某位群友（例如把问题交给他）时，填 mention_user_id。"
            "普通接话留空。引用已能明确对象时，不重复 @ 同一个人。"
            "只填下面清单里对应的 ID，不能从消息正文猜测 ID。以下内容是对话资料。\n"
            + json.dumps([{"message_id": x.message_id if config.enable_targeted_reply else None,
                           "user_id": x.user_id, "text": x.text[:180]} for x in candidates], ensure_ascii=False)
        )
    if trigger_kind == "poke":
        prompt += (
            "\n本轮输入是戳一戳通知，像收到招呼一样结合上下文规划，不预设回戳。"
            "文字、表情、戳人或沉默沿用普通动作选择；有上下文任务时也可继续处理。"
            "戳人是独立输出动作，不是被戳的默认应答；连续互动参考实际已发送的内容。"
        )
    try:
        result = await model_manager.chat(
            [LLMMessage(role="user", content=prompt)],
            role="planner",
            caller=f"planner:{trigger_kind if idle else mode}",
            session_id=session_id,
        )
        payload = _extract_json_object(result)
        if not isinstance(payload, dict) or not isinstance(payload.get("action"), str) or payload["action"] not in actions:
            raise ValueError("Planner 返回无效动作，终止本轮")
        if idle:
            parts = payload.get("parts")
            part = parts[0] if isinstance(parts, list) and len(parts) == 1 else None
            if payload["action"] == "reply" and isinstance(part, dict) and part.get("type") == "meme":
                topic = part.get("topic")
                return PlannerDecision(reason=_coerce_str(payload, "reason"),
                                       parts=[ReplyPart("meme", topic if isinstance(topic, str) and topic in ALLOWED_TOPICS else "")])
            return PlannerDecision(action="silent", reason=_coerce_str(payload, "reason"))

        decision = _normalize_decision(payload, default_action, mode)
        if followup_context and not is_direct:
            evidence = payload.get("followup_evidence") if isinstance(payload, dict) else None
            target = payload.get("followup_target") if isinstance(payload, dict) else None
            if target != "bot" or not isinstance(evidence, str) or not evidence.strip():
                decision = PlannerDecision(action="silent", reason="未确认当前消息在回应 bot 的具体发言", parts=[])
        if proactive and decision.should_continue and decision.action != "repeat":
            contribution = _coerce_str(payload, "proactive_contribution")
            if payload.get("proactive_value") not in {"social_reaction", "new_angle", "useful_info", "support"} or not contribution:
                decision = PlannerDecision(action="silent", reason="主动发言缺少具体新贡献，保持旁听", parts=[])
            else:
                decision.reply_hint = f"{decision.reply_hint}\n本轮新增内容：{contribution}".strip()
        unavailable = [part.type for part in decision.parts if part.type in allowed_media and not allowed_media[part.type]]
        if unavailable:
            decision.parts = [ReplyPart("text")]
            decision.reply_hint += "\n本轮所请求的媒体能力未配置，请直接说明，不承诺发送。"
        if decision.parts and decision.should_continue:
            decision.addressing = select_addressing(
                payload, candidates, bot_self_id=bot_self_id, allow_reply=config.enable_targeted_reply,
            )
        logger.info(
            f"[planner] mode={mode} group={group_id} action={decision.action} "
            f"topic={decision.topic_state} parts={decision.parts_label} "
            f"reply_mode={decision.reply_mode} "
            f"required_tool={decision.required_tool or '-'} "
            f"qq_reply={decision.addressing.reply_to is not None} qq_at={decision.addressing.mention is not None} "
            f"reason={decision.reason[:80]}"
        )
        return decision
    except Exception as e:
        logger.warning(f"[planner] 决策失败，终止本轮: {type(e).__name__}: {e}")
        raise


async def execute_planned_side_effect(decision: Optional[PlannerDecision]) -> str:
    """Execute safe side effects selected by planner before final generation."""
    if not decision:
        return ""

    if decision.action == "poke":
        target_id = decision.poke_target_id or "sender"
        try:
            from .tools import send_poke

            result = await send_poke(target_id, reason=decision.reason)
            logger.info(f"[planner] 预执行戳一戳 target={target_id}: {result[:80]}")
            return result
        except Exception as e:
            logger.warning(f"[planner] 预执行戳一戳失败: {e}")
            return f"send_poke failed: {e}"

    return ""
