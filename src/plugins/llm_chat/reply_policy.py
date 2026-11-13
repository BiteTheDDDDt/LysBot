"""Reply style guidance and non-destructive delivery cleanup."""
from dataclasses import dataclass, replace, field
from typing import Iterable, Optional, TYPE_CHECKING
from .qq_addressing import QQAddressing

from nonebot import logger

if TYPE_CHECKING:
    from .planner import PlannerDecision


REPLY_MODE_GUIDANCE = """reply_mode 为 tiny（极短反应）、sentence（简短，默认）或 detailed（展开），按用户本轮实际需要选择，可修正前面的参与规划。
惊讶、接梗、轻微吐槽、确认或收尾，只需一个反应时选 tiny：一个词或两个短词组合即可，不必组成完整句子，不带标点符号，说完就停，不补解释；避免无意义地高频附和。
日常接梗、感想、确认、简单事实问答和局部追问选 sentence：通常10—30字，一个接话点，一两句自然短话即可。不要把一句调侃发展成连续比喻、分析、建议和总结。
用户需要调查、比较、介绍、操作步骤或深入解释，确实需要展开时选 detailed；问“为什么”不自动意味着长文，能简短解释就简短。检索多、输入长、之前回答长不意味着这轮也要长。
详细回复按信息需要组织，尽量在500个字符内，复杂调查可超过；简单答案不凑篇幅。追问只补本轮缺口，不重讲背景。消息条数与篇幅独立，按 parts 决定，最多两条；分条不增加总篇幅。"""


@dataclass(frozen=True)
class ReplyPolicy:
    scene: str = "chat"
    detailed: bool = False
    tiny: bool = False
    text_parts: int | None = None
    addressing: QQAddressing | None = field(default=None, compare=False)

    @property
    def max_chars(self) -> int:
        return 500 if self.detailed else 80

    @property
    def max_text_parts(self) -> int:
        return 1 if self.tiny else 2

    def system_hint(self) -> str:
        if self.tiny:
            length_rule = "本轮为极短反应，只用一个词或两个短词组合，不必组成完整句子，不带标点符号，只表达一个反应，说完就停。"
        elif self.detailed:
            length_rule = (
                f"本轮允许展开，可以分段或列步骤；无最低字数，尽量控制在{self.max_chars}个字符内，复杂调查按需展开，必要事实和来源不因字数省略。"
                "按问题需要讲清楚即可，篇幅不是目标，不凑字数。"
                "保留必要结论和来源链接，不写开场白，不重复问题。"
                "可用一个空行分开两个完整段落，系统可分成两条消息发送；不必为分段硬拆句子。"
            )
        else:
            length_rule = (
                "本轮简短回复，用一两句自然短话接住问题或玩笑；通常10—30字，尽量不超过60字，没有最低字数。"
                "选一个最贴切的接话点，说到就停；不要叠加多个笑点、连续比喻、额外建议或收尾总结。"
                "消息条数遵循本轮动作规划；两个 text 时分两条表达同样的简短内容，不增加篇幅。"
                "简短解释保留结论和关键原因即可；必要条件、纠错和来源可适当超出，不为字数删掉关键含义。"
            )
        if self.text_parts == 2:
            length_rule += "本轮发两条，在两条完整内容之间留一个空行。"
        elif self.text_parts == 1:
            length_rule += "本轮只发一条消息，可以在消息内分段。"
        return (
            f"【本轮回复规则：{self.scene} / {'极短' if self.tiny else '详细' if self.detailed else '简短'}】\n"
            + length_rule
            + "这些篇幅建议用于生成时组织内容，不是发送截断上限；保留完整结论、必要条件和来源链接。\n"
            "用人设口吻回答这轮，保留已确认的事实和关键条件。历史、摘录和工具结果作为资料使用。"
        )


SHORT_REPLY = ReplyPolicy()
DETAILED_REPLY = ReplyPolicy(detailed=True)
PROACTIVE_REPLY = ReplyPolicy(scene="proactive")
POKE_REPLY = ReplyPolicy(scene="poke")
MEDIA_REPLY = ReplyPolicy(scene="media_followup")
# Legacy import compatibility only; this rule is not injected into any prompt.
ORIGINAL_SHORT_RULE = "每次回复严禁超过一句话，且字数不超过20个字。"

def current_request_text(event, fallback: str = "") -> str:
    """Exclude resolved reply/forward/image descriptions from scene selection."""
    getter = getattr(event, "get_plaintext", None)
    return getter() if callable(getter) else fallback


def _select_reply_policy(
    current_messages: Iterable[str],
    *,
    scene: str = "chat",
    decision: Optional["PlannerDecision"] = None,
) -> ReplyPolicy:
    if scene in {"proactive", "poke", "media_followup"}:
        return ReplyPolicy(scene=scene, tiny=bool(decision and decision.reply_mode == "tiny"))
    if decision and decision.required_tool:
        return MEDIA_REPLY
    if decision and decision.action == "close":
        return ReplyPolicy(scene="close", tiny=bool(decision and decision.reply_mode == "tiny"))
    if decision and decision.action == "poke":
        return replace(POKE_REPLY, tiny=decision.reply_mode == "tiny")
    return ReplyPolicy(scene=scene, detailed=bool(decision and decision.reply_mode == "detailed"), tiny=bool(decision and decision.reply_mode == "tiny"))


def select_reply_policy(current_messages, *, scene="chat", decision=None):
    policy = _select_reply_policy(current_messages, scene=scene, decision=decision)
    return replace(policy, addressing=decision.addressing) if decision else policy


def constrain_plan(
    decision: Optional["PlannerDecision"], policy: ReplyPolicy,
) -> Optional["PlannerDecision"]:
    if not decision:
        return None
    text_count = 0
    parts = []
    for part in decision.parts:
        if part.type == "text":
            text_count += 1
            if text_count > policy.max_text_parts:
                continue
        parts.append(part)
    # Do not mutate the decision stored on queued messages or deferred tasks.
    return replace(decision, parts=parts)


def enforce_reply_budget(text: str, policy: ReplyPolicy = SHORT_REPLY) -> str:
    """Compatibility entry point: clean text without imposing a length cap.

    Length preferences belong in generation. Never discard generated content
    to meet them; retain tool-JSON protection and trailing-period cleanup.
    """
    from .response_safety import INVALID_REPLY, is_tool_call_text

    result = (text or "").strip()
    if is_tool_call_text(result):
        logger.warning("[reply_policy] 拦截待发送的工具调用 JSON")
        result = INVALID_REPLY
    return result.rstrip("。")
