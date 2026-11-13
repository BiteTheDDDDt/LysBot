"""Reply turn planning and execution helpers."""
import asyncio
import random
import re
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Optional

from nonebot import logger

from .planner import PlannerDecision, ReplyPart
from .reply_policy import ReplyPolicy, SHORT_REPLY, constrain_plan, enforce_reply_budget


TextSender = Callable[[str, bool], Awaitable[None]]


@dataclass
class SentTurn:
    """What was actually sent for one reply turn."""

    sent_texts: list[str]
    sent_meme: Optional[dict]

    @property
    def sent_anything(self) -> bool:
        return bool(self.sent_texts or self.sent_meme)

    @property
    def text_for_memory(self) -> str:
        return "\n".join(self.sent_texts).strip()


def wants_meme(decision: Optional[PlannerDecision]) -> bool:
    return bool(
        decision
        and decision.should_continue
        and decision.meme_part
        and decision.meme_topic
    )


def is_meme_only(decision: Optional[PlannerDecision]) -> bool:
    """A picked reaction needs no factual planning or text generation."""
    return bool(wants_meme(decision) and all(part.type == "meme" for part in decision.parts))


async def prepare_planned_meme(decision: Optional[PlannerDecision]) -> Optional[dict]:
    """Pick a meme before generation so the LLM can know what may be sent."""
    if not wants_meme(decision):
        return None

    try:
        from .tools import pick_meme

        planned = await pick_meme(decision.meme_topic, reason=decision.reason)
        if planned:
            logger.info(
                f"[turn] 预选表情包 topic={decision.meme_topic} "
                f"parts={decision.parts_label} id={planned.get('id')}"
            )
        else:
            logger.info(f"[turn] 表情包预选为空 topic={decision.meme_topic}")
        return planned
    except Exception as e:
        logger.warning(f"[turn] 预选表情包失败: {e}")
        return None


def _first_nonempty_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_two_texts(text: str, *, preserve_paragraphs: bool = False) -> list[str]:
    """Split natural language into at most two short messages."""
    raw = (text or "").strip()
    clean = _first_nonempty_text(raw)
    if not clean:
        return []
    if "```" in raw or "$$" in raw:
        return [raw]  # Never break a fenced code/math block across messages.

    paragraphs = re.split(r"\n\s*\n", raw, maxsplit=1)
    if len(paragraphs) == 2 and all(part.strip() for part in paragraphs):
        return [part.strip() for part in paragraphs]

    def cap_two(parts: list[str]) -> list[str]:
        cleaned = [_first_nonempty_text(p) for p in parts if _first_nonempty_text(p)]
        if len(cleaned) <= 2:
            return cleaned
        return [cleaned[0], ("\n" if preserve_paragraphs else " ").join(cleaned[1:])]

    line_parts = cap_two(re.split(r"[\r\n]+", raw))
    if len(line_parts) >= 2:
        return line_parts

    sentence_parts = []
    start = 0
    for match in re.finditer(r"[。！？!?]+", clean):
        end = match.end()
        part = clean[start:end].strip()
        if part:
            sentence_parts.append(part)
        start = end
    tail = clean[start:].strip()
    if tail:
        sentence_parts.append(tail)
    if len(sentence_parts) >= 2:
        return cap_two(sentence_parts)

    soft_parts = [p.strip() for p in re.split(r"[，,；;]\s*", clean, maxsplit=1) if p.strip()]
    if len(soft_parts) >= 2 and all(len(p) <= 60 for p in soft_parts[:2]):
        return soft_parts[:2]

    return [clean]


def build_turn_items(
    *,
    text: str,
    decision: Optional[PlannerDecision],
    planned_meme: Optional[dict],
    reply_policy: ReplyPolicy = SHORT_REPLY,
) -> list[tuple[str, str | dict]]:
    """Return ordered ('text'|'meme', payload) items, capped at two."""
    text = enforce_reply_budget(text, reply_policy)
    decision = constrain_plan(decision, reply_policy)
    clean_text = text.strip()
    # The content planner can expand a one-text participation plan. Natural
    # paragraph breaks then suffice; no additional output schema is needed.
    if (reply_policy.text_parts is None and reply_policy.detailed and clean_text and not planned_meme
            and (not decision or all(p.type == "text" for p in decision.parts))
            and "```" not in clean_text and "$$" not in clean_text):
        paragraphs = re.split(r'\n\s*\n', clean_text, maxsplit=1)
        if len(paragraphs) == 2 and all(p.strip() for p in paragraphs):
            return [("text", p.strip().rstrip("。")) for p in paragraphs]
    if not decision:
        return [("text", clean_text)] if clean_text else []

    if reply_policy.text_parts and decision.action in {"reply", "close", "poke"} and not decision.meme_part and not decision.required_tool:
        decision = replace(decision, parts=[ReplyPart("text")] * reply_policy.text_parts)
    text_slots = decision.text_part_count
    text_parts = (
        split_two_texts(text, preserve_paragraphs=reply_policy.detailed)[:text_slots]
        if text_slots >= 2
        else ([clean_text] if text_slots == 1 and clean_text else [])
    )
    text_index = 0
    items: list[tuple[str, str | dict]] = []
    for part in decision.parts:
        if part.type == "text":
            if text_index < len(text_parts):
                items.append(("text", text_parts[text_index].rstrip("。")))
                text_index += 1
        elif part.type == "meme" and planned_meme:
            items.append(("meme", planned_meme))
        # image/voice are side effects performed inside the Chat LLM tool loop.

    # A failed media tool produces a textual error even for an image-only or
    # voice-only plan. Never discard that explanation just because no text
    # output was originally requested.
    if not items and clean_text:
        items.append(("text", clean_text))
    return items[:2]


async def send_response_turn(
    *,
    text: str,
    decision: Optional[PlannerDecision],
    planned_meme: Optional[dict],
    send_text: TextSender,
    delay_range: tuple[float, float] = (0.7, 2.0),
    reply_policy: ReplyPolicy = SHORT_REPLY,
) -> SentTurn:
    """Send complete text in the planned order and record delivered content."""
    from .tools import send_picked_meme

    items = build_turn_items(text=text, decision=decision, planned_meme=planned_meme, reply_policy=reply_policy)
    sent_texts: list[str] = []
    sent_meme: Optional[dict] = None

    for idx, (kind, payload) in enumerate(items):
        is_first = not (sent_texts or sent_meme)
        try:
            if idx > 0:
                await asyncio.sleep(random.uniform(*delay_range))

            if kind == "text":
                msg = str(payload or "").strip()
                if not msg:
                    continue
                await send_text(msg, is_first)
                sent_texts.append(msg)
                continue

            if kind == "meme":
                result = await send_picked_meme(payload if isinstance(payload, dict) else None, allow_reply=is_first)
                logger.info(f"[turn] 发送表情包结果: {result[:100]}")
                if getattr(result, "delivery_status", None) == "sent":
                    from .tools import get_last_sent_meme

                    sent_meme = get_last_sent_meme()
        except (Exception, asyncio.CancelledError):
            # Deliberate: a partial send already happened, so stop sending but
            # still return what was delivered for the caller to record. Nothing
            # else runs after this loop, so the cancelled task ends here.
            if not sent_texts and not sent_meme:
                raise
            logger.warning("[turn] 后续发送中断，保留已经发出的回复记录")
            break

    return SentTurn(sent_texts=sent_texts, sent_meme=sent_meme)
