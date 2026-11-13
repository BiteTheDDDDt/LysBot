"""Delivery acknowledgements shared by digests and generated media."""
import asyncio

from nonebot import logger

from .conversation import ensure_send_allowed
from .delivery_outbox import DeliveryOutbox
from .execution import ExecutionPolicy, execution_scope
from .config import config
from .qq_addressing import QQAddressing

SEND_POLICY = ExecutionPolicy("delivery.send", permission="send",
                              timeout=lambda: max(1.0, float(config.delivery_timeout)), check_result=False)


async def deliver_once(send, *, session_id=None):
    """One network attempt. Only an explicit rejection permits a safe fallback."""
    try:
        async with execution_scope(SEND_POLICY, session_id=session_id):
            receipt = await send()
    except Exception as error:
        if is_definite_send_failure(error):
            raise
        logger.warning(f"[投递] 回执未确认，不自动重发: {type(error).__name__}: {error}")
        return "unconfirmed", None
    if not isinstance(receipt, dict) or receipt.get("message_id") is None:
        return "unconfirmed", None
    return "sent", receipt


async def deliver_media(bot, *, group_id, user_id, media_type, data, addressing: QQAddressing = QQAddressing()):
    """Send an already generated artifact, independently from its producer."""
    from nonebot.adapters.onebot.v11 import Message, MessageSegment
    from .agent_runner import record_tool_delivery
    from .conversation import session_key
    if media_type not in {"image", "voice"}:
        raise ValueError(f"Unsupported media type: {media_type}")
    segment = MessageSegment.image(data) if media_type == "image" else MessageSegment.record(data)
    message = addressing.prefix() + segment if group_id else Message([segment])
    async def send():
        if group_id:
            return await bot.send_group_msg(group_id=group_id, message=message)
        return await bot.send_private_msg(user_id=user_id, message=message)
    try:
        status, receipt = await deliver_once(send, session_id=session_key(user_id, group_id))
    except asyncio.CancelledError:
        await record_tool_delivery("unconfirmed", None)
        raise
    await record_tool_delivery(status, receipt)
    return status, receipt


def is_ambiguous_image_send_error(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    info = getattr(error, "info", None)
    if not isinstance(info, dict) or info.get("retcode") != 1200:
        return False
    detail = "\n".join(str(info.get(key) or "") for key in ("message", "wording")).lower()
    return "timeout" in detail and ("sendmsg" in detail or "onmsginfolistupdate" in detail)


def is_definite_send_failure(error: Exception) -> bool:
    """Only an explicit OneBot rejection proves that a fallback is safe."""
    info = getattr(error, "info", None)
    if is_ambiguous_image_send_error(error) or not isinstance(info, dict):
        return False
    detail = " ".join(str(info.get(key) or "") for key in ("message", "wording")).lower()
    if any(marker in detail for marker in ("timeout", "timed out", "超时", "connection", "disconnect", "连接")):
        return False
    return info.get("retcode") not in (None, 0)


async def deliver_digest(
    bot, group_id: int, *, outbox: DeliveryOutbox, key: str, payload: dict,
    text: str, image: str | None = None,
) -> str:
    """Share image/text acknowledgement handling and a durable idempotency key."""
    attempts = [("image", image), ("text", text)] if image else [("text", text)]

    async def finish_safely(status, *, result=None, error=None):
        try:
            await outbox.finish(key, status, attempt_token=attempt_token, message_id=(result or {}).get("message_id"),
                                error_type=type(error).__name__ if error else None)
        except Exception as checkpoint_error:
            # The durable sending claim still prevents a second network attempt.
            logger.error(f"[投递] 回执保存失败 key={key}: {checkpoint_error}")

    for medium, message in attempts:
        ensure_send_allowed(group_id)
        attempt_token = await outbox.claim(key, group_id=group_id, medium=medium, payload=payload)
        if attempt_token is None:
            return "already_attempted"
        try:
            ensure_send_allowed(group_id)
        except BaseException:
            # No network call started, so invalidation is a safe retry boundary.
            await outbox.finish(key, "failed", attempt_token=attempt_token, error_type="superseded_before_send")
            raise
        try:
            status, result = await deliver_once(
                lambda: bot.send_group_msg(group_id=group_id, message=message), session_id=f"group_{group_id}")
        except asyncio.CancelledError as error:
            await finish_safely("unknown", error=error)
            raise
        except Exception as error:
            if is_definite_send_failure(error):
                # If this commit fails, stop: retaining sending is safer than retrying.
                await outbox.finish(key, "failed", attempt_token=attempt_token, error_type=type(error).__name__)
                if medium == "image":
                    logger.warning(f"[投递] 图片明确失败，回退文本 group={group_id}: {error}")
                    continue
                raise
            await finish_safely("unknown", error=error)
            logger.warning(f"[投递] 回执未确认，停止自动重发 key={key}: {error}")
            return f"{medium}_unconfirmed"
        if status != "sent":
            await finish_safely("unknown", error=ValueError("No message receipt"))
            return f"{medium}_unconfirmed"
        await finish_safely("sent", result=result)
        return medium
    raise RuntimeError("No digest delivery attempted")
