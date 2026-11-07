"""Optional example plugin: deliver local task results privately to MASTER_QQ."""
import os
from pathlib import Path

from nonebot import get_bots, logger, require
from lysbot import notify_master

from .storage import drain

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler


@scheduler.scheduled_job("interval", seconds=10, id="queued_notifications", max_instances=1,
                         coalesce=True, misfire_grace_time=30)
async def deliver_pending():
    bots = get_bots()
    if len(bots) != 1:
        return  # Wait for one unambiguous QQ connection before claiming jobs.
    bot = next(iter(bots.values()))
    root = Path(os.environ.get("LLM_CHAT_DATA_DIR", "data")) / "queued_notifications"

    async def send(text):
        return await notify_master(bot, text)

    try:
        counts = await drain(root, send)
        if counts:
            logger.info("Queued notifications processed: {}", counts)
    except Exception as error:
        logger.warning("Queued notifications failed: {}", type(error).__name__)
