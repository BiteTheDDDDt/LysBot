"""Share one standalone joke after a quiet group interval, without an LLM call."""
import time
from datetime import datetime

from nonebot import get_bots, logger, require

from .config import config
from .conversation import serialized_turn, tracked_job
from .memory import memory_manager
from .memes import meme_manager
from .proactive import proactive_manager

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler


def quiet_enough(latest, human, previous_human, last_attempt, now):
    if not latest or not human or human == previous_human:
        return False
    return (now - datetime.fromisoformat(latest).timestamp() >= config.idle_meme_threshold
            and now - last_attempt >= config.idle_meme_cooldown)


async def activity(group_id):
    db = await memory_manager._get_db()
    async with db.execute(
        "SELECT MAX(timestamp),MAX(CASE WHEN role='user' THEN timestamp END) "
        "FROM messages_detail WHERE session_id=? AND is_thinking=0",
        (f"group_{group_id}",),
    ) as cur:
        return await cur.fetchone()


@serialized_turn(lambda bot, group_id: f"group_{group_id}", background=True)
async def share_idle_meme(bot, group_id):
    from .tools import (_meme_context, set_meme_context, send_picked_meme,
                        get_last_sent_meme, format_sent_meme_for_memory)

    if not config.enable_proactive or not config.idle_meme_enabled:
        return False
    if meme_manager.is_on_cooldown(group_id):
        return False
    # Use local time consistently with message timestamps. Avoid waking quiet groups overnight.
    if not 9 <= datetime.now().hour < 23:
        return False
    db = await meme_manager._get_db()
    await db.execute("CREATE TABLE IF NOT EXISTS idle_meme_state (group_id INTEGER PRIMARY KEY, human_at TEXT, attempted_at REAL)")
    async with db.execute("SELECT human_at,attempted_at FROM idle_meme_state WHERE group_id=?", (group_id,)) as cur:
        previous = await cur.fetchone()
    latest, human = await activity(group_id)
    if not quiet_enough(latest, human, previous[0] if previous else None, previous[1] if previous else 0, time.time()):
        return False
    picked = await meme_manager.pick_random(None, meme_kind="joke")
    if not picked:
        return False
    meme_id, path, description = picked
    # New human or scheduled bot messages arriving during selection supersede the idle trigger.
    if tuple(await activity(group_id)) != (latest, human):
        return False
    async with db.execute("SELECT topic FROM memes WHERE id=?", (meme_id,)) as cur:
        row = await cur.fetchone()
    if not row:
        return False
    # Persist the attempt before sending: a restart must not replay an uncertain delivery.
    await db.execute("INSERT OR REPLACE INTO idle_meme_state VALUES(?,?,?)", (group_id, human, time.time()))
    await db.commit()
    token = _meme_context.set(None)
    try:
        set_meme_context(group_id, 0, bot)
        result = await send_picked_meme({"id": meme_id, "file_path": path, "topic": row[0],
                                      "description": description, "reason": "群内连续两小时无消息，分享一张独立网络梗图"}, allow_reply=False)
        sent = get_last_sent_meme()
        if not sent:
            logger.info(f"[idle_meme] group={group_id} send not confirmed: {result[:100]}")
            return False
        await memory_manager.add_message(0, "assistant", format_sent_meme_for_memory(sent), group_id)
        proactive_manager.mark_just_spoke(group_id)
        return True
    finally:
        _meme_context.reset(token)


@scheduler.scheduled_job("interval", minutes=5, id="idle_meme_share", coalesce=True, max_instances=1, misfire_grace_time=60)
@tracked_job
async def check_idle_groups():
    if not config.enable_proactive or not config.idle_meme_enabled:
        return
    bots = list(get_bots().values())
    if not bots:
        return
    bot = bots[0]
    try:
        groups = await bot.get_group_list()
    except Exception as error:
        logger.warning(f"[idle_meme] group list unavailable: {type(error).__name__}")
        return
    for group in groups:
        try:
            await share_idle_meme(bot, int(group["group_id"]))
        except Exception as error:
            logger.warning(f"[idle_meme] group={group.get('group_id')} failed: {type(error).__name__}")
