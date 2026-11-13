import time
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
import aiosqlite
import nonebot

nonebot.init()

from src.plugins.llm_chat import idle_memes as idle
from src.plugins.llm_chat.planner import PlannerDecision, ReplyPart


class IdleMemeTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_idle_send_is_persistent_and_not_repeated_without_humans(self):
        from src.plugins.llm_chat import tools
        db = await aiosqlite.connect(':memory:')
        await db.execute('CREATE TABLE memes(id INTEGER PRIMARY KEY, topic TEXT)')
        await db.execute("INSERT INTO memes VALUES(1,'funny')")
        await db.commit()
        old = datetime.fromtimestamp(time.time() - 7400).isoformat()

        async def sent(*args, **kwargs):
            ctx = tools._meme_context.get()
            ctx['last_sent'] = {'id': 1, 'description': '独立笑话', 'topic': 'funny', 'message_id': 2}
            return '已发送'
        try:
            with patch.object(idle, 'datetime', SimpleNamespace(now=lambda: SimpleNamespace(hour=12), fromisoformat=datetime.fromisoformat)), patch.object(idle, "plan_chat_action", AsyncMock(return_value=PlannerDecision(parts=[ReplyPart("meme")]))), patch.object(idle, 'activity', AsyncMock(return_value=(old, old))), patch.object(idle.meme_manager, 'is_on_cooldown', return_value=False), patch.object(idle.meme_manager, '_get_db', AsyncMock(return_value=db)), patch.object(idle.meme_manager, 'pick_random', AsyncMock(return_value=(1, 'path.jpg', '独立笑话'))) as pick, patch.object(tools, 'send_picked_meme', AsyncMock(side_effect=sent)) as send, patch.object(idle.memory_manager, 'add_message', AsyncMock()) as remember, patch.object(idle.proactive_manager, 'mark_just_spoke'):
                self.assertTrue(await idle.share_idle_meme(AsyncMock(), 98765))
                self.assertFalse(await idle.share_idle_meme(AsyncMock(), 98765))
            pick.assert_awaited_once_with(None, meme_kind='joke')
            self.assertFalse(send.await_args.kwargs['allow_reply'])
            remember.assert_awaited_once()
            self.assertEqual((await (await db.execute('SELECT human_at FROM idle_meme_state')).fetchone())[0], old)
        finally:
            await db.close()

    def test_quiet_interval_requires_no_recent_bot_or_human_activity(self):
        now = time.time()
        old = datetime.fromtimestamp(now - 7300).isoformat()
        recent = datetime.fromtimestamp(now - 100).isoformat()
        self.assertTrue(idle.quiet_enough(old, old, None, 0, now))
        self.assertFalse(idle.quiet_enough(recent, old, None, 0, now))
        self.assertFalse(idle.quiet_enough(old, old, old, 0, now))
        self.assertFalse(idle.quiet_enough(old, old, None, now - 4000, now))
        self.assertFalse(idle.quiet_enough(None, None, None, 0, now))

    async def test_disabled_feature_does_not_fetch_groups(self):
        with patch.object(idle.config, 'idle_meme_enabled', False), \
             patch.object(idle, 'get_bots') as bots:
            await idle.check_idle_groups()
        bots.assert_not_called()
