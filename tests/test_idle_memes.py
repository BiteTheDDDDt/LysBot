import time
import unittest
from contextlib import ExitStack
from datetime import datetime
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
import aiosqlite
import nonebot

nonebot.init()

from src.plugins.llm_chat import idle_memes as idle, tools
from src.plugins.llm_chat.planner import PlannerDecision, ReplyPart


class IdleMemeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = await aiosqlite.connect(':memory:')
        self.addAsyncCleanup(self.db.close)
        await self.db.execute('CREATE TABLE memes(id INTEGER PRIMARY KEY, topic TEXT)')
        await self.db.execute("INSERT INTO memes VALUES(1,'funny')")
        await self.db.commit()
        self.bot = AsyncMock()
        self.old = datetime.fromtimestamp(time.time() - 7400).isoformat()
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(patch.object(idle, 'datetime', SimpleNamespace(
            now=lambda: SimpleNamespace(hour=12), fromisoformat=datetime.fromisoformat)))
        patches.enter_context(patch.object(idle, 'activity', AsyncMock(return_value=(self.old, self.old))))
        patches.enter_context(patch.object(idle.meme_manager, 'is_on_cooldown', return_value=False))
        patches.enter_context(patch.object(idle.meme_manager, '_get_db', AsyncMock(return_value=self.db)))
        self.decision = PlannerDecision(parts=[ReplyPart("meme")])
        self.plan = patches.enter_context(patch.object(idle, 'plan_chat_action', AsyncMock(return_value=self.decision)))
        self.meme = (1, 'path.jpg', '独立笑话')
        self.pick = patches.enter_context(patch.object(idle.meme_manager, 'pick_random', AsyncMock(return_value=self.meme)))
        self.send = patches.enter_context(patch.object(tools, 'send_picked_meme', AsyncMock(side_effect=self.mark_sent)))
        self.remember = patches.enter_context(patch.object(idle.memory_manager, 'add_message', AsyncMock()))
        patches.enter_context(patch.object(idle.proactive_manager, 'mark_just_spoke'))

    async def mark_sent(self, *args, **kwargs):
        tools._meme_context.get()['last_sent'] = {
            'id': 1, 'description': '独立笑话', 'topic': 'funny', 'message_id': 2,
        }
        return tools.ToolDeliveryResult('已发送', 'sent')

    async def test_successful_idle_send_is_persistent_and_not_repeated_without_humans(self):
        self.assertTrue(await idle.share_idle_meme(self.bot, 98765))
        self.assertFalse(await idle.share_idle_meme(self.bot, 98765))
        self.pick.assert_awaited_once_with(None, meme_kind='joke')
        self.assertFalse(self.send.await_args.kwargs['allow_reply'])
        self.remember.assert_awaited_once()
        self.assertEqual((await (await self.db.execute('SELECT human_at FROM idle_meme_state')).fetchone())[0], self.old)

    async def test_planner_or_selection_failure_leaves_quiet_group_eligible(self):
        self.plan.side_effect = [RuntimeError('temporary'), self.decision, self.decision]
        self.pick.side_effect = [None, self.meme]
        with self.assertRaisesRegex(RuntimeError, 'temporary'):
            await idle.share_idle_meme(self.bot, 98765)
        self.assertFalse(await idle.share_idle_meme(self.bot, 98765))
        self.assertTrue(await idle.share_idle_meme(self.bot, 98765))
        self.send.assert_awaited_once()

    async def test_only_definite_delivery_failure_allows_another_attempt(self):
        self.send.side_effect = [
            tools.ToolDeliveryResult('被拒绝', 'failed'),
            tools.ToolDeliveryResult('回执未知', 'unconfirmed'),
        ]
        for _ in range(3):
            self.assertFalse(await idle.share_idle_meme(self.bot, 98765))
        self.assertEqual(self.send.await_count, 2)
        self.remember.assert_not_awaited()

    async def test_intentional_silence_handles_episode_without_delivery_cooldown(self):
        self.plan.return_value = PlannerDecision(action='silent')
        self.assertFalse(await idle.share_idle_meme(self.bot, 98765))
        self.assertFalse(await idle.share_idle_meme(self.bot, 98765))
        self.plan.assert_awaited_once()
        self.send.assert_not_awaited()
        row = await (await self.db.execute('SELECT human_at,attempted_at FROM idle_meme_state')).fetchone()
        self.assertEqual(row, (self.old, 0))

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
