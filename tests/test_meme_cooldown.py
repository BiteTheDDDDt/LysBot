import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import nonebot
from nonebot.adapters.onebot.v11.exception import ActionFailed
nonebot.init()
from src.plugins.llm_chat import tools
from src.plugins.llm_chat.memes import meme_manager


class MemeCooldownTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        tools.clear_meme_context()

    async def test_directed_reply_can_pick_during_proactive_cooldown(self):
        tools.set_meme_context(77, 1, bot=object(), directed=True)
        with patch.object(meme_manager, 'is_on_cooldown', return_value=True), \
             patch.object(meme_manager, 'pick_random', AsyncMock(return_value=(7, 'file.png', '哭哭'))):
            picked = await tools.pick_meme('sad')
        self.assertEqual(picked['id'], 7)
        tools._meme_context.get()['sent_in_round'] = True
        self.assertIsNone(await tools.pick_meme('sad'))
        self.assertIn('already_sent', tools.format_meme_selection_failure())

    async def test_proactive_cooldown_is_preserved_and_explained_accurately(self):
        tools.set_meme_context(77, 1, bot=object())
        with patch.object(meme_manager, 'is_on_cooldown', return_value=True), \
             patch.object(meme_manager, 'pick_random', AsyncMock()) as pick:
            self.assertIsNone(await tools.pick_meme('sad'))
        pick.assert_not_awaited()
        hint = tools.format_meme_selection_failure()
        self.assertIn('status=cooldown', hint)
        self.assertIn('不是没有合适表情', hint)

    async def test_missing_candidate_and_provider_error_have_distinct_status(self):
        tools.set_meme_context(77, 1, bot=object(), directed=True)
        with patch.object(meme_manager, 'pick_random', AsyncMock(return_value=None)):
            self.assertIsNone(await tools.pick_meme('sad'))
        self.assertIn('status=no_candidate', tools.format_meme_selection_failure())
        with patch.object(meme_manager, 'pick_random', AsyncMock(side_effect=RuntimeError('failed'))):
            with self.assertRaises(RuntimeError):
                await tools.pick_meme('sad')
        self.assertIn('status=selection_error', tools.format_meme_selection_failure())

    async def test_meme_delivery_distinguishes_rejection_from_unknown_receipt(self):
        bot = SimpleNamespace(send_group_msg=AsyncMock(side_effect=[
            ActionFailed(retcode=100, message='rejected'),
            TimeoutError(),
        ]))
        tools.set_meme_context(77, 1, bot=bot, directed=True)
        with TemporaryDirectory() as folder, patch.object(meme_manager, 'mark_sent') as mark:
            image = Path(folder) / 'test.png'
            image.write_bytes(b'example meme')
            meme = {'id': 1, 'file_path': str(image), 'topic': 'funny'}
            self.assertEqual((await tools.send_picked_meme(meme)).delivery_status, 'failed')
            self.assertEqual((await tools.send_picked_meme(meme)).delivery_status, 'unconfirmed')
            self.assertEqual((await tools.send_picked_meme(meme)).delivery_status, 'unconfirmed')
        self.assertEqual(bot.send_group_msg.await_count, 2)
        self.assertIsNone(tools.get_last_sent_meme())
        mark.assert_not_called()
