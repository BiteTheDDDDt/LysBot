import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import nonebot
nonebot.init()

from src.plugins.llm_chat import chat, planner


class PlannerFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_never_fabricates_text_or_media_decision(self):
        for request in ['你好', '生成一张图片', '根据群友印象生成一张素人自拍九宫格照片']:
            with patch.object(planner, '_resolve_planner_context', AsyncMock(return_value='')), \
                 patch.object(planner.model_manager, 'chat', AsyncMock(side_effect=asyncio.TimeoutError)):
                with self.assertRaises(asyncio.TimeoutError):
                    await planner.plan_chat_action(session_id='group_77', group_id=77, current_message=request)

    async def test_invalid_planner_output_aborts_instead_of_using_default_action(self):
        for result in ['', '已生成九宫格', '{}', '{"action":"unknown"}']:
            with patch.object(planner, '_resolve_planner_context', AsyncMock(return_value='')), \
                 patch.object(planner.model_manager, 'chat', AsyncMock(return_value=result)):
                with self.assertRaises(ValueError):
                    await planner.plan_chat_action(session_id='group_77', group_id=77, current_message='生成图片')

    async def test_merged_planner_failure_notifies_only_an_explicit_request(self):
        bot = SimpleNamespace(send=AsyncMock(), send_group_msg=AsyncMock())
        event = SimpleNamespace(get_plaintext=lambda: '根据群友印象生成一张自拍九宫格照片')
        pending = chat.PendingMessage(user_id=123, sender_name='甲', message=event.get_plaintext(), event=event)
        for direct, error in [(True, asyncio.TimeoutError), (False, ValueError)]:
            bot.send.reset_mock()
            with patch.object(chat, '_group_message_direction', return_value=(direct, False)), \
                 patch.object(chat, '_eligible_event_followup', return_value=True), \
                 patch.object(chat, 'plan_chat_action', AsyncMock(side_effect=error)), \
                 patch.object(chat.model_manager, 'run_turn', AsyncMock()) as content:
                await chat.handle_merged_chat(bot, 77, [pending])
            content.assert_not_awaited()
            if direct:
                bot.send.assert_awaited_once_with(event, chat.REQUEST_FAILURE)
            else:
                bot.send.assert_not_awaited()
        bot.send_group_msg.assert_not_awaited()

    async def test_private_planner_failure_notifies_once_before_execution(self):
        bot = SimpleNamespace(send=AsyncMock())
        event = SimpleNamespace(user_id=123)
        prepared = SimpleNamespace(message="生成图片", images_base64=[], faces=[], image_slots=[])
        with patch.object(chat, 'prepare_incoming_message', AsyncMock(return_value=prepared)), \
             patch.object(chat, 'plan_chat_action', AsyncMock(side_effect=ValueError)), \
             patch.object(chat, 'handle_single_chat', AsyncMock()) as execute:
            await chat._process_chat(bot, event)
        execute.assert_not_awaited()
        bot.send.assert_awaited_once_with(event, chat.REQUEST_FAILURE)
