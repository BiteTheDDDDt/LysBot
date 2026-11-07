import json
import unittest
from unittest.mock import AsyncMock, patch

import nonebot
nonebot.init()
from src.plugins.llm_chat import planner
from src.plugins.llm_chat.capabilities import VOICE_INTENT_POLICY, VOICE_TOOL


class VoiceIntentTests(unittest.IsolatedAsyncioTestCase):
    async def plan(self, message, payload, direct=True):
        model = AsyncMock(return_value=json.dumps(payload))
        with patch.object(planner, '_resolve_planner_context', AsyncMock(return_value='h1k1: 坐飞机不敢查jrrp')), \
             patch.object(planner.config, 'enable_sentiment_analysis', False), \
             patch.object(planner.model_manager, 'chat', model):
            decision = await planner.plan_chat_action(session_id='group_77', group_id=77,
                                                      current_message=message, is_direct=direct)
        return decision, model.await_args.args[0][0].content

    async def test_voice_decision_for_third_party_survives_normalization(self):
        decision, prompt = await self.plan('为@某人(QQ:42) 播放 see you again', {
            'action': 'reply', 'parts': [{'type': 'voice'}],
            'reply_hint': '为对方用原创送别台词接梗，不播放原曲',
        })
        self.assertEqual(decision.required_tool, 'send_voice')
        self.assertEqual([part.type for part in decision.parts], ['voice'])
        self.assertIn('送别', decision.reply_hint)
        self.assertIn('是否明确指向 bot: True', prompt)
        self.assertIn('某人是任务受众', prompt)
        self.assertIn(VOICE_INTENT_POLICY, prompt)
        self.assertIn(VOICE_INTENT_POLICY, VOICE_TOOL['function']['description'])

    async def test_play_keyword_does_not_override_semantic_decision(self):
        for message, action, direct in [
            ('播放 See You Again 原曲，不要念台词', 'reply', True),
            ('播放这个视频', 'reply', True),
            ('这个播放器为什么没声音', 'reply', True),
            ('我给他播放 see you again 了', 'silent', False),
        ]:
            with self.subTest(message=message):
                decision, _ = await self.plan(message, {
                    'action': action, 'parts': [{'type': 'text'}] if action == 'reply' else [],
                }, direct)
                self.assertEqual(decision.action, action)
                self.assertFalse(decision.required_tool)
