import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
nonebot.init()

from src.plugins.llm_chat import chat, models
from src.plugins.llm_chat.agent_runner import current_run
from src.plugins.llm_chat.planner import PlannerDecision, ReplyPart
from src.plugins.llm_chat.reply_policy import DETAILED_REPLY, SHORT_REPLY
from src.plugins.llm_chat.turns import build_turn_items, send_response_turn
from src.plugins.llm_chat.utils import clean_response, normalize_qq_text


class ReplyPresentationTests(unittest.IsolatedAsyncioTestCase):
    def test_prose_decoration_preserves_code_and_math(self):
        text = '# 标题\n**重点**\n* 说明\n`a ** b`\n```python\na ** b\n```\n$x ** y$'
        self.assertEqual(normalize_qq_text(text),
                         '标题\n重点\n• 说明\n`a ** b`\n```python\na ** b\n```\n$x ** y$')

    def test_cleanup_preserves_answer_content_in_both_modes(self):
        examples = [
            '1. 输入3美元\n2. 输出15美元\n来源：https://example.com/pricing',
            '价格分别是3美元、15美元。结论：输入输出分别计费',
            '理由：输出需要逐步生成。\n分析：不能只比较输入价格。',
            '[官方定价](https://example.com/pricing)，输入3美元',
            '示例：`vector<int> values;`，要求 a < b 且 b > c',
            '```python\nif a < b:\n    print("<think>literal</think>")\n```',
            '$a < b$，以及 $$a < b > c$$',
            '标签示例：<think>这里是文档中的字面内容</think>',
            '<think>没有闭合的示例',
        ]
        for detailed in (False, True):
            for original in examples:
                with self.subTest(detailed=detailed, original=original):
                    self.assertEqual(clean_response(original, preserve_paragraphs=detailed), original)

    def test_cleanup_only_removes_explicit_leading_wrappers(self):
        self.assertEqual(clean_response('<think>内部内容</think>\n[助手]：完整答复'), '完整答复')
        self.assertEqual(clean_response('assistant: 完整答复'), '完整答复')
        self.assertEqual(clean_response('磷叶石 很喜欢这个角色'), '磷叶石 很喜欢这个角色')

    async def test_single_and_merged_handlers_preserve_numbered_answers_and_code(self):
        event = SimpleNamespace(get_plaintext=lambda: "查价格并给个例子")
        original = (
            '[价格](https://example.com/pricing)\n'
            '1. 输入3美元\n2. 输出15美元\n'
            '结论：分别计费\n示例：`vector<int> values;`，a < b 且 b > c'
        )
        for batch in (False, True):
            for detailed in (False, True):
                with self.subTest(batch=batch, detailed=detailed):
                    decision = PlannerDecision(reply_mode='detailed' if detailed else 'sentence')
                    bot, saved = SimpleNamespace(send=AsyncMock()), AsyncMock()
                    with (
                        patch.object(chat.memory_manager, 'get_history', AsyncMock(return_value=[])),
                        patch.object(chat.memory_manager, 'get_session_model', return_value='test-model'),
                        patch.object(chat.memory_manager, 'add_message', saved),
                        patch.object(chat, 'execute_planned_side_effect', AsyncMock()),
                        patch.object(chat, 'prepare_planned_meme', AsyncMock(return_value=None)),
                        patch.object(chat, 'build_context_messages', AsyncMock(return_value=[])),
                        patch.object(chat.model_manager, 'run_turn', AsyncMock(return_value=original)),
                        patch.object(chat, '_finalize_immediate_tool_turn', AsyncMock(return_value=False)),
                        patch.object(chat.proactive_manager, 'mark_just_spoke'),
                        patch('src.plugins.llm_chat.followups.answered'),
                        patch('src.plugins.llm_chat.model_identity.update_group_card_for_response', AsyncMock()),
                    ):
                        if batch:
                            pending = chat.PendingMessage(
                                user_id=42, sender_name='甲', message='查价格并给个例子',
                                event=event, planner_decision=decision,
                            )
                            await chat._do_handle_merged_chat(bot, 99, [pending], primary_user_id=42)
                        else:
                            await chat._do_handle_single_chat(
                                bot, event, '查价格并给个例子', [], [], 42, planner_decision=decision,
                            )
                    bot.send.assert_awaited_once_with(event, original)
                    self.assertEqual(saved.await_args.args[2], original)

    async def test_expanded_one_slot_sends_two_complete_paragraphs(self):
        text = clean_response('核心结论。\n\n第一条依据。\n第二条依据。', preserve_paragraphs=True)
        send = AsyncMock()
        result = await send_response_turn(text=text,
            decision=PlannerDecision(parts=[ReplyPart('text')]), planned_meme=None,
            reply_policy=DETAILED_REPLY, send_text=send, delay_range=(0, 0))
        self.assertEqual(result.sent_texts, ['核心结论', '第一条依据。\n第二条依据'])
        self.assertEqual([c.args[1] for c in send.await_args_list], [True, False])
        self.assertEqual(result.text_for_memory, '核心结论\n第一条依据。\n第二条依据')

    def test_code_block_is_not_split_into_multiple_messages(self):
        items = build_turn_items(text='示例\n\n```python\nx = 1\n\nprint(x)\n```',
            decision=PlannerDecision(parts=[ReplyPart('text')]), planned_meme=None,
            reply_policy=DETAILED_REPLY)
        self.assertEqual(len(items), 1)

    async def test_complete_pricing_answer_is_sent_without_compaction(self):
        original = (
            'GCP 官方常规价，每百万 token、美元，输入/输出：\n'
            'Haiku 4.5：1/5\nSonnet 4.6：3/15\nOpus 4.6：5/25\n'
            '批量、缓存、长上下文和区域会影响最终价格。\n'
            'https://cloud.google.com/vertex-ai/generative-ai/pricing'
        )
        for policy in (SHORT_REPLY, DETAILED_REPLY):
            manager = models.ModelManager()
            async def call(messages, **kwargs):
                if kwargs['role'] == 'planner':
                    current_run.get().reply_mode = 'sentence'
                    return models.ChatResponse('列出价格、计费条件和来源')
                self.assertTrue(kwargs['caller'].endswith(':render'))
                self.assertFalse(current_run.get().reply_policy.detailed)
                return models.ChatResponse(original)
            with patch.object(manager, '_chat_impl', side_effect=call) as calls:
                result = await manager.run_turn([models.Message('user', '查一下模型定价')], reply_policy=policy)
            self.assertEqual(calls.call_count, 2)
            send = AsyncMock()
            delivered = await send_response_turn(text=clean_response(str(result), preserve_paragraphs=policy.detailed),
                decision=PlannerDecision(parts=[ReplyPart('text')]), planned_meme=None,
                reply_policy=policy, send_text=send, delay_range=(0, 0))
            self.assertEqual(delivered.text_for_memory, original)
            send.assert_awaited_once_with(original, True)
