import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import nonebot
nonebot.init()
from src.plugins.llm_chat import agent_runner, commands, models, tools
from test_tool_rounds import FakeToolMessage, make_response, make_tool_call


class JrrpPlanningTests(unittest.IsolatedAsyncioTestCase):
    def test_recent_group_context_preserves_authors_and_readable_source_ids(self):
        history = [models.Message('user', '我准备买车', name='甲', sender_id=101, source_id=1),
                   models.Message('user', '我刚加薪', name='乙', sender_id=202, source_id=2),
                   models.Message('assistant', '上次评语', source_id=3)]
        own = commands._build_jrrp_recent_context(history, group_id=7, sender_name='甲', user_id=101, generation=4)
        surrounding = commands._build_jrrp_recent_context(history, group_id=7, sender_name='甲', user_id=101,
                                                         include_others=True, generation=4)
        self.assertIn('h:4:messages:1', own)
        self.assertNotIn('加薪', own)
        self.assertIn('乙', surrounding)
        self.assertIn('加薪', surrounding)
        self.assertIn('上次评语', surrounding)

    async def test_jrrp_planner_can_query_then_render_and_keep_selected_evidence(self):
        manager = models.ModelManager()
        adapter = models.OpenAICompatibleAdapter('openai:test', 'dummy', 'https://example.test/v1', 'test')
        event = SimpleNamespace(user_id=101, sender=SimpleNamespace(card='', nickname='甲'))
        query = make_tool_call('query', 'query_history', {'sql': 'SELECT content FROM messages_detail WHERE sender_id=101 LIMIT 1'})
        # Keep the chosen source beyond the old 650-character overwrite boundary.
        selected = '理解本轮互动。' * 100 + '选用本人原话“准备买车” [h:0:messages:81]，用于调侃买车预算。'
        plan = json.dumps({'response_mode': 'social', 'verify_group_identity': False, 'content': selected})
        saved = []
        real_run = manager.run_turn
        async def run(*a, **kw):
            result = await real_run(*a, **kw)
            saved.append(result.agent_run)
            return result
        try:
            with patch.object(agent_runner, '_runtime_path', None), \
                 patch.object(commands, '_build_jrrp_memory_context', AsyncMock(return_value='本人近况尚待查找')), \
                 patch.object(commands, '_calculate_2bot_jrrp_value', return_value=49), \
                 patch.object(commands, '_record_command_interaction', AsyncMock()), \
                 patch.object(commands.jrrp_cmd, 'send', AsyncMock(return_value={'message_id': 9})) as send, \
                 patch.object(commands.model_manager, 'run_turn', side_effect=run), \
                 patch.object(manager, '_resolve_candidates', return_value=('openai:test', ['openai:test'])), \
                 patch.object(manager, 'get_or_build_adapter', return_value=adapter), \
                 patch.object(adapter, '_create_completion', AsyncMock(side_effect=[
                     make_response(FakeToolMessage([query])),
                     make_response(FakeToolMessage([], content=plan)),
                     make_response(FakeToolMessage([], content='49分，车还没买，预算先被你开跑了')),
                 ])) as completion, \
                 patch.object(tools, 'execute_tool', AsyncMock(return_value=json.dumps({
                     'status': 'ok', 'items': [{'id': 'h:0:messages:81', 'kind': 'messages',
                     'content': '准备买车', 'sender_id': 101, 'role': 'user'}],
                 }, ensure_ascii=False))) as query_tool:
                await commands.handle_jrrp(event)
            self.assertEqual(completion.await_count, 3)
            query_tool.assert_awaited_once()
            self.assertEqual(query_tool.await_args.args[0], 'query_history')
            self.assertIn('准备买车', saved[0].planning_result)
            self.assertIn('h:0:messages:81', saved[0].planning_result)
            self.assertIn('与聊天内容无因果关系', saved[0].planning_result)
            self.assertIn('49分', send.await_args.args[0])
        finally:
            await adapter.client.close()
