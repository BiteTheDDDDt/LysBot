import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
from nonebot.adapters.onebot.v11 import Message

nonebot.init()

from src.plugins.llm_chat import agent_runner, commands, tools
from src.plugins.llm_chat.model_types import Message as LLMMessage, current_session_id
from src.plugins.llm_chat.reply_policy import ReplyPolicy


class FactCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidates_are_limited_to_the_active_session_and_visible_messages(self):
        recent = AsyncMock(return_value=[
            LLMMessage("user", "某个待查说法", sender_id=101, source_id=67),
            LLMMessage("assistant", "内部想法", is_thinking=True),
            LLMMessage("user", "参考资料", is_reference=True),
        ])
        with patch.object(commands.memory_manager, "get_recent_session_messages", recent):
            token = current_session_id.set("group_7")
            try:
                result = json.loads(await tools.execute_tool("start_fact_check", {}))
                self.assertEqual(result["status"], "ok")
                self.assertEqual([row["source_id"] for row in result["recent_messages"]], [67])
                result = json.loads(await tools.execute_tool("start_fact_check", {"session_id": "group_8"}))
                self.assertEqual(result["status"], "invalid_arguments")
            finally:
                current_session_id.reset(token)
        recent.assert_awaited_once_with("group_7", limit=18)

    def test_planner_can_select_fact_check_and_command_limits_other_tools(self):
        self.assertIn("start_fact_check", {x["function"]["name"] for x in tools.get_chat_tools()})
        token = agent_runner.current_run.set(SimpleNamespace(
            reply_policy=ReplyPolicy(scene="fact_check"), progress_callback=None, sources={}))
        try:
            names = {x["function"]["name"] for x in tools.get_chat_tools()}
            self.assertIn("start_fact_check", names)
            self.assertIn("recall_history", names)
            self.assertTrue(names <= tools.READ_TOOLS)
        finally:
            agent_runner.current_run.reset(token)

    async def test_command_passes_topic_and_quote_to_normal_planner_and_records_answer(self):
        event = SimpleNamespace(user_id=101, sender=SimpleNamespace(card="", nickname="测试"),
                                reply=SimpleNamespace(message=Message("被引用的说法")))
        with (patch.object(commands.model_manager, "run_turn", AsyncMock(return_value="查证结论")) as run,
              patch.object(commands.dczx_cmd, "send", AsyncMock(return_value={"message_id": 1})) as send,
              patch.object(commands, "_record_command_interaction", AsyncMock()) as record):
            await commands.handle_dczx(SimpleNamespace(), event, Message("指定主题"))
        self.assertEqual(run.await_args.kwargs["session_id"], "user_101")
        self.assertEqual(run.await_args.kwargs["reply_policy"].scene, "fact_check")
        messages = run.await_args.args[0]
        self.assertIn("指定主题", messages[-1].content)
        self.assertTrue(any(row.is_reference and "被引用的说法" in row.content for row in messages))
        send.assert_awaited_once_with("查证结论")
        self.assertEqual(record.await_args.kwargs["assistant_text"], "查证结论")
