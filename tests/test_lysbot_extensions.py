import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from lysbot import plugins
from src.plugins.llm_chat import capabilities, tools
from src.plugins.llm_chat.agent_runner import AgentRunner, current_run


class ExtensionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.name = "test_extension_lookup"

    async def asyncTearDown(self):
        capabilities.CAPABILITIES.pop(self.name, None)
        capabilities.READ_TOOLS.discard(self.name)
        tools.TOOL_FUNCTIONS.pop(self.name, None)
        plugins._master_tools.pop(self.name, None)
        for collection in (capabilities.BASE_CHAT_TOOLS, capabilities.AVAILABLE_TOOLS):
            collection[:] = [tool for tool in collection if tool["function"]["name"] != self.name]

    async def test_master_tool_is_hidden_and_rejected_for_other_users(self):
        handler = AsyncMock(return_value="done")
        plugins.register_tool(self.name, "Test", {"type": "object", "properties": {}}, handler,
                              master_only=True)
        run = AgentRunner("group_7")
        run.requester_id = "99"
        token = current_run.set(run)
        try:
            with patch.object(plugins, "is_master", side_effect=lambda user: user == "42"):
                self.assertNotIn(self.name, {tool["function"]["name"] for tool in tools.get_chat_tools()})
                with self.assertRaises(PermissionError):
                    await tools.TOOL_FUNCTIONS[self.name]()
                handler.assert_not_awaited()
                run.requester_id = "42"
                self.assertIn(self.name, {tool["function"]["name"] for tool in tools.get_chat_tools()})
                self.assertEqual(await tools.TOOL_FUNCTIONS[self.name](), "done")
                context = handler.call_args.args[0]
                self.assertEqual(context.requester_id, "42")
                self.assertEqual(context.session_id, "group_7")
        finally:
            current_run.reset(token)

    async def test_tools_cannot_replace_builtin_or_run_without_context(self):
        handler = AsyncMock(return_value="done")
        with self.assertRaises(ValueError):
            plugins.register_tool("send_poke", "Test", {"type": "object"}, handler)
        plugins.register_tool(self.name, "Test", {"type": "object"}, handler, read_only=True)
        self.assertIn(self.name, capabilities.READ_TOOLS)
        token = current_run.set(None)
        try:
            with self.assertRaises(PermissionError):
                await tools.TOOL_FUNCTIONS[self.name]()
            handler.assert_not_awaited()
        finally:
            current_run.reset(token)

    async def test_notification_uses_only_configured_owner(self):
        bot = SimpleNamespace(send_private_msg=AsyncMock(return_value={"message_id": 123}))
        with patch.object(plugins, "get_driver", return_value=SimpleNamespace(
            config=SimpleNamespace(superusers={"42"})
        )):
            status, receipt = await plugins.notify_master(bot, "任务完成")
        self.assertEqual(status, "sent")
        self.assertEqual(receipt["message_id"], 123)
        self.assertEqual(bot.send_private_msg.call_args.kwargs["user_id"], 42)
