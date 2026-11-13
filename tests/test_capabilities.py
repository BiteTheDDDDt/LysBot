import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import capabilities, conversation, tools


class CapabilityTests(unittest.TestCase):
    def test_every_declared_tool_has_one_schema_and_execution_owner(self):
        names = [item["function"]["name"] for item in tools.AVAILABLE_TOOLS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), set(capabilities.CAPABILITIES))
        self.assertEqual(
            set(tools.TOOL_FUNCTIONS),
            {name for name, cap in capabilities.CAPABILITIES.items() if not cap.runner_managed},
        )
        self.assertTrue(capabilities.MEDIA_TOOLS.isdisjoint(capabilities.READ_TOOLS))

    def test_missing_implementation_fails_at_binding(self):
        with self.assertRaisesRegex(ValueError, "Missing capability handler"):
            capabilities.bind_handlers({})


class TaskOwnershipTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        conversation.start()

    async def test_nested_and_child_activities_keep_ownership_on_cancel(self):
        entered = asyncio.Event()

        async def child():
            with conversation.activity_scope("search_web", kind="capability", run_id="run-test"):
                entered.set()
                await asyncio.Event().wait()

        with conversation.session_scope("group_987654"):
            parent = conversation.task_snapshot()["activities"][-1]
            task = asyncio.create_task(child())
            try:
                await asyncio.wait_for(entered.wait(), 2)
                rows = conversation.task_snapshot()["activities"]
                owned = next(row for row in rows if row["run_id"] == "run-test")
                self.assertEqual(owned["parent_id"], parent["id"])
                self.assertEqual(owned["session_id"], "group_987654")
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self.assertIn(parent["id"], {r["id"] for r in conversation.task_snapshot()["activities"]})
                self.assertNotIn(owned["id"], {r["id"] for r in conversation.task_snapshot()["activities"]})
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(conversation.task_snapshot()["activity_count"], 0)

    async def test_tracking_does_not_refresh_an_invalidated_session(self):
        with conversation.session_scope("group_987655"):
            conversation.invalidate("group_987655")
            with conversation.activity_scope("generate_image", kind="capability"):
                with self.assertRaises(conversation.TurnSuperseded):
                    conversation.ensure_send_allowed()
        self.assertEqual(conversation.task_snapshot()["activity_count"], 0)

    async def test_shutdown_waits_for_owned_cleanup(self):
        entered, cleaned = asyncio.Event(), asyncio.Event()

        @conversation.tracked_job
        async def news():
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        task = asyncio.create_task(news())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await conversation.close()
            self.assertTrue(cleaned.is_set())
            self.assertTrue(task.cancelled())
            self.assertEqual(conversation.task_snapshot()["task_count"], 0)
            self.assertEqual(conversation.task_snapshot()["activity_count"], 0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            conversation.start()

    async def test_group_facts_use_actual_group_and_preserve_displayed_model_suffix(self):
        from src.plugins.llm_chat.model_types import current_session_id

        bot = SimpleNamespace(
            self_id=999,
            get_group_member_info=AsyncMock(return_value={
                "user_id": 999, "card": "磷叶石 [GPT-6.1 Sol]", "nickname": "磷叶石", "role": "member",
            }),
            get_group_member_list=AsyncMock(return_value=[
                {"user_id": 2, "nickname": "管理员", "role": "admin"},
                {"user_id": 3, "nickname": "普通成员", "role": "member"},
                {"user_id": 4, "card": "群主名片", "nickname": "群主", "role": "owner"},
            ]),
        )
        token = current_session_id.set("group_7")
        try:
            tools.set_poke_context(7, 42, bot)
            own = json.loads(await tools.execute_tool("get_group_facts", {"scope": "bot"}))
            self.assertEqual(own["items"][0]["display_name"], "磷叶石 [GPT-6.1 Sol]")
            self.assertTrue(own["items"][0]["is_bot"])
            bot.get_group_member_info.assert_awaited_once_with(group_id=7, user_id=999, no_cache=True)
            roles = json.loads(await tools.execute_tool("get_group_facts", {"scope": "leadership"}))
            self.assertEqual([item["role"] for item in roles["items"]], ["owner", "admin"])
            bot.get_group_member_list.assert_awaited_once_with(group_id=7, no_cache=True)
            bad = json.loads(await tools.execute_tool("get_group_facts", {"scope": "bot", "group_id": 8}))
            self.assertEqual(bad["status"], "invalid_arguments")
            tools.set_poke_context(8, 42, bot)
            bad = json.loads(await tools.execute_tool("get_group_facts", {"scope": "bot"}))
            self.assertEqual(bad["status"], "unavailable")
            self.assertNotIn("get_group_facts", {t["function"]["name"] for t in tools.get_chat_tools()})
            self.assertEqual(bot.get_group_member_info.await_count, 1)
        finally:
            tools.clear_poke_context()
            current_session_id.reset(token)

    async def test_tool_error_does_not_leak_activity_or_arguments(self):
        async def failing(query):
            snapshot = conversation.task_snapshot()
            self.assertNotIn("private-input", str(snapshot))
            self.assertEqual(snapshot["activities"][-1]["name"], "search_web")
            raise RuntimeError("upstream failed")

        with patch.dict(tools.TOOL_FUNCTIONS, search_web=failing):
            result = await tools.execute_tool("search_web", {"query": "private-input"})
        self.assertIn("unavailable", result)
        self.assertEqual(conversation.task_snapshot()["activity_count"], 0)
