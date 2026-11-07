import asyncio
import unittest
from unittest.mock import patch

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
