import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import nonebot
nonebot.init()

from src.plugins.llm_chat import conversation
from src.plugins.llm_chat.execution import ExecutionPolicy, execution_scope
from src.plugins.llm_chat.delivery import deliver_once


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        conversation.start()

    async def test_invalidated_queued_work_never_enters_handler(self):
        gate = asyncio.Semaphore(0)
        entered = False
        async def job():
            nonlocal entered
            async with execution_scope(ExecutionPolicy("queued"), session_id="group_998801", slots=gate):
                entered = True
        task = asyncio.create_task(job())
        await asyncio.sleep(0)
        self.assertIn("queued", {r["phase"] for r in conversation.task_snapshot()["activities"]})
        conversation.invalidate("group_998801")
        gate.release()
        with self.assertRaises(conversation.TurnSuperseded):
            await task
        self.assertFalse(entered)
        self.assertEqual(gate._value, 1)
        self.assertEqual(conversation.task_snapshot()["activity_count"], 0)

    async def test_timeout_releases_slot_and_owned_client(self):
        client = type("Client", (), {"aclose": AsyncMock()})()
        gate = asyncio.Semaphore(1)
        with self.assertRaises(TimeoutError):
            async with execution_scope(ExecutionPolicy("timed", timeout=lambda: 0.01), slots=gate) as resources:
                resources.own_client(client)
                await asyncio.Event().wait()
        client.aclose.assert_awaited_once()
        self.assertEqual(gate._value, 1)
        self.assertEqual(conversation.task_snapshot()["activity_count"], 0)

    async def test_cancelled_thread_resource_is_collected_and_removed(self):
        ready = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "late.png"
            def factory():
                loop.call_soon_threadsafe(ready.set)
                release.wait(2)
                path.write_bytes(b"test")
                return path
            async def job():
                async with execution_scope(ExecutionPolicy("card")) as resources:
                    await resources.create(factory, lambda p: p.unlink(missing_ok=True))
            task = asyncio.create_task(job())
            try:
                await asyncio.wait_for(ready.wait(), 2)
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()  # Parent/child shutdown may deliver cancellation twice.
                await asyncio.sleep(0)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(path.exists())
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    async def test_unknown_transport_and_missing_receipt_are_not_success(self):
        for send in (AsyncMock(side_effect=ConnectionError("lost")), AsyncMock(return_value={})):
            status, receipt = await deliver_once(send)
            self.assertEqual(status, "unconfirmed")
            self.assertIsNone(receipt)
            send.assert_awaited_once()

    async def test_explicit_rejection_remains_available_for_safe_fallback(self):
        error = RuntimeError("rejected")
        error.info = {"retcode": 100, "message": "rejected"}
        with self.assertRaises(RuntimeError):
            await deliver_once(AsyncMock(side_effect=error))

    async def test_receipt_is_retained_even_if_session_changed_during_send(self):
        async def send():
            conversation.invalidate("group_998802")
            return {"message_id": 42}
        with conversation.session_scope("group_998802"):
            self.assertEqual(await deliver_once(send), ("sent", {"message_id": 42}))
