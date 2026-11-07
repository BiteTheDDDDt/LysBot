import asyncio
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, patch


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/queued_notifications"
spec = importlib.util.spec_from_file_location("notification_example_storage", EXAMPLE / "storage.py")
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


class QueuedNotificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "queue"

    def receipt(self):
        return json.loads(next((self.root / "receipts").glob("*.json")).read_text(encoding="utf-8"))

    async def test_standalone_cli_submits_without_loading_bot(self):
        result = subprocess.run([
            sys.executable, str(EXAMPLE / "enqueue.py"), "--directory", str(self.root),
            "--id", "build-demo", "--text", "构建已完成", "--ttl", "60",
        ], capture_output=True, text=True, encoding="utf-8", check=True)
        self.assertEqual(json.loads(result.stdout), {"id": "build-demo", "queued": True, "state": "pending"})
        self.assertEqual(len(list((self.root / "pending").glob("*.json"))), 1)

    async def test_concurrent_producers_publish_one_complete_job(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: storage.enqueue(self.root, "build/../same-id", "完成"), range(8)))
        self.assertEqual(sum(result["queued"] for result in results), 1)
        jobs = list((self.root / "pending").iterdir())
        self.assertEqual(len(jobs), 1)
        self.assertEqual(len(jobs[0].stem), 64)
        self.assertEqual(json.loads(jobs[0].read_text(encoding="utf-8"))["text"], "完成")
        with self.assertRaises(ValueError):
            storage.enqueue(self.root, "build/../same-id", "另一份内容")

    async def test_invalid_submissions_do_not_create_jobs(self):
        for task_id, text, ttl in [("", "完成", 1), ("id", " ", 1), ("id", "x" * 1801, 1),
                                   ("id", "完成", 0), ("id", "完成", True), ("id", "完成", float("nan"))]:
            with self.subTest(task_id=task_id, ttl=ttl), self.assertRaises(ValueError):
                storage.enqueue(self.root, task_id, text, ttl)
        self.assertFalse(self.root.exists())

    async def test_restart_and_resubmission_do_not_duplicate_delivery(self):
        storage.enqueue(self.root, "report-demo", "报告完成")
        send = AsyncMock(return_value=("sent", {"message_id": 7}))
        self.assertEqual(await storage.drain(self.root, send), {"sent": 1})
        self.assertEqual(await storage.drain(self.root, send), {})
        self.assertFalse(storage.enqueue(self.root, "report-demo", "报告完成")["queued"])
        send.assert_awaited_once_with("报告完成")
        self.assertEqual(self.receipt()["message_id"], 7)

    async def test_two_consumers_make_one_send_attempt(self):
        storage.enqueue(self.root, "report-demo", "报告完成")
        send = AsyncMock(return_value=("sent", {"message_id": 8}))
        results = await asyncio.gather(storage.drain(self.root, send), storage.drain(self.root, send))
        self.assertEqual(sum(result.get("sent", 0) for result in results), 1)
        send.assert_awaited_once()

    async def test_expired_and_invalid_jobs_are_not_sent(self):
        with patch.object(storage.time, "time", return_value=100):
            storage.enqueue(self.root, "expired", "旧报告", ttl=5)
        (self.root / "pending/broken.json").write_text("{", encoding="utf-8")
        send = AsyncMock()
        with patch.object(storage.time, "time", return_value=106):
            self.assertEqual(await storage.drain(self.root, send), {"expired": 1, "rejected": 1})
        send.assert_not_awaited()

    async def test_unconfirmed_send_is_not_retried(self):
        storage.enqueue(self.root, "report-demo", "报告完成")
        send = AsyncMock(return_value=("unconfirmed", None))
        self.assertEqual(await storage.drain(self.root, send), {"unknown": 1})
        await storage.drain(self.root, send)
        send.assert_awaited_once()
        self.assertEqual(self.receipt()["status"], "unknown")

    async def test_timeout_does_not_repeat_a_potentially_delivered_message(self):
        storage.enqueue(self.root, "report-demo", "报告完成")
        send = AsyncMock(side_effect=TimeoutError())
        await storage.drain(self.root, send)
        await storage.drain(self.root, send)
        send.assert_awaited_once()
        self.assertEqual(self.receipt()["error_type"], "TimeoutError")

    async def test_cancellation_leaves_a_nonretryable_receipt(self):
        storage.enqueue(self.root, "report-demo", "报告完成")
        started = asyncio.Event()

        async def send(text):
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(storage.drain(self.root, send))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        again = AsyncMock()
        await storage.drain(self.root, again)
        again.assert_not_awaited()
        self.assertEqual(self.receipt()["status"], "unknown")

    async def test_receipt_write_failure_keeps_claim_and_prevents_resend(self):
        storage.enqueue(self.root, "report-demo", "报告完成")
        send = AsyncMock(return_value=("sent", {"message_id": 9}))
        with patch.object(storage, "_save", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                await storage.drain(self.root, send)
        await storage.drain(self.root, send)
        send.assert_awaited_once()
        self.assertEqual(self.receipt()["status"], "sending")

    async def test_incomplete_existing_claim_is_not_retried(self):
        storage.enqueue(self.root, "report-demo", "报告完成")
        receipt = self.root / "receipts" / next((self.root / "pending").glob("*.json")).name
        receipt.parent.mkdir()
        receipt.write_text("", encoding="utf-8")
        send = AsyncMock()
        self.assertEqual(await storage.drain(self.root, send), {})
        send.assert_not_awaited()

    async def test_batch_limit_leaves_remaining_jobs_for_next_poll(self):
        storage.enqueue(self.root, "first", "第一份报告")
        storage.enqueue(self.root, "second", "第二份报告")
        send = AsyncMock(return_value=("sent", {"message_id": 10}))
        self.assertEqual(await storage.drain(self.root, send, limit=1), {"sent": 1})
        self.assertEqual(await storage.drain(self.root, send, limit=1), {"sent": 1})
        self.assertEqual(send.await_count, 2)

    async def test_installed_plugin_uses_master_api_and_waits_for_one_bot(self):
        code = '''import asyncio,json,os,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
os.environ.update(LLM_CHAT_DATA_DIR=sys.argv[1],LLM_CHAT_DATABASE_PATH=str(Path(sys.argv[1])/'history.db'))
import nonebot
nonebot.init(superusers={'10001'})
plugin=nonebot.load_plugin('examples.queued_notifications')
assert plugin is not None
worker=plugin.module
from examples.queued_notifications.storage import enqueue
from nonebot_plugin_apscheduler import scheduler
assert scheduler.get_job('queued_notifications') is not None
root=Path(sys.argv[1])/'queued_notifications'
enqueue(root,'report','Synthetic report complete')
path=next((root/'pending').glob('*.json'))
payload=json.loads(path.read_text());payload['recipient']='99999';path.write_text(json.dumps(payload))
bot=SimpleNamespace(send_private_msg=AsyncMock(return_value={'message_id':42}))
async def run():
 with patch.object(worker,'get_bots',return_value={'10002':bot}):
  await worker.deliver_pending();await worker.deliver_pending()
 bot.send_private_msg.assert_awaited_once()
 assert bot.send_private_msg.call_args.kwargs['user_id']==10001
 enqueue(root,'later','Another report')
 for bots in [{},{'10002':bot,'10003':bot}]:
  with patch.object(worker,'get_bots',return_value=bots):await worker.deliver_pending()
 assert bot.send_private_msg.await_count==1
 assert len(list((root/'receipts').glob('*.json')))==1
 print('PLUGIN_INTEGRATION_OK')
asyncio.run(run())
'''
        result = subprocess.run([sys.executable, "-c", code, str(self.root)], capture_output=True,
                                text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        self.assertIn("PLUGIN_INTEGRATION_OK", result.stdout)
