import asyncio
import copy
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import local_embedding as local
from src.plugins.llm_chat.config import config
from src.plugins.llm_chat.memory_index import embed, embedding_settings, embedding_version


class LocalEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    def settings(self):
        with (
            patch.object(config, "memory_embedding_model", local.MODEL),
            patch.object(config, "memory_embedding_revision", local.REVISION),
        ):
            return embedding_settings()

    def test_windows_include_every_token_and_final_tail(self):
        for size in [0, 1, 447, 448, 449, 832, 833, 1701]:
            ids = list(range(size))
            windows = local.token_windows(ids)
            self.assertEqual({value for window in windows for value in window}, set(ids))
            self.assertTrue(all(len(window) <= 448 for window in windows))
            if size:
                self.assertEqual(windows[-1][-1], ids[-1])

    def test_fingerprint_covers_artifacts_policy_and_revision(self):
        settings = self.settings()
        original = embedding_version(settings)
        for field in ["artifacts", "task_policy", "revision"]:
            changed = copy.deepcopy(settings)
            changed[field] = "changed"
            self.assertNotEqual(embedding_version(changed), original)
            with self.assertRaises(ValueError):
                local.validate_settings(changed)

    def test_missing_or_tampered_model_fails_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "files_missing"):
                local.verify_files(directory)
            (Path(directory) / "model_optimized.onnx").write_bytes(b"incorrect model")
            with self.assertRaisesRegex(ValueError, "hash_mismatch"):
                local.verify_files(directory)

    async def test_local_dispatch_needs_no_provider_credentials_or_http(self):
        with patch.object(local, "encode_local", AsyncMock(return_value=[[3, 4] + [0] * 510])) as encode:
            result = await embed(["中文问题"], self.settings(), query=True)
        self.assertAlmostEqual(struct.unpack("<512f", result[0])[0], 0.6)
        self.assertEqual(result.usage, {})
        self.assertTrue(encode.call_args.kwargs["query"])

    async def test_cancel_waits_for_inflight_native_batch(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def compute(*args):
            entered.set()
            release.wait(2)
            finished.set()
            return [[1] + [0] * 511]

        with patch.object(local, "encode_sync", compute):
            task = asyncio.create_task(local.encode_local(["文本"], self.settings()))
            await asyncio.wait_for(asyncio.to_thread(entered.wait), 2)
            task.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(finished.is_set())
