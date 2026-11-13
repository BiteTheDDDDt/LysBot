"""Pinned, offline Chinese BGE inference. No downloads or remote fallback at runtime."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from pathlib import Path

MODEL = "local:bge-small-zh-v1.5"
DIMENSION = 512
REVISION = "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59"
REPOSITORY = "Qdrant/bge-small-zh-v1.5"
FILES = {
    "model_optimized.onnx": "1294ea4b6331115a353d81f96b85e8c8d7fdcc284453d5b2fab5b016230aad38",
    "tokenizer.json": "48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26",
}
POLICY = "bge-zh-query-cls-448-overlap64-mean-v1"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
WINDOW = 448
OVERLAP = 64
_lock = threading.Lock()
_engine = None
_engine_key = None


def validate_settings(settings):
    if settings["model"] != MODEL or settings["revision"] != REVISION or settings["dimension"] != DIMENSION:
        raise ValueError("local_embedding_model_revision_or_dimension_mismatch")
    if settings.get("task_policy") != POLICY or settings.get("artifacts") != FILES:
        raise ValueError("local_embedding_policy_mismatch")


def model_directory():
    from .config import config
    from .paths import DATA_DIR

    return (
        Path(config.memory_local_model_dir) if config.memory_local_model_dir else DATA_DIR / "models/bge-small-zh-v1.5"
    )


def verify_files(directory):
    for filename, expected in FILES.items():
        path = Path(directory) / filename
        if not path.is_file():
            raise ValueError("local_embedding_files_missing:run_prepare_local_embedding")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("local_embedding_artifact_hash_mismatch:" + filename)


def token_windows(ids):
    """Cover all tokens, including the tail; no tokenizer truncation."""
    result = []
    for start in range(0, max(1, len(ids)), WINDOW - OVERLAP):
        result.append(ids[start : start + WINDOW])
        if start + WINDOW >= len(ids):
            break
    return result


class LocalEmbedding:
    def __init__(self, directory, threads):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        verify_files(directory)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(8, int(threads)))
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(Path(directory) / "model_optimized.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.cls = self.tokenizer.token_to_id("[CLS]")
        self.sep = self.tokenizer.token_to_id("[SEP]")
        self.pad = self.tokenizer.token_to_id("[PAD]")
        if None in (self.cls, self.sep, self.pad):
            raise ValueError("local_embedding_special_tokens_missing")

    def encode(self, texts, query=False):
        import numpy as np

        if not texts:
            return []
        if len(texts) > 32 or any(not isinstance(text, str) or len(text) > 16000 for text in texts):
            raise ValueError("local_embedding_batch_too_large")
        segments = []
        for owner, text in enumerate(texts):
            ids = self.tokenizer.encode((QUERY_PREFIX if query else "") + text, add_special_tokens=False).ids
            for window in token_windows(ids):
                segments.append((owner, [self.cls, *window, self.sep]))
        # Similar lengths avoid large padding costs. Restore each document's order below.
        segments.sort(key=lambda segment: len(segment[1]))
        sums = np.zeros((len(texts), 512), dtype=np.float32)
        counts = np.zeros(len(texts), dtype=np.int32)
        for start in range(0, len(segments), 8):
            batch = segments[start : start + 8]
            width = max(len(ids) for _, ids in batch)
            input_ids = np.full((len(batch), width), self.pad, dtype=np.int64)
            mask = np.zeros_like(input_ids)
            for row, (_, ids) in enumerate(batch):
                input_ids[row, : len(ids)] = ids
                mask[row, : len(ids)] = 1
            values = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": mask,
                    "token_type_ids": np.zeros_like(input_ids),
                },
            )[0][:, 0, :]
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if values.shape != (len(batch), 512) or not np.isfinite(values).all() or (norms == 0).any():
                raise ValueError("local_embedding_invalid_output")
            values = values / norms
            for row, (owner, _) in enumerate(batch):
                sums[owner] += values[row]
                counts[owner] += 1
        means = sums / counts[:, None]
        return means.tolist()  # Shared embed() validates and L2-normalizes the final document vectors.


def encode_sync(texts, settings, query=False):
    global _engine, _engine_key
    from .config import config

    validate_settings(settings)
    directory = model_directory().resolve()
    key = (directory, config.memory_local_threads)
    # One shared model and bounded ONNX threads prevent three backfill batches from oversubscribing the CPU.
    with _lock:
        if _engine_key != key:
            _engine = LocalEmbedding(directory, config.memory_local_threads)
            _engine_key = key
        return _engine.encode(texts, query=query)


async def encode_local(texts, settings, query=False):
    pending = asyncio.create_task(asyncio.to_thread(encode_sync, texts, settings, query))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        # Native inference cannot be interrupted. Drain this bounded batch before closing the worker.
        await asyncio.gather(pending, return_exceptions=True)
        raise
