"""Local file queue. A durable receipt permits at most one send attempt per ID."""
import asyncio
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path


MAX_TEXT = 1800


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _filename(task_id):
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest() + ".json"


def _temporary_json(directory, value):
    path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False) as file:
            path = Path(file.name)
            json.dump(value, file, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        return path
    except BaseException:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


def _save(path, value):
    temporary = _temporary_json(path.parent, value)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _claim(path, value):
    try:
        with path.open("x", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        return True
    except FileExistsError:
        return False


def enqueue(directory, task_id, text, ttl=3600):
    """Publish a complete job atomically; retries with the same ID never replace it."""
    if not isinstance(task_id, str) or not 1 <= len(task_id.strip()) <= 200:
        raise ValueError("Task ID must contain 1..200 characters")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise ValueError(f"Text must contain 1..{MAX_TEXT} characters")
    if not _finite_number(ttl) or ttl <= 0:
        raise ValueError("TTL must be a positive number of seconds")
    root = Path(directory)
    pending = root / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    path = pending / _filename(task_id)
    receipt = root / "receipts" / path.name
    if receipt.exists():
        return {"id": task_id, "queued": False, "state": "already_attempted"}
    payload = {"id": task_id, "text": text, "expires_at": time.time() + ttl}
    temporary = _temporary_json(pending, payload)
    try:
        # Same-directory hard link publishes without exposing a partial JSON or
        # overwriting a concurrent producer's job. Requires a local filesystem.
        try:
            os.link(temporary, path)
        except FileExistsError:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("id") != task_id or previous.get("text") != text:
                raise ValueError("Task ID already exists with different content")
            return {"id": task_id, "queued": False, "state": "already_queued"}
    finally:
        temporary.unlink(missing_ok=True)
    return {"id": task_id, "queued": True, "state": "pending"}


async def drain(directory, send, *, limit=20, timeout=20):
    """send(text) returns notify_master's (status, receipt). Never retry uncertainty."""
    root = Path(directory)
    if not (root / "pending").is_dir():
        return {}
    receipts = root / "receipts"
    receipts.mkdir(exist_ok=True)
    counts = {}
    for path in sorted((root / "pending").glob("*.json")):
        if sum(counts.values()) >= limit:
            break
        receipt = receipts / path.name
        if receipt.exists():
            continue
        record = {"id": path.stem, "updated_at": time.time()}
        status = "sending"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            task_id, text, expires = payload["id"], payload["text"], payload["expires_at"]
            if (not isinstance(task_id, str) or not 1 <= len(task_id.strip()) <= 200
                    or _filename(task_id) != path.name or not isinstance(text, str)
                    or not text.strip() or len(text) > MAX_TEXT
                    or not _finite_number(expires)):
                raise ValueError("Invalid job")
            record.update(id=task_id, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
            if time.time() >= expires:
                status = "expired"
        except (ValueError, UnicodeError, KeyError, TypeError, OverflowError):
            status = "rejected"
        if not _claim(receipt, {**record, "status": status}):
            continue
        if status == "sending":
            try:
                result, acknowledgement = await asyncio.wait_for(send(text), timeout=timeout)
                if result != "sent" or not isinstance(acknowledgement, dict) or acknowledgement.get("message_id") is None:
                    raise ValueError("Send not confirmed")
                status = "sent"
                record["message_id"] = acknowledgement["message_id"]
            except asyncio.CancelledError:
                _save(receipt, {**record, "status": "unknown", "error_type": "CancelledError"})
                raise
            except Exception as error:
                status = "unknown"
                record["error_type"] = type(error).__name__
            _save(receipt, {**record, "status": status})
        counts[status] = counts.get(status, 0) + 1
    return counts
