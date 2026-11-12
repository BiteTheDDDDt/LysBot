"""Small SQLite write timing helpers; no queue or background task."""
from contextlib import asynccontextmanager
import time
from .logging_setup import general_logger


SQLITE_WRITE_RETRIES = 3
SQLITE_WRITE_RETRY_BASE_DELAY = 0.5


@asynccontextmanager
async def measured_write_lock(lock, operation):
    queued = time.monotonic()
    async with lock:
        acquired = time.monotonic()
        try:
            yield
        finally:
            waited, held = acquired - queued, time.monotonic() - acquired
            if max(waited, held) >= 1:
                general_logger.warning(f"[sqlite_write] operation={operation} wait={waited:.3f}s held={held:.3f}s")
