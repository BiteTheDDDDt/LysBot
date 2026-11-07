"""Bounded, session-scoped originals for image edits; only IDs enter text history."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import time


@dataclass(frozen=True)
class ImageReference:
    id: str
    base64: str
    label: str
    created_at: float


class ImageReferenceStore:
    def __init__(self, *, max_items=128, max_bytes=32 * 1024 * 1024, ttl=1800):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.ttl = ttl
        self._entries: OrderedDict[tuple[str, str], ImageReference] = OrderedDict()
        self._bytes = 0

    def _remove(self, key):
        self._bytes -= len(self._entries.pop(key).base64)

    def _prune(self):
        now = time.monotonic()
        for key, ref in list(self._entries.items()):
            if now - ref.created_at >= self.ttl:
                self._remove(key)
        while self._entries and (
            len(self._entries) > self.max_items or self._bytes > self.max_bytes
        ):
            self._remove(next(iter(self._entries)))

    def remember(self, session_id: str, encoded: str, label: str) -> ImageReference:
        ref_id = "img_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        ref = ImageReference(ref_id, encoded, " ".join(label.split())[:200], time.monotonic())
        key = (session_id, ref_id)
        if key in self._entries:
            self._remove(key)
        self._entries[key] = ref
        self._bytes += len(encoded)
        self._prune()
        # The active turn can use an original even when it exceeds the cache budget.
        return ref

    def snapshot(self, session_id: str) -> dict[str, ImageReference]:
        self._prune()
        return {ref.id: ref for (sid, _), ref in self._entries.items() if sid == session_id}

    def clear(self, session_id: str | None = None):
        for key in list(self._entries):
            if session_id is None or key[0] == session_id:
                self._remove(key)


image_reference_store = ImageReferenceStore()


def reference_marker(ref: ImageReference) -> str:
    return f"[图片引用:{ref.id}]"
