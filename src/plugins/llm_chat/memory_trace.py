"""Record bounded injected references, without another model request."""

import asyncio
import json
import re

from .memory_index import connect, generation
from .memory_changes import version


async def reference_labels(path, sid, memories):
    if not any(item.get("memory_id") for kind in ("summaries", "important_info") for item in memories.get(kind, [])):
        return

    def load():
        with connect(path) as db:
            return generation(db, sid)

    gen = await asyncio.to_thread(load)
    for kind in ("summaries", "important_info"):
        for item in memories.get(kind, []):
            if item.get("memory_id"):
                item["memory_ref"] = f"[h:{gen}:{kind}:{item['memory_id']}]"


async def attach_refs(path, sid, messages):
    if not any(m.is_reference and re.search(r"h:\d+:(summaries|important_info):\d+", m.content) for m in messages):
        return

    def load():
        with connect(path) as db:
            gen = generation(db, sid)
            for message in messages:
                if not message.is_reference:
                    continue
                refs = []
                for ref, kind, mid in re.findall(rf"(h:{gen}:(summaries|important_info):(\d+))", message.content):
                    row = db.execute(f"SELECT * FROM {kind} WHERE id=? AND session_id=?", (int(mid), sid)).fetchone()
                    if row:
                        meta = json.loads(row["provenance_json"])
                        refs.append(
                            {
                                "id": ref,
                                "version": version(row),
                                "content": row["content"][:500],
                                "source_ids": meta.get("source_ids", [])[:40],
                                "assertion": meta.get("assertion"),
                                "stage": "injected",
                            }
                        )
                message.memory_refs = refs

    await asyncio.to_thread(load)
