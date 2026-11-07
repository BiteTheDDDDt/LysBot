"""Small, transactional edits to extracted facts; original messages stay immutable."""

from __future__ import annotations

import json
import re
import time
import uuid

from .memory_index import connect, digest, encode, generation


def active_sql(kind, alias=None):
    """Shared correction visibility; historical supersession remains profile policy."""
    if kind == "messages":
        return "1"
    table = alias or kind
    return f"COALESCE(json_extract({table}.provenance_json,'$.memory_state'),'active')='active'"


def correction_epoch(db, sid):
    row = db.execute("SELECT value FROM kv_meta WHERE key=?", ("memory_correction:" + sid,)).fetchone()
    return row[0] if row else ""


def snapshot(row):
    return {key: row[key] for key in ("user_id", "info_type", "content", "provenance_json")}


def version(row):
    return digest(encode(dict(row)))


def _invalidate(db, sid):
    # Summaries may contain indirect conclusions with incomplete dependency metadata.
    # Quarantine existing summaries rather than launch a background LLM rewrite.
    db.execute(
        "UPDATE summaries SET provenance_json=json_set(provenance_json,'$.memory_state','stale') WHERE session_id=?",
        (sid,),
    )
    db.execute(
        "INSERT INTO kv_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        ("memory_correction:" + sid, uuid.uuid4().hex),
    )
    cutoff = db.execute("SELECT COALESCE(max(id),0) FROM messages_detail WHERE session_id=?", (sid,)).fetchone()[0]
    db.execute(
        "INSERT INTO kv_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        ("memory_corrected_through:" + sid, str(cutoff)),
    )
    # In-flight extraction already checks the job still exists before committing.
    db.execute("DELETE FROM important_info_jobs WHERE session_id=?", (sid,))


def change(
    path,
    sid,
    actor,
    *,
    action,
    ref="",
    content=None,
    owner_id=None,
    assertion=None,
    reason="",
    operation_id="",
    confirmation="",
    admin=False,
    correction_source_ids=(),
):
    """One fact per preview. Execute/undo require the exact user-visible receipt."""
    if not actor:
        raise ValueError("missing_actor")
    with connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        gen = generation(db, sid)
        if action == "preview":
            match = re.fullmatch(r"h:(\d+):important_info:(\d+)", ref)
            if not match or int(match[1]) != gen:
                raise ValueError("current_fact_reference_required")
            row = db.execute(
                "SELECT * FROM important_info WHERE session_id=? AND id=?", (sid, int(match[2]))
            ).fetchone()
            if not row or (not admin and str(row["user_id"]) != str(actor)):
                raise ValueError("only_own_memory_or_admin_can_change")
            before = snapshot(row)
            after = dict(before)
            meta = json.loads(after["provenance_json"])
            if content is not None:
                if not isinstance(content, str) or not 1 <= len(content.strip()) <= 1000:
                    raise ValueError("content_must_be_1_to_1000_chars")
                after["content"] = content.strip()
            if owner_id is not None:
                if type(owner_id) is not int or owner_id <= 0:
                    raise ValueError("invalid_owner")
                if not db.execute(
                    "SELECT 1 FROM messages_detail WHERE session_id=? AND sender_id=? LIMIT 1", (sid, owner_id)
                ).fetchone():
                    raise ValueError("owner_not_known_in_current_chat")
                after["user_id"] = owner_id
            if assertion is not None:
                if assertion not in {"asserted", "negated", "planned", "reported", "uncertain", "withdrawn"}:
                    raise ValueError("invalid_assertion")
                if assertion == "withdrawn":
                    meta["memory_state"] = "withdrawn"
                else:
                    meta["assertion"] = assertion
            # Attribution corrections must not silently strengthen reported claims.
            if owner_id is not None and owner_id != before["user_id"] and assertion is None:
                meta["assertion"] = "reported"
            op = uuid.uuid4().hex[:12]
            meta["manual_change"] = op
            support = []
            hashes = {}
            for mid in correction_source_ids:
                source = db.execute(
                    "SELECT content FROM messages_detail WHERE id=? AND session_id=? AND sender_id=? AND role='user'",
                    (mid, sid, actor),
                ).fetchone()
                if source:
                    support.append(mid)
                    hashes[str(mid)] = digest(source[0])
            meta.update(
                provenance="manual",
                source_ids=support,
                source_hashes=hashes,
                authors=[str(actor)],
                correction={"actor": str(actor), "reason": reason, "operation_id": op},
            )
            after["provenance_json"] = encode(meta)
            if not reason.strip() or len(reason) > 500:
                raise ValueError("correction_reason_required_max_500")
            db.execute(
                "INSERT INTO memory_changes VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    op,
                    sid,
                    gen,
                    str(actor),
                    row["id"],
                    "preview",
                    encode(before),
                    encode(after),
                    reason,
                    time.time(),
                    None,
                    correction_epoch(db, sid),
                ),
            )
            return {
                "status": "preview",
                "operation_id": op,
                "before": before,
                "after": after,
                "confirm_text": "确认记忆 " + op,
                "note": "请展示修改前后内容；用户发送 confirm_text 后执行。原消息不变，旧摘要暂停使用。",
            }
        if action not in {"apply", "undo"}:
            raise ValueError("invalid_change_action")
        op = db.execute(
            "SELECT * FROM memory_changes WHERE id=? AND session_id=? AND generation=?", (operation_id, sid, gen)
        ).fetchone()
        if not op or (not admin and op["actor"] != str(actor)):
            raise ValueError("operation_not_available")
        expected_text = ("确认记忆 " if action == "apply" else "撤销记忆 ") + operation_id
        if confirmation.strip() != expected_text:
            raise ValueError("user_must_send: " + expected_text)
        target_status = "applied" if action == "apply" else "undone"
        if op["status"] == target_status:
            return {"status": target_status, "operation_id": operation_id, "already_done": True}
        if op["status"] != ("preview" if action == "apply" else "applied"):
            raise ValueError("operation_state_conflict")
        if action == "apply" and (time.time() - op["created_at"] > 1800 or op["epoch"] != correction_epoch(db, sid)):
            raise ValueError("preview_expired_recreate")
        row = db.execute("SELECT * FROM important_info WHERE session_id=? AND id=?", (sid, op["fact_id"])).fetchone()
        expected = json.loads(op["before_json"] if action == "apply" else op["after_json"])
        replacement = json.loads(op["after_json"] if action == "apply" else op["before_json"])
        if not row or snapshot(row) != expected:
            raise ValueError("memory_changed_since_preview_or_edit")
        db.execute(
            "UPDATE important_info SET user_id=?,info_type=?,content=?,provenance_json=? WHERE id=? AND session_id=?",
            (*[replacement[k] for k in ("user_id", "info_type", "content", "provenance_json")], op["fact_id"], sid),
        )
        db.execute(
            "UPDATE memory_changes SET status=?,changed_at=? WHERE id=?", (target_status, time.time(), operation_id)
        )
        _invalidate(db, sid)
        return {
            "status": target_status,
            "operation_id": operation_id,
            "memory": replacement,
            "undo_text": "撤销记忆 " + operation_id,
        }


def inspect(path, sid, *, owner_id=None, run_id=None):
    with connect(path) as db:
        gen = generation(db, sid)
        if run_id:
            row = db.execute(
                "SELECT state_json FROM agent_runs WHERE id=? AND session_id=? AND generation=?", (run_id, sid, gen)
            ).fetchone()
            if not row:
                raise ValueError("answer_record_unavailable")
            state = json.loads(row[0])
            return {
                "status": "ok",
                "injected_memories": state.get("injected_memories", []),
                "retrieved_evidence": [
                    {
                        "id": key,
                        "excerpt": value.get("excerpt", "")[:500],
                        "source_ids": value.get("source_ids", [])[:20],
                        "read_state": value.get("read_state"),
                    }
                    for key, value in list(state.get("evidence", {}).items())[:10]
                ],
                "cited_evidence_ids": (state.get("final_answer") or {}).get("evidence_ids", []),
                "note": "注入、检索与明确引用是不同记录；不能把进入上下文解释为确定的因果。检索依据最多展示10条短摘录，完整内容用 read_history。",
            }
        if owner_id is None:
            raise ValueError("owner_id_or_run_id_required")
        rows = db.execute(
            "SELECT * FROM important_info WHERE session_id=? AND user_id=? ORDER BY id DESC LIMIT 30", (sid, owner_id)
        ).fetchall()
        return {
            "status": "ok",
            "memories": [
                {
                    "ref": f"h:{gen}:important_info:{r['id']}",
                    "user_id": r["user_id"],
                    "content": r["content"][:200],
                    "created_at": r["created_at"],
                    "provenance": {
                        key: value
                        for key, value in json.loads(r["provenance_json"]).items()
                        if key in {"assertion", "source_ids", "memory_state", "correction", "supersedes"}
                    },
                }
                for r in rows
            ],
            "note": "仅最近30条，正文最多200字；历史或撤回记录不是当前事实。更多记录用 query_history，原文用 read_history。",
            "changes": [
                dict(r)
                for r in db.execute(
                    "SELECT id,fact_id,status,reason,created_at,changed_at FROM memory_changes WHERE session_id=? AND generation=? AND fact_id IN (SELECT id FROM important_info WHERE session_id=? AND user_id=?) ORDER BY created_at DESC LIMIT 20",
                    (sid, gen, sid, owner_id),
                )
            ],
        }
