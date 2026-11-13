"""Scoped hybrid discovery and original-source navigation for the chat agent."""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import re
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone

from .config import config
from .memory_index import CHUNK_VERSION, SOURCE_TABLES, connect, digest, embed, encode, generation, tokens

_CURSOR_SECRET = secrets.token_bytes(32)
_QUERY_STOP_WORDS = frozenset(
    [
        "的",
        "了",
        "和",
        "是",
        "在",
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "谁",
        "有人",
        "哪个",
        "哪些",
        "什么",
        "怎么",
        "怎么样",
        "是否",
        "吗",
        "呢",
        "吧",
        "啊",
        "都",
        "就",
        "也",
        "还",
        "又",
        "被",
        "把",
        "给",
        "让",
        "与",
        "及",
        "或",
        "而",
        "一个",
        "一些",
        "这个",
        "那个",
        "这",
        "那",
        "有",
        "没有",
        "过",
        "想",
        "想要",
        "需要",
        "请",
        "帮",
        "找",
        "查",
        "关于",
        "说",
        "聊",
        "聊过",
        "提到",
        "提到了",
    ]
)


def result(status="ok", items=None, **metadata):
    return {"status": status, "items": items or [], "truncated": False, "next_cursor": None, "warnings": [], **metadata}


def failure(code, *, retryable=False, invalid=False):
    return result("invalid_arguments" if invalid else "unavailable", error={"code": code, "retryable": retryable})


def parse_time(value):
    if not value:
        return None
    match = re.fullmatch(r"(\d{1,5})([mhd])", str(value).strip().lower())
    if match:
        return (
            datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)
            - timedelta(seconds=int(match[1]) * {"m": 60, "h": 3600, "d": 86400}[match[2]])
        ).isoformat()
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo:
        parsed = parsed.astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None)
    return parsed.isoformat()


def cursor_encode(key, offset, *, state=None):
    payload = encode({"key": key, "offset": offset, "state": state}).encode()
    signature = hmac.digest(_CURSOR_SECRET, payload, "sha256")
    return base64.urlsafe_b64encode(signature + payload).decode()


def cursor_data(cursor):
    if not isinstance(cursor, str) or len(cursor) > 2048:
        raise ValueError("invalid_cursor")
    try:
        raw = base64.urlsafe_b64decode(cursor)
        if not hmac.compare_digest(raw[:32], hmac.digest(_CURSOR_SECRET, raw[32:], "sha256")):
            raise ValueError("cursor_expired_or_modified")
        return json.loads(raw[32:].decode())
    except (ValueError, UnicodeError) as error:
        raise ValueError("invalid_cursor") from error


def cursor_offset(cursor, key):
    if not cursor:
        return 0
    data = cursor_data(cursor)
    if data.get("key") != key or type(data.get("offset")) is not int or not 0 <= data["offset"] <= 1048576:
        raise ValueError("expired_or_mismatched_cursor")
    return data["offset"]


class HistorySearch:
    def __init__(self, path):
        self.path = path

    def _connect(self, *, vectors=False, deadline=None):
        return connect(self.path, vectors=vectors, deadline=deadline or time.monotonic() + 8)

    def _support_live(self, db, row, kind, sid):
        if kind == "messages":
            return True
        from .memory_changes import active_sql
        if not db.execute(f"SELECT 1 FROM {kind} WHERE id=? AND session_id=? AND {active_sql(kind)}",
                          (row["id"], sid)).fetchone():
            return False
        meta = json.loads(row["provenance_json"] or "{}")
        hashes = meta.get("source_hashes", {})
        for mid in meta.get("source_ids", []):
            source = db.execute(
                "SELECT content FROM messages_detail WHERE id=? AND session_id=? AND COALESCE(is_thinking,0)=0",
                (mid, sid),
            ).fetchone()
            if not source or (str(mid) in hashes and digest(source[0]) != hashes[str(mid)]):
                return False
        return True

    async def profile(self, sid, owners, budget=1000):
        """A bounded current view, computed from assertions and live support."""

        def load():
            lines = []
            with self._connect() as db:
                gen = generation(db, sid)
                for owner in list(dict.fromkeys(owners))[:5]:
                    records = []
                    for row in db.execute(
                        """SELECT * FROM important_info WHERE session_id=? AND user_id=?
                        AND info_type<>'emotion_state' ORDER BY id DESC LIMIT 80""",
                        (sid, owner),
                    ):
                        meta = json.loads(row["provenance_json"] or "{}")
                        if not self._support_live(db, row, "important_info", sid):
                            continue
                        records.append((dict(row), meta))
                    superseded = {
                        mid
                        for _, meta in records
                        if meta.get("assertion") in {"asserted", "negated"}
                        for mid in meta.get("supersedes", [])
                    }
                    grouped = {}
                    legacy = []
                    for row, meta in records:
                        if row["id"] in superseded:
                            continue
                        if meta.get("provenance") not in {"derived", "manual"}:
                            if len(legacy) < 2:
                                legacy.append(
                                    f"旧提取，当前有效性待核实：{row['content'][:120]} [h:{gen}:important_info:{row['id']}]"
                                )
                            continue
                        label = {
                            "asserted": "陈述",
                            "planned": "计划",
                            "negated": "否定",
                            "reported": "转述",
                            "uncertain": "未确认",
                        }.get(meta.get("assertion"), "未确认")
                        if meta.get("manual_change"):
                            label = "人工纠正·" + label
                        attr = meta.get("attribute") or row["info_type"]
                        grouped.setdefault(attr, []).append((row, meta, label))
                    for attr, group in list(grouped.items())[:5]:
                        conflict = len({r["content"] for r, m, _ in group if m.get("assertion") == "asserted"}) > 1
                        for row, meta, label in group[:2]:
                            lines.append(
                                f"用户 {owner} · {attr} · {'说法待核对' if conflict else label}：{row['content'][:180]} [h:{gen}:important_info:{row['id']}]"
                            )
                    lines.extend(f"用户 {owner} · {line}" for line in legacy)
            chosen = []
            length = 0
            for line in lines:
                if length + len(line) + 1 > budget:
                    break
                chosen.append(line)
                length += len(line) + 1
            return "\n".join(chosen)

        try:
            return await asyncio.to_thread(load)
        except (sqlite3.Error, OSError):
            return None

    def _snapshot(self, sid):
        with self._connect() as db:
            db.execute("BEGIN")
            gen = generation(db, sid)
            revision = db.execute(
                "SELECT value FROM kv_meta WHERE key=?", ("memory_revision:" + sid,)
            ).fetchone()
            active = db.execute("SELECT value FROM kv_meta WHERE key='memory_active_embedding'").fetchone()
            version = active[0] if active else ""
            settings = db.execute("SELECT value FROM kv_meta WHERE key=?", ("memory_embedding:" + version,)).fetchone()
            jobs = db.execute(
                "SELECT count(*),min(source_id) FROM memory_index_jobs WHERE session_id=? AND generation=?", (sid, gen)
            ).fetchone()
            chunk = db.execute("SELECT value FROM kv_meta WHERE key='memory_active_chunk'").fetchone()
            by_source = {}
            for kind, table in SOURCE_TABLES.items():
                pending = db.execute(
                    "SELECT count(*),min(source_id),sum(attempts>=8) FROM memory_index_jobs WHERE session_id=? AND generation=? AND kind=?",
                    (sid, gen, kind),
                ).fetchone()
                high = db.execute(f"SELECT max(id) FROM {table} WHERE session_id=?", (sid,)).fetchone()[0] or 0
                by_source[kind] = {
                    "source_high_watermark": high,
                    "continuous_projected_through": min(high, max(0, pending[1] - 1))
                    if pending[1] is not None
                    else high,
                    "pending": pending[0],
                    "failed": pending[2] or 0,
                }
            return (
                gen,
                version,
                json.loads(settings[0]) if settings else None,
                {
                    "source_revision": int(revision[0]) if revision else 0,
                    "projection_jobs_pending": jobs[0],
                    "earliest_pending_source": jobs[1],
                    "embedding_version": version,
                    "chunk_version": chunk[0] if chunk else CHUNK_VERSION,
                    "by_source": by_source,
                },
            )

    async def search(
        self,
        sid,
        *,
        query="",
        mode="hybrid",
        sources=None,
        author_ids=None,
        subject_ids=None,
        sender_names=None,
        since=None,
        until=None,
        roles=None,
        exact_phrase="",
        match_mode="any",
        sort="relevance",
        limit=8,
        cursor=None,
        group_by=None,
        exclude_message_ids=None,
        **unused,
    ):
        started = time.monotonic()
        metrics = {"candidate_counts": {}, "overlap_removed": 0}
        if not sid or sid == "_unknown_":
            return failure("missing_session", invalid=True)
        try:
            if mode not in {"hybrid", "keyword", "semantic", "timeline", "aggregate"}:
                raise ValueError("invalid_mode")
            if not isinstance(query, str) or not isinstance(exact_phrase, str) or len(exact_phrase) > 1000:
                raise ValueError("query_and_exact_phrase_must_be_strings")
            if sort not in {"relevance", "oldest", "newest"} or match_mode not in {"any", "all"}:
                raise ValueError("invalid_sort_or_match_mode")
            for values in (sources, author_ids, subject_ids, sender_names, roles):
                if values is not None and (not isinstance(values, list) or len(values) > 10):
                    raise ValueError("filters_must_be_arrays_with_at_most_10_items")
            if any(not isinstance(v, str) or not v for values in (sources, sender_names, roles) for v in values or []):
                raise ValueError("filter_items_must_be_nonempty_strings")
            if any(
                type(v) not in (str, int) or not str(v).isdigit() or int(v) <= 0
                for values in (author_ids, subject_ids)
                for v in values or []
            ):
                raise ValueError("invalid_person_id")
            if type(limit) is not int or not 1 <= limit <= 30 or len(query) > 1000:
                raise ValueError("invalid_query_or_limit")
            requested_times = [since, until]
            since, until = parse_time(since), parse_time(until)
            cursor_state = {}
            if cursor:
                cursor_state = cursor_data(cursor).get("state") or {}
                if cursor_state.get("times"):
                    since, until = cursor_state["times"]
            if since and until and since > until:
                raise ValueError("invalid_time_range")
            sources = sources or ["messages", "summaries", "important_info"]
            if not set(sources) <= {*SOURCE_TABLES, "generated_media"}:
                raise ValueError("invalid_sources")
            if roles and not set(roles) <= {"user", "assistant", "system"}:
                raise ValueError("invalid_roles")
            author_ids = [str(int(v)) for v in author_ids or []]
            subject_ids = [str(int(v)) for v in subject_ids or []]
            if any(len(v) > 50 for v in sender_names or []):
                raise ValueError("invalid_name")
            gen, version, settings, coverage = await asyncio.to_thread(self._snapshot, sid)
            filters = {
                "sources": sources,
                "author_ids": author_ids,
                "subject_ids": subject_ids,
                "sender_names": sender_names or [],
                "since": since,
                "until": until,
                "roles": roles or [],
                "exact_phrase": exact_phrase,
                "match_mode": match_mode,
                "sort": sort,
            }
            filters["chunk_version"] = coverage["chunk_version"]
            filters["exclude_message_ids"] = sorted(set(int(i) for i in (exclude_message_ids or []) if type(i) is int and i > 0))[:30]
            metrics["snapshot_ms"] = round((time.monotonic() - started) * 1000)
            snapshot = digest(encode(coverage))
            if cursor and cursor_state.get("snapshot") != snapshot:
                raise ValueError("cursor_expired_retry_search")
            key = digest(encode([sid, gen, version, query, mode, filters, group_by, requested_times, snapshot]))
            offset = cursor_offset(cursor, key)
            if mode == "aggregate":
                return await asyncio.to_thread(self._aggregate, sid, gen, query, filters, group_by)
            warnings, vector = [], None
            embedding_started = time.monotonic()
            if mode in {"hybrid", "semantic"} and query and not exact_phrase:
                if settings and not settings["model"].startswith("local:"):
                    # Selecting local inference must not incur remote query costs during a rebuild.
                    warnings.append("local_embedding_rebuild_pending")
                elif settings:
                    try:
                        vector = (await embed([query], settings, query=True))[0]
                    except Exception as error:  # noqa: BLE001 -- any provider failure must retain lexical results.
                        warnings.append("semantic_unavailable:" + type(error).__name__)
                else:
                    warnings.append("semantic_not_initialized")
            metrics["embedding_ms"] = round((time.monotonic() - embedding_started) * 1000)
            retrieval_started = time.monotonic()
            items, truncated, backend_warnings = await asyncio.to_thread(
                self._retrieve, sid, gen, version, query, mode, filters, vector, offset, limit, metrics
            )
            metrics["retrieval_ms"] = round((time.monotonic() - retrieval_started) * 1000)
            metrics["total_ms"] = round((time.monotonic() - started) * 1000)
            warnings.extend(backend_warnings)
            candidate_key = metrics.pop("_candidate_key", None)
            # Even unchanged source/index counts can yield different ranks after
            # a backend recovers. Never apply an old offset to a different pool.
            if cursor and cursor_state.get("candidates") != candidate_key:
                raise ValueError("cursor_expired_retry_search")
            # Source changes during embedding/retrieval invalidate the snapshot.
            with self._connect() as db:
                if generation(db, sid) != gen:
                    return failure("scope_changed", retryable=True)
                revision = db.execute(
                    "SELECT value FROM kv_meta WHERE key=?", ("memory_revision:" + sid,)
                ).fetchone()
                if (int(revision[0]) if revision else 0) != coverage["source_revision"]:
                    return failure("scope_changed_retry_search", retryable=True)
            if coverage["projection_jobs_pending"]:
                warnings.append("index_incomplete_original_search_included")
            effective = (
                mode
                if (
                    vector is not None
                    and not any(w in warnings for w in ("vector_backend_unavailable", "vector_query_budget_exhausted"))
                )
                or mode not in {"hybrid", "semantic"}
                else "keyword"
            )
            return result(
                "partial" if warnings else ("ok" if items else "empty"),
                items,
                effective_scope={"session_id": sid, "generation": gen},
                effective_filters=filters,
                requested_mode=mode,
                effective_mode=effective,
                coverage=coverage,
                metrics=metrics,
                evidence_hint="相关片段只是线索，不证明人物与事件的关系。先说明已知与缺口；必要时按账号/时间用 query_history 精查，再 read_history 核对前后文。候选达到上限请收窄条件；无新增证据就停止。related_source_ids 是合并窗口的额外线索，需要回读，不能视作已核验。",
                warnings=warnings,
                truncated=truncated,
                next_cursor=cursor_encode(key, offset + limit, state={
                    "times": [since, until], "snapshot": snapshot, "candidates": candidate_key,
                })
                if truncated and len(items) == limit and offset + limit < 300
                else None,
                hits=items,
                count=len(items),
            )
        except (ValueError, TypeError, KeyError) as error:
            return failure(str(error), invalid=True)
        except (sqlite3.Error, OSError) as error:
            return failure(
                "query_budget_exhausted_refine_filters" if "interrupted" in str(error) else type(error).__name__,
                retryable="interrupted" not in str(error),
            )

    def _canonical_where(self, sid, kind, query, filters):
        clauses, params = ["session_id=?"], [sid]
        from .memory_changes import active_sql
        clauses.append(active_sql(kind))
        column = "timestamp" if kind == "messages" else "json_extract(provenance_json,'$.event_start')"
        if kind == "messages":
            clauses.append("COALESCE(is_thinking,0)=0")
            excluded = filters.get("exclude_message_ids", [])
            if excluded:
                clauses.append("id NOT IN (" + ",".join("?" for _ in excluded) + ")")
                params.extend(excluded)
        for bound, operator in (("since", ">="), ("until", "<=")):
            if filters[bound]:
                clauses.append(f"{column}{operator}?")
                params.append(filters[bound])
        if filters["author_ids"]:
            if kind != "messages":
                clauses.append("0")
            else:
                clauses.append("CAST(sender_id AS TEXT) IN (" + ",".join("?" for _ in filters["author_ids"]) + ")")
                params.extend(filters["author_ids"])
        if filters["subject_ids"]:
            if kind != "important_info":
                clauses.append("0")
            else:
                clauses.append("CAST(user_id AS TEXT) IN (" + ",".join("?" for _ in filters["subject_ids"]) + ")")
                params.extend(filters["subject_ids"])
        if filters["roles"]:
            if kind != "messages":
                clauses.append("0")
            else:
                clauses.append("role IN (" + ",".join("?" for _ in filters["roles"]) + ")")
                params.extend(filters["roles"])
        if filters["sender_names"]:
            if kind != "messages":
                clauses.append("0")
            else:
                clauses.append("(" + " OR ".join("instr(name,?)>0" for _ in filters["sender_names"]) + ")")
                params.extend(filters["sender_names"])
        terms = [filters["exact_phrase"]] if filters["exact_phrase"] else query.split()
        if terms:
            op = " AND " if filters["match_mode"] == "all" or filters["exact_phrase"] else " OR "
            clauses.append("(" + op.join("instr(lower(content),?)>0" for _ in terms[:20]) + ")")
            params.extend(t.lower() for t in terms[:20])
        if kind == "messages" and "generated_media" in filters["sources"] and "messages" not in filters["sources"]:
            clauses.append(
                "role='assistant' AND (content LIKE '%【生成图片%' OR content LIKE '%【语音%' OR content LIKE '%【表情包%')"
            )
        return " AND ".join(clauses), params

    def _original_item(self, row, kind, gen, *, full=False):
        data = dict(row)
        metadata = json.loads(data.get("provenance_json") or "{}")
        content = data["content"]
        return {
            "id": f"h:{gen}:{kind}:{data['id']}",
            "kind": kind,
            "source": kind,
            "excerpt": content if full else content[:500],
            "content": content if full else content[:500],
            "source_ids": [f"h:{gen}:{kind}:{data['id']}"],
            "sender_id": data.get("sender_id"),
            "model_spec": data.get("model_spec"),
            "authors": [str(data["sender_id"])] if data.get("sender_id") is not None else metadata.get("authors", []),
            "subjects": [str(data["user_id"])] if kind == "important_info" else [],
            "name": data.get("name"),
            "role": data.get("role"),
            "event_time": data.get("timestamp") or metadata.get("event_start"),
            "recorded_at": data.get("created_at") or data.get("timestamp"),
            "provenance": "original" if kind == "messages" else metadata.get("provenance", "legacy_unlinked"),
            "evidence_kind": ("user_message" if data.get("role") == "user" else "assistant_or_system_message")
            if kind == "messages" else "derived_assertion",
            "assertion": metadata.get("assertion"),
            "read_state": "original" if full else "excerpt",
            "content_hash": digest(content),
            "correction": metadata.get("correction"),
            "truncated": not full and len(content) > 500,
        }

    def _projection_where(self, sid, gen, filters, query=""):
        """Apply the same canonical scope before either backend chooses top K."""
        kinds = set(filters["sources"])
        if "generated_media" in kinds:
            kinds.add("messages")
        branches = []
        params = [sid, gen, filters.get("chunk_version", CHUNK_VERSION)]
        for kind in sorted(kinds & SOURCE_TABLES.keys()):
            where, args = self._canonical_where(sid, kind, query, filters)
            where = where.replace(kind + ".", "s.")
            where = re.sub(
                r"(?<![.\w])(id|session_id|timestamp|provenance_json|is_thinking|sender_id|user_id|role|name|content)\b",
                lambda m: "s." + m[0],
                where,
            )
            branches.append(f"""(m.kind=? AND EXISTS(SELECT 1 FROM memory_links l
                CROSS JOIN {SOURCE_TABLES[kind]} s ON s.id=l.source_id
                WHERE l.item_id=m.id AND l.source_kind=? AND l.relation='derived_from' AND {where}))""")
            params.extend([kind, kind, *args])
        return (
            "m.session_id=? AND m.generation=? AND m.chunk_version=? AND (" + " OR ".join(branches) + ")"
            " AND NOT EXISTS(SELECT 1 FROM memory_index_jobs j WHERE j.session_id=m.session_id AND j.kind=m.kind"
            " AND j.source_id=m.source_id AND j.generation=m.generation AND j.priority=0)"
        ), params

    def _retrieve(self, sid, gen, version, query, mode, filters, vector, offset, limit, metrics=None):
        metrics = metrics if metrics is not None else {"candidate_counts": {}, "overlap_removed": 0}
        # Bound each candidate pool; the caller rejects cursors when it changes.
        deadline = time.monotonic() + 8
        candidate_limit = 300
        if offset >= candidate_limit:
            raise ValueError("candidate_budget_exhausted_refine_query")
        ranked, warnings = [], []
        capped = False
        with self._connect(deadline=deadline) as db:
            if generation(db, sid) != gen:
                return [], False, ["scope_changed"]
            allowed = set(filters["sources"])
            if "generated_media" in allowed:
                allowed.add("messages")
            # A requested semantic search ranks only vector results when available.
            # The lexical fallback still works if embedding failed, with its warning.
            lexical = mode != "semantic" or vector is None
            # Each source has an independent rank list. Concatenating tables
            # before RRF gives every old fact priority over every user message.
            for kind in (k for k in ("messages", "summaries", "important_info") if k in allowed) if lexical else []:
                where, args = self._canonical_where(sid, kind, query, filters)
                direction = "ASC" if filters["sort"] == "oldest" else "DESC"
                rows = db.execute(
                    f"SELECT * FROM {SOURCE_TABLES[kind]} WHERE {where} ORDER BY id {direction} LIMIT ?",
                    (*args, candidate_limit + 1),
                ).fetchall()
                capped |= len(rows) > candidate_limit
                metrics["candidate_counts"]["original_" + kind] = len(rows)
                ranked.append([
                    self._original_item(row, kind, gen)
                    for row in rows[:candidate_limit]
                    if self._support_live(db, row, kind, sid)
                ])
            if lexical and query and mode != "timeline" and not filters["exact_phrase"]:
                search_tokens = [word for word in tokens(query) if word not in _QUERY_STOP_WORDS][:24]
                fts_query = (" AND " if filters["match_mode"] == "all" else " OR ").join(
                    '"' + t.replace('"', '""') + '"' for t in search_tokens
                )
                if fts_query:
                    try:
                        scoped, args = self._projection_where(sid, gen, filters, " ".join(search_tokens))
                        rows = db.execute(
                            f"""SELECT m.* FROM memory_fts f JOIN memory_items m ON m.id=f.rowid
                            WHERE memory_fts MATCH ? AND {scoped}
                            ORDER BY bm25(memory_fts),m.id LIMIT ?""",
                            (fts_query, *args, candidate_limit + 1),
                        ).fetchall()
                        capped |= len(rows) > candidate_limit
                        metrics["candidate_counts"]["fts"] = len(rows)
                        ranked.append(self._projection_items(db, rows[:candidate_limit], sid, gen, filters))
                    except sqlite3.OperationalError as error:
                        warnings.append(
                            "fts_query_budget_exhausted" if "interrupted" in str(error) else "fts_backend_unavailable"
                        )
        if vector is not None and re.fullmatch(r"[a-f0-9]{16}", version):
            try:
                with self._connect(vectors=True, deadline=deadline) as db:
                    scoped, args = self._projection_where(sid, gen, filters)
                    # vec0 evaluates the eligible ID set before its batched
                    # cosine scan. Point-reading every embedding is much slower.
                    matches = db.execute(
                        f"""SELECT item_id,content_hash,distance FROM memory_vec_{version}
                        WHERE embedding MATCH ? AND k=? AND session_id=?
                        AND item_id IN (SELECT m.id FROM memory_items m WHERE {scoped})
                        ORDER BY distance""",
                        (vector, candidate_limit + 1, sid, *args),
                    ).fetchall()
                    capped |= len(matches) > candidate_limit
                    metrics["candidate_counts"]["vector"] = len(matches)
                    rows = []
                    for match in matches:
                        row = db.execute(
                            "SELECT * FROM memory_items WHERE id=? AND content_hash=? AND session_id=? AND generation=?",
                            (match["item_id"], match["content_hash"], sid, gen),
                        ).fetchone()
                        if row:
                            rows.append(row)
                    if len(rows) != len(matches):
                        warnings.append("stale_vectors_skipped")
                    ranked.append(self._projection_items(db, rows[:candidate_limit], sid, gen, filters))
            except (ImportError, sqlite3.Error, OSError, RuntimeError) as error:
                warnings.append(
                    "vector_query_budget_exhausted" if "interrupted" in str(error) else "vector_backend_unavailable"
                )
                if not lexical:
                    # A failed semantic-only query has no lexical results yet.
                    # Give the fallback its own bounded deadline: the vector
                    # query may already have consumed the original SQL budget.
                    items, truncated, fallback_warnings = self._retrieve(
                        sid, gen, version, query, mode, filters, None, offset, limit, metrics
                    )
                    return items, truncated, warnings + fallback_warnings
        scores, by_id = {}, {}
        for channel in ranked:
            for rank, item in enumerate(channel):
                by_id[item["id"]] = item
                scores[item["id"]] = scores.get(item["id"], 0) + 1 / (60 + rank + 1)
        items = [by_id[k] for k in sorted(scores, key=scores.get, reverse=True)]
        if mode == "timeline" or filters["sort"] in {"oldest", "newest"}:
            # Unknown event times must not sort before the oldest known event.
            known = [item for item in items if item.get("event_time")]
            unknown = [item for item in items if not item.get("event_time")]
            known.sort(key=lambda item: item["event_time"], reverse=filters["sort"] != "oldest")
            items = known + unknown
        seen, unique = set(), []
        for item in items:
            refs = set(item["source_ids"])
            if refs and refs <= seen:
                metrics["overlap_removed"] += 1
                continue
            overlap = next((prior for prior in unique if item.get("kind") == prior.get("kind") == "messages"
                            and len(refs) > 1 and len(prior["source_ids"]) > 1
                            and len(refs & set(prior["source_ids"])) / max(len(refs), len(prior["source_ids"])) >= 0.75), None)
            if overlap is not None:
                overlap["related_source_ids"] = sorted(set(overlap.get("related_source_ids", [])) | (refs - set(overlap["source_ids"])))
                metrics["overlap_removed"] += 1
                seen.update(refs)
                continue
            seen.update(refs)
            unique.append(item)
        metrics["_candidate_key"] = digest(encode([
            [item["id"], item["source_ids"], item.get("event_time")] for item in unique
        ]))
        more = len(unique) > offset + limit and offset + limit < candidate_limit
        if capped or len(unique) > candidate_limit:
            warnings.append("candidate_budget_limited_refine_query_for_exhaustive_results")
        return unique[offset : offset + limit], more or capped, warnings

    def _projection_items(self, db, rows, sid, gen, filters):
        items = []
        for row in rows:
            row = dict(row)
            if row["kind"] not in filters["sources"] and not (
                row["kind"] == "messages" and "generated_media" in filters["sources"]
            ):
                continue
            meta = json.loads(row["metadata_json"])
            if filters["subject_ids"] and not set(filters["subject_ids"]) & set(meta.get("subjects", [])):
                continue
            if (
                row["kind"] != "messages"
                and filters["since"]
                and (not row["event_start"] or row["event_start"] < filters["since"])
            ):
                continue
            if (
                row["kind"] != "messages"
                and filters["until"]
                and (not row["event_start"] or row["event_start"] > filters["until"])
            ):
                continue
            links = db.execute(
                "SELECT source_kind,source_id FROM memory_links WHERE item_id=? AND relation='derived_from'",
                (row["id"],),
            ).fetchall()
            originals = []
            stale = False
            for kind, source_id in links:
                if kind not in SOURCE_TABLES:
                    continue
                source = db.execute(
                    f"SELECT * FROM {SOURCE_TABLES[kind]} WHERE id=? AND session_id=?", (source_id, sid)
                ).fetchone()
                if not source:
                    stale = True
                    break
                if not self._support_live(db, source, kind, sid):
                    stale = True
                    break
                if kind == "messages" and source["is_thinking"]:
                    stale = True
                    break
                source_hashes = meta.get("source_hashes", {})
                if (
                    kind == "messages"
                    and str(source_id) in source_hashes
                    and digest(source["content"]) != source_hashes[str(source_id)]
                ):
                    stale = True
                    break
                if kind == "messages":
                    if source_id in filters.get("exclude_message_ids", []):
                        continue
                    if filters["since"] and source["timestamp"] < filters["since"]:
                        continue
                    if filters["until"] and source["timestamp"] > filters["until"]:
                        continue
                    if (
                        "generated_media" in filters["sources"]
                        and "messages" not in filters["sources"]
                        and not (
                            source["role"] == "assistant"
                            and any(marker in source["content"] for marker in ("【生成图片", "【语音", "【表情包"))
                        )
                    ):
                        continue
                    if filters["author_ids"] and str(source["sender_id"]) not in filters["author_ids"]:
                        continue
                    if filters["roles"] and source["role"] not in filters["roles"]:
                        continue
                    if filters["sender_names"] and not any(
                        n in (source["name"] or "") for n in filters["sender_names"]
                    ):
                        continue
                    originals.append(self._original_item(source, kind, gen))
                elif not filters["author_ids"] and not filters["roles"] and not filters["sender_names"]:
                    originals.append(self._original_item(source, kind, gen))
            if stale or not originals:
                continue
            # Filtered author results expose only the selected speakers' text.
            content = (
                "\n".join(o["excerpt"] for o in originals)
                if any(filters[k] for k in ("author_ids", "roles", "sender_names", "since", "until"))
                or filters.get("exclude_message_ids")
                or ("generated_media" in filters["sources"] and "messages" not in filters["sources"])
                else row["content"]
            )
            items.append(
                {
                    "id": f"i:{gen}:{row['id']}",
                    "kind": row["kind"],
                    "source": row["kind"],
                    "excerpt": content[:500],
                    "content": content[:500],
                    "source_ids": [o["id"] for o in originals],
                    "authors": list(dict.fromkeys(a for o in originals for a in o["authors"])),
                    "subjects": meta.get("subjects", []),
                    "event_time": row["event_start"],
                    "recorded_at": row["recorded_at"],
                    "provenance": meta.get("provenance"),
                    "evidence_kind": "message_window" if row["kind"] == "messages" else "derived_assertion",
                    "assertion": meta.get("assertion"),
                    "read_state": "excerpt",
                    "truncated": len(content) > 500,
                }
            )
        return items

    def _aggregate(self, sid, gen, query, filters, group_by):
        if filters["sources"] != ["messages"]:
            raise ValueError("aggregate_requires_messages_source")
        if group_by not in {None, "author", "day", "role"}:
            raise ValueError("invalid_group_by")
        where, args = self._canonical_where(sid, "messages", query, filters)
        field = {"author": "sender_id", "day": "substr(timestamp,1,10)", "role": "role"}.get(group_by)
        with self._connect() as db:
            if generation(db, sid) != gen:
                return failure("scope_changed", retryable=True)
            if field:
                values = [
                    {"key": r[0], "count": r[1]}
                    for r in db.execute(
                        f"SELECT {field},count(*) FROM messages_detail WHERE {where} GROUP BY {field}", args
                    )
                ]
            else:
                values = db.execute(f"SELECT count(*) FROM messages_detail WHERE {where}", args).fetchone()[0]
        return result(
            aggregation={
                "unit": "messages",
                "group_by": group_by,
                "values": values,
                "query_semantics": "exact substring tokens; not semantic topic counts",
            },
            effective_scope={"session_id": sid, "generation": gen},
            effective_filters=filters,
        )

    async def read(self, sid, *, ids=None, anchor_id=None, before=3, after=3, cursor=None, limit=20):
        try:
            if any(type(value) is not int for value in (before, after, limit)):
                raise ValueError("read_bounds_must_be_integers")
            if ids is not None and (not isinstance(ids, list) or any(not isinstance(ref, str) for ref in ids)):
                raise ValueError("ids_must_be_an_array_of_strings")
            if anchor_id is not None and not isinstance(anchor_id, str):
                raise ValueError("anchor_id_must_be_string")
            if not sid or sid == "_unknown_" or not 0 <= before <= 20 or not 0 <= after <= 20 or not 1 <= limit <= 30:
                raise ValueError("invalid_read_scope_or_bounds")
            return await asyncio.to_thread(self._read, sid, ids or [], anchor_id, before, after, cursor, limit)
        except (ValueError, TypeError, KeyError) as error:
            return failure(str(error), invalid=True)
        except (sqlite3.Error, OSError) as error:
            return failure(
                "query_budget_exhausted_refine_filters" if "interrupted" in str(error) else type(error).__name__,
                retryable="interrupted" not in str(error),
            )

    def _read(self, sid, ids, anchor, before, after, cursor, limit):
        with self._connect() as db:
            gen = generation(db, sid)
            key = digest(encode([sid, gen, ids, anchor, before, after]))
            offset = cursor_offset(cursor, key)
            requested = list(ids) + ([anchor] if anchor else [])
            if not requested or len(requested) > 30:
                raise ValueError("source_ids_required_max_30")
            refs = []
            for ref in requested:
                parts = str(ref).split(":")
                if len(parts) < 3 or parts[0] not in {"h", "i"} or int(parts[1]) != gen:
                    raise ValueError("source_expired_or_invalid")
                if parts[0] == "i":
                    if len(parts) != 3:
                        raise ValueError("invalid_source_id")
                    if not db.execute(
                        "SELECT 1 FROM memory_items WHERE id=? AND session_id=? AND generation=?",
                        (int(parts[2]), sid, gen),
                    ).fetchone():
                        raise ValueError("source_not_available_in_scope")
                    refs.extend(
                        (r[0], r[1])
                        for r in db.execute(
                            "SELECT source_kind,source_id FROM memory_links WHERE item_id=? AND relation='derived_from'",
                            (int(parts[2]),),
                        )
                    )
                else:
                    if len(parts) != 4 or parts[2] not in SOURCE_TABLES:
                        raise ValueError("invalid_source_id")
                    refs.append((parts[2], int(parts[3])))
            if anchor:
                if len(refs) != 1 or refs[0][0] != "messages":
                    raise ValueError("anchor_must_be_original_message")
                mid = refs[0][1]
                if not db.execute(
                    "SELECT 1 FROM messages_detail WHERE id=? AND session_id=? AND COALESCE(is_thinking,0)=0",
                    (mid, sid),
                ).fetchone():
                    raise ValueError("source_not_available_in_scope")
                prev = [
                    r[0]
                    for r in db.execute(
                        "SELECT id FROM messages_detail WHERE session_id=? AND id<? AND COALESCE(is_thinking,0)=0 ORDER BY id DESC LIMIT ?",
                        (sid, mid, before),
                    )
                ]
                nxt = [
                    r[0]
                    for r in db.execute(
                        "SELECT id FROM messages_detail WHERE session_id=? AND id>? AND COALESCE(is_thinking,0)=0 ORDER BY id LIMIT ?",
                        (sid, mid, after),
                    )
                ]
                refs = [("messages", v) for v in [*reversed(prev), mid, *nxt]]
            refs = list(dict.fromkeys(refs))
            items = []
            state = (cursor_data(cursor).get("state") or {}) if cursor else {}
            char_offset = int(state.get("char_offset", 0))
            used = 0
            next_offset = offset
            next_char = 0
            for index, (kind, source_id) in enumerate(refs[offset : offset + limit], start=offset):
                row = db.execute(
                    f"SELECT * FROM {SOURCE_TABLES[kind]} WHERE id=? AND session_id=?", (source_id, sid)
                ).fetchone()
                if not row or (kind == "messages" and row["is_thinking"]):
                    raise ValueError("source_deleted_or_outside_scope")
                if not self._support_live(db, row, kind, sid):
                    raise ValueError("supporting_source_deleted_or_changed")
                item = self._original_item(row, kind, gen, full=True)
                start_char = char_offset if index == offset else 0
                if not 0 <= start_char <= len(item["content"]):
                    raise ValueError("invalid_source_offset")
                excerpt = item["content"][start_char : start_char + max(0, 12000 - used)]
                used += len(excerpt)
                item.update(
                    content=excerpt,
                    excerpt=excerpt,
                    location={"start": start_char, "end": start_char + len(excerpt)},
                    truncated=start_char + len(excerpt) < len(row["content"]),
                )
                if kind != "messages":
                    meta = json.loads(row["provenance_json"] or "{}")
                    item["supporting_source_ids"] = [f"h:{gen}:messages:{i}" for i in meta.get("source_ids", [])]
                items.append(item)
                next_offset = index if item["truncated"] else index + 1
                next_char = start_char + len(excerpt) if item["truncated"] else 0
                if used >= 12000:
                    break
            more = next_offset < len(refs)
            return result(
                "ok" if items else "empty",
                items,
                effective_scope={"session_id": sid, "generation": gen},
                truncated=more,
                next_cursor=cursor_encode(key, next_offset, state={"char_offset": next_char}) if more else None,
            )
