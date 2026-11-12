"""Scoped SQL history queries. Connection lifetime belongs to MemoryManager."""
from typing import Awaitable, Callable, Dict, List, Optional
import aiosqlite


class MemoryQueries:
    def __init__(self, get_db: Callable[[], Awaitable[aiosqlite.Connection]]):
        self._get_db = get_db

    async def search_messages_detail(
        self,
        session_id: str,
        keyword: Optional[str] = None,
        sender_name: Optional[str] = None,
        sender_ids: Optional[List[int]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        role: Optional[str] = None,
        roles: Optional[List[str]] = None,
        keyword_mode: str = "all",
        limit: int = 20,
    ) -> List[Dict]:
        """按字段检索 messages_detail。用于 recall_history 工具。

        keyword: 单关键词或多个空格分隔
        keyword_mode: all=全部命中，any=任一命中
        sender_name: 模糊匹配 name 字段
        since/until: ISO 格式时间戳字符串
        role: user/assistant/system/all
        返回 [{id, role, name, sender_id, content, timestamp}, ...]，按时间升序
        """
        sql = [
            "SELECT id, role, name, sender_id, content, timestamp, model_spec "
            "FROM messages_detail WHERE session_id = ?"
        ]
        params: List = [session_id]

        # 排除 thinking 消息（自我思考），通常用户不需要召回
        sql.append("AND is_thinking = 0")

        if keyword:
            keywords = [k for k in keyword.split() if k.strip()][:5]
            if keywords:
                if (keyword_mode or "").lower() == "any" and len(keywords) > 1:
                    sql.append("AND (" + " OR ".join(["content LIKE ?" for _ in keywords]) + ")")
                    params.extend([f"%{kw}%" for kw in keywords])
                else:
                    for kw in keywords:
                        sql.append("AND content LIKE ?")
                        params.append(f"%{kw}%")
        if sender_name:
            sql.append("AND name LIKE ?")
            params.append(f"%{sender_name}%")
        normalized_sender_ids = []
        for raw_sender_id in sender_ids or []:
            try:
                normalized_sender_ids.append(int(raw_sender_id))
            except (TypeError, ValueError):
                continue
        if normalized_sender_ids:
            placeholders = ",".join("?" for _ in normalized_sender_ids)
            sql.append(f"AND sender_id IN ({placeholders})")
            params.extend(normalized_sender_ids)

        normalized_roles = []
        for raw_role in roles or ([role] if role else []):
            normalized_role = str(raw_role or "").strip().lower()
            if normalized_role in ("bot", "assistant"):
                normalized_role = "assistant"
            if normalized_role in ("user", "assistant", "system") and normalized_role not in normalized_roles:
                normalized_roles.append(normalized_role)
        if normalized_roles:
            placeholders = ",".join("?" for _ in normalized_roles)
            sql.append(f"AND role IN ({placeholders})")
            params.extend(normalized_roles)
        if since:
            sql.append("AND timestamp >= ?")
            params.append(since)
        if until:
            sql.append("AND timestamp <= ?")
            params.append(until)

        # 倒序取最近 limit 条，再正序返回（更符合阅读习惯）
        sql.append("ORDER BY timestamp DESC LIMIT ?")
        params.append(max(1, min(int(limit), 100)))

        db = await self._get_db()
        async with db.execute(" ".join(sql), tuple(params)) as cursor:
            rows = await cursor.fetchall()
        results = [
            {
                "id": r[0],
                "role": r[1],
                "name": r[2],
                "sender_id": r[3],
                "content": r[4],
                "timestamp": r[5],
                "model_spec": r[6],
            }
            for r in rows
        ]
        results.reverse()
        return results


    async def recall_memory(
        self,
        *,
        session_id: str,
        query: str = "",
        sources: Optional[List[str]] = None,
        sender_ids: Optional[List[int]] = None,
        sender_names: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        roles: Optional[List[str]] = None,
        match_mode: str = "all",
        sort: str = "relevance",
        limit: int = 20,
        context_before: int = 0,
        context_after: int = 0,
    ) -> dict:
        """Search all memory layers in the current session and return structured hits."""
        allowed_sources = {
            "messages",
            "summaries",
            "important_info",
            "generated_media",
        }
        normalized_sources = [
            source
            for source in (sources or ["messages", "summaries", "important_info"])
            if source in allowed_sources
        ]
        if not normalized_sources:
            normalized_sources = ["messages"]

        safe_limit = max(1, min(int(limit), 30))
        safe_before = max(0, min(int(context_before), 5))
        safe_after = max(0, min(int(context_after), 5))
        normalized_match_mode = "any" if str(match_mode).lower() == "any" else "all"
        normalized_sort = str(sort or "relevance").lower()
        if normalized_sort not in {"relevance", "newest", "oldest"}:
            normalized_sort = "relevance"

        keywords = [part.strip() for part in str(query or "").split() if part.strip()][:8]
        normalized_sender_ids: List[int] = []
        for raw_sender_id in sender_ids or []:
            try:
                sender_id = int(raw_sender_id)
            except (TypeError, ValueError):
                continue
            if sender_id not in normalized_sender_ids:
                normalized_sender_ids.append(sender_id)
        normalized_sender_names = [
            str(name).strip()
            for name in sender_names or []
            if str(name).strip()
        ][:8]
        normalized_roles = []
        for raw_role in roles or []:
            normalized_role = str(raw_role or "").strip().lower()
            if normalized_role == "bot":
                normalized_role = "assistant"
            if normalized_role in {"user", "assistant", "system"}:
                normalized_roles.append(normalized_role)

        def append_keyword_filters(sql: List[str], params: List, column: str = "content") -> None:
            if not keywords:
                return
            if normalized_match_mode == "any" and len(keywords) > 1:
                sql.append(
                    "AND (" + " OR ".join(f"{column} LIKE ?" for _ in keywords) + ")"
                )
                params.extend(f"%{keyword}%" for keyword in keywords)
            else:
                for keyword in keywords:
                    sql.append(f"AND {column} LIKE ?")
                    params.append(f"%{keyword}%")

        def relevance_score(content: str) -> int:
            lowered = str(content or "").casefold()
            return sum(lowered.count(keyword.casefold()) for keyword in keywords)

        db = await self._get_db()
        hits: List[dict] = []
        wants_messages = "messages" in normalized_sources
        wants_media = "generated_media" in normalized_sources
        media_markers = ("【生成图片", "【生成语音", "【我发送了一张表情包】")

        if wants_messages or wants_media:
            sql = [
                "SELECT id, role, name, sender_id, content, timestamp "
                "FROM messages_detail WHERE session_id = ? AND is_thinking = 0"
            ]
            params: List = [session_id]
            append_keyword_filters(sql, params)
            if normalized_sender_ids:
                placeholders = ",".join("?" for _ in normalized_sender_ids)
                sql.append(f"AND sender_id IN ({placeholders})")
                params.extend(normalized_sender_ids)
            if normalized_sender_names:
                sql.append(
                    "AND (" + " OR ".join("name LIKE ?" for _ in normalized_sender_names) + ")"
                )
                params.extend(f"%{name}%" for name in normalized_sender_names)
            if normalized_roles:
                placeholders = ",".join("?" for _ in normalized_roles)
                sql.append(f"AND role IN ({placeholders})")
                params.extend(normalized_roles)
            if since:
                sql.append("AND timestamp >= ?")
                params.append(since)
            if until:
                sql.append("AND timestamp <= ?")
                params.append(until)
            if wants_media and not wants_messages:
                sql.append(
                    "AND (" + " OR ".join("content LIKE ?" for _ in media_markers) + ")"
                )
                params.extend(f"{marker}%" for marker in media_markers)
            elif wants_messages and not wants_media:
                sql.append(
                    "AND NOT (" + " OR ".join("content LIKE ?" for _ in media_markers) + ")"
                )
                params.extend(f"{marker}%" for marker in media_markers)
            order = "ASC" if normalized_sort == "oldest" else "DESC"
            sql.append(f"ORDER BY timestamp {order}, id {order} LIMIT ?")
            params.append(max(safe_limit * 5, 50) if normalized_sort == "relevance" else safe_limit)
            async with db.execute(" ".join(sql), tuple(params)) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                content = str(row[4] or "")
                source = (
                    "generated_media"
                    if content.startswith(media_markers)
                    else "messages"
                )
                hits.append({
                    "id": f"message:{row[0]}",
                    "message_id": row[0],
                    "source": source,
                    "timestamp": row[5],
                    "role": row[1],
                    "sender_id": row[3],
                    "sender_name": row[2],
                    "content": content,
                    "score": relevance_score(content),
                })

        if "summaries" in normalized_sources:
            sql = [
                "SELECT id, content, time_range, created_at "
                "FROM summaries WHERE session_id = ?"
            ]
            params = [session_id]
            append_keyword_filters(sql, params)
            if since:
                sql.append("AND created_at >= ?")
                params.append(since)
            if until:
                sql.append("AND created_at <= ?")
                params.append(until)
            order = "ASC" if normalized_sort == "oldest" else "DESC"
            sql.append(f"ORDER BY created_at {order}, id {order} LIMIT ?")
            params.append(max(safe_limit * 3, 30) if normalized_sort == "relevance" else safe_limit)
            async with db.execute(" ".join(sql), tuple(params)) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                hits.append({
                    "id": f"summary:{row[0]}",
                    "source": "summaries",
                    "timestamp": row[3],
                    "time_range": row[2],
                    "role": "system",
                    "sender_id": None,
                    "sender_name": None,
                    "content": row[1],
                    "score": relevance_score(row[1]),
                })

        if "important_info" in normalized_sources:
            sql = [
                "SELECT id, user_id, info_type, content, created_at "
                "FROM important_info WHERE session_id = ?"
            ]
            params = [session_id]
            append_keyword_filters(sql, params)
            if normalized_sender_ids:
                placeholders = ",".join("?" for _ in normalized_sender_ids)
                sql.append(f"AND user_id IN ({placeholders})")
                params.extend(normalized_sender_ids)
            if since:
                sql.append("AND created_at >= ?")
                params.append(since)
            if until:
                sql.append("AND created_at <= ?")
                params.append(until)
            order = "ASC" if normalized_sort == "oldest" else "DESC"
            sql.append(f"ORDER BY created_at {order}, id {order} LIMIT ?")
            params.append(max(safe_limit * 3, 30) if normalized_sort == "relevance" else safe_limit)
            async with db.execute(" ".join(sql), tuple(params)) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                hits.append({
                    "id": f"important_info:{row[0]}",
                    "source": "important_info",
                    "timestamp": row[4],
                    "info_type": row[2],
                    "role": "system",
                    "sender_id": row[1],
                    "sender_name": None,
                    "content": row[3],
                    "score": relevance_score(row[3]),
                })

        if normalized_sort == "relevance":
            hits.sort(
                key=lambda item: (
                    int(item.get("score") or 0),
                    str(item.get("timestamp") or ""),
                ),
                reverse=True,
            )
        else:
            hits.sort(
                key=lambda item: str(item.get("timestamp") or ""),
                reverse=normalized_sort == "newest",
            )
        hits = hits[:safe_limit]

        if safe_before or safe_after:
            for hit in hits:
                message_id = hit.get("message_id")
                if not message_id:
                    continue
                context_rows = []
                if safe_before:
                    async with db.execute(
                        """
                        SELECT id, role, name, sender_id, content, timestamp
                        FROM messages_detail
                        WHERE session_id = ? AND is_thinking = 0 AND id < ?
                        ORDER BY id DESC LIMIT ?
                        """,
                        (session_id, message_id, safe_before),
                    ) as cursor:
                        before_rows = list(reversed(await cursor.fetchall()))
                    context_rows.extend(before_rows)
                if safe_after:
                    async with db.execute(
                        """
                        SELECT id, role, name, sender_id, content, timestamp
                        FROM messages_detail
                        WHERE session_id = ? AND is_thinking = 0 AND id > ?
                        ORDER BY id ASC LIMIT ?
                        """,
                        (session_id, message_id, safe_after),
                    ) as cursor:
                        context_rows.extend(await cursor.fetchall())
                hit["context"] = [
                    {
                        "id": f"message:{row[0]}",
                        "role": row[1],
                        "sender_name": row[2],
                        "sender_id": row[3],
                        "content": row[4],
                        "timestamp": row[5],
                    }
                    for row in context_rows
                ]

        return {
            "session_id": session_id,
            "query": query,
            "sources": normalized_sources,
            "match_mode": normalized_match_mode,
            "sort": normalized_sort,
            "count": len(hits),
            "hits": hits,
        }
