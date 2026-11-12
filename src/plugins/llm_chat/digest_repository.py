"""Published briefing/news persistence using the shared memory transaction lock."""
import asyncio
import json
from datetime import date, datetime
from typing import Awaitable, Callable, Dict, List, Optional

import aiosqlite

from .logging_setup import memory_logger as _log
from .sqlite_utils import measured_write_lock, SQLITE_WRITE_RETRIES, SQLITE_WRITE_RETRY_BASE_DELAY
from .model_types import Message
from .config import config


class DigestRepository:
    def __init__(self, get_db: Callable[[], Awaitable[aiosqlite.Connection]],
                 write_lock: asyncio.Lock, rollback, is_locked):
        self._get_db = get_db
        self._write_lock = write_lock
        self._rollback_safely = rollback
        self._is_database_locked_error = is_locked

    async def get_digest_messages(self, session_id: str, start: datetime, end: datetime) -> tuple[List[Message], int]:
        """Read original messages for a fixed schedule window, independent of compression."""
        db = await self._get_db()
        where = ("session_id=? AND timestamp>=? AND timestamp<? AND is_thinking=0 "
                 "AND role IN ('user','assistant') AND content NOT LIKE '【每日群聊简报】%' "
                 "AND content NOT LIKE '【每日新闻播报】%'")
        params = (session_id, start.isoformat(), end.isoformat())
        async with db.execute(f"SELECT COUNT(*) FROM messages_detail WHERE {where} AND role='user'", params) as cursor:
            total_users = (await cursor.fetchone())[0]
        # Stream chronologically and allocate one text budget to each hour, so busy evenings
        # cannot completely erase the morning from a daily digest.
        per_hour = max(200, int(config.briefing_context_max_chars) // 24)
        hourly_used: Dict[int, int] = {}
        messages = []
        async with db.execute(
            f"SELECT id,role,name,sender_id,content,timestamp FROM messages_detail WHERE {where} ORDER BY timestamp,id", params
        ) as cursor:
            async for source_id, role, name, sender_id, content, timestamp in cursor:
                hour_slot = int((datetime.fromisoformat(timestamp) - start).total_seconds() // 3600)
                remaining = per_hour - hourly_used.get(hour_slot, 0)
                if remaining <= 0:
                    continue
                text = str(content)[:min(remaining, 1200)]
                hourly_used[hour_slot] = hourly_used.get(hour_slot, 0) + len(text) + 40
                messages.append(Message(role=role,name=name,sender_id=sender_id,content=text,timestamp=timestamp,source_id=source_id))
        return messages, int(total_users)


    async def recent_news_urls(self, group_id: int, since: datetime) -> set[str]:
        db = await self._get_db()
        urls = set()
        async with db.execute(
            "SELECT payload_json FROM news_broadcasts WHERE group_id=? AND broadcast_at>=? AND sent_as!='skipped_no_news'",
            (group_id, since.isoformat()),
        ) as cursor:
            async for (payload_json,) in cursor:
                try:
                    urls.update(str(item.get("url") or "").rstrip("/").lower()
                                for item in json.loads(payload_json).get("items", []))
                except (ValueError, AttributeError, TypeError):
                    continue
        return urls


    async def save_published_briefing(
        self,
        group_id: int,
        card_data: dict,
        *,
        sent_as: str,
        published_at: Optional[datetime] = None,
        briefing_date: date | str | None = None,
    ) -> None:
        """保存已成功发送的结构化简报，供后续新闻播报使用。"""
        published_at = published_at or datetime.now()
        if isinstance(briefing_date, date):
            briefing_date_value = briefing_date.isoformat()
        elif briefing_date:
            briefing_date_value = str(briefing_date)
        else:
            briefing_date_value = published_at.date().isoformat()
        card_json = json.dumps(card_data, ensure_ascii=False)
        db = await self._get_db()
        for attempt in range(SQLITE_WRITE_RETRIES):
            async with measured_write_lock(self._write_lock, "memory.save_published_briefing"):
                from .conversation import memory_write_allowed
                if not memory_write_allowed(f"group_{group_id}"):
                    return
                try:
                    await db.execute(
                        """
                        INSERT INTO daily_briefings (
                            group_id, briefing_date, card_json, sent_as, published_at
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(group_id, briefing_date) DO UPDATE SET
                            card_json = excluded.card_json,
                            sent_as = excluded.sent_as,
                            published_at = excluded.published_at
                        """,
                        (
                            int(group_id),
                            briefing_date_value,
                            card_json,
                            sent_as,
                            published_at.isoformat(),
                        ),
                    )
                    await db.commit()
                    return
                except aiosqlite.OperationalError as error:
                    await self._rollback_safely(db)
                    if not self._is_database_locked_error(error) or attempt >= SQLITE_WRITE_RETRIES - 1:
                        raise
                except BaseException:
                    await self._rollback_safely(db)
                    raise
            await asyncio.sleep(SQLITE_WRITE_RETRY_BASE_DELAY * (attempt + 1))


    async def has_published_briefing_for_date(
        self,
        group_id: int,
        briefing_date: date | str,
    ) -> bool:
        """判断某群某个业务日期的简报是否已经成功发布。"""
        date_value = briefing_date.isoformat() if isinstance(briefing_date, date) else str(briefing_date)
        db = await self._get_db()
        async with db.execute(
            "SELECT 1 FROM daily_briefings WHERE group_id = ? AND briefing_date = ? LIMIT 1",
            (int(group_id), date_value),
        ) as cursor:
            return await cursor.fetchone() is not None


    async def has_pending_briefing_for_news(self, *, published_after: datetime, now: Optional[datetime] = None) -> bool:
        """快速判断是否存在尚未播报的近期简报，供补偿任务避免空轮询。"""
        db = await self._get_db()
        async with db.execute(
            """
            SELECT 1
            FROM daily_briefings AS b
            LEFT JOIN news_broadcasts AS n ON n.briefing_id = b.id
            WHERE b.published_at >= ?
              AND b.id = (SELECT id FROM daily_briefings WHERE group_id=b.group_id ORDER BY published_at DESC, id DESC LIMIT 1)
              AND (n.id IS NULL OR n.sent_as = 'skipped_no_news'
                   OR (n.sent_as = 'retry_pending' AND json_extract(n.payload_json, '$.retry_at') <= ?))
            LIMIT 1
            """,
            (published_after.isoformat(), (now or datetime.now()).isoformat()),
        ) as cursor:
            return await cursor.fetchone() is not None


    async def get_pending_briefing_for_news(
        self,
        group_id: int,
        *,
        published_after: datetime,
        now: Optional[datetime] = None,
    ) -> Optional[dict]:
        """获取最近一份尚未用于新闻播报的已发布简报。"""
        db = await self._get_db()
        async with db.execute(
            """
            SELECT b.id, b.group_id, b.briefing_date, b.card_json,
                   b.sent_as, b.published_at, n.payload_json, n.sent_as
            FROM daily_briefings AS b
            LEFT JOIN news_broadcasts AS n ON n.briefing_id = b.id
            WHERE b.group_id = ?
              AND b.published_at >= ?
              AND b.id = (SELECT id FROM daily_briefings WHERE group_id=b.group_id ORDER BY published_at DESC, id DESC LIMIT 1)
              AND (n.id IS NULL OR n.sent_as = 'skipped_no_news'
                   OR (n.sent_as = 'retry_pending' AND json_extract(n.payload_json, '$.retry_at') <= ?))
            ORDER BY b.published_at DESC
            LIMIT 1
            """,
            (int(group_id), published_after.isoformat(), (now or datetime.now()).isoformat()),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None
        try:
            card_data = json.loads(row[3])
        except (TypeError, json.JSONDecodeError):
            _log.warning(f"已发布简报 JSON 损坏，跳过: briefing_id={row[0]}")
            return None
        if not isinstance(card_data, dict):
            return None
        return {
            "id": row[0],
            "group_id": row[1],
            "briefing_date": row[2],
            "card_data": card_data,
            "sent_as": row[4],
            "published_at": row[5],
            "news_attempts": int(json.loads(row[6] or "{}").get("attempts", 0)) if row[7] == "retry_pending" else 0,
        }


    async def recent_news_topics(self, group_id: int, *, since: datetime, before: datetime) -> list[str]:
        """Only expose published topic summaries to the news planner, not members or raw chat."""
        db = await self._get_db()
        async with db.execute(
            "SELECT card_json FROM daily_briefings WHERE group_id=? AND published_at>=? "
            "AND published_at<=? ORDER BY published_at DESC LIMIT 7",
            (int(group_id), since.isoformat(), before.isoformat()),
        ) as cursor:
            rows = await cursor.fetchall()
        topics = []
        for row in rows:
            try:
                card = json.loads(row[0])
            except (TypeError, ValueError):
                continue
            if not isinstance(card, dict):
                continue
            for topic in card.get("topics") or []:
                text = str(topic).strip()[:200]
                if text and text not in topics:
                    topics.append(text)
                if len(topics) >= 16:
                    return topics
        return topics


    async def get_latest_published_briefing(
        self,
        group_id: int,
        *,
        published_after: datetime,
    ) -> Optional[dict]:
        """获取最近一份已发布简报，供 dry-run/人工补跑使用（不受消费记录影响）。"""
        db = await self._get_db()
        async with db.execute(
            """
            SELECT id, group_id, briefing_date, card_json, sent_as, published_at
            FROM daily_briefings
            WHERE group_id = ?
              AND published_at >= ?
            ORDER BY published_at DESC
            LIMIT 1
            """,
            (int(group_id), published_after.isoformat()),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None
        try:
            card_data = json.loads(row[3])
        except (TypeError, json.JSONDecodeError):
            _log.warning(f"已发布简报 JSON 损坏，跳过: briefing_id={row[0]}")
            return None
        if not isinstance(card_data, dict):
            return None
        return {
            "id": row[0],
            "group_id": row[1],
            "briefing_date": row[2],
            "card_data": card_data,
            "sent_as": row[4],
            "published_at": row[5],
        }


    async def save_news_broadcast(
        self,
        briefing_id: int,
        group_id: int,
        payload: dict,
        *,
        sent_as: str,
        broadcast_at: Optional[datetime] = None,
    ) -> None:
        """标记某份简报已经生成并发送过新闻播报，避免跨重启重复。"""
        broadcast_at = broadcast_at or datetime.now()
        db = await self._get_db()
        for attempt in range(SQLITE_WRITE_RETRIES):
            async with measured_write_lock(self._write_lock, "memory.save_news_broadcast"):
                from .conversation import memory_write_allowed
                if not memory_write_allowed(f"group_{group_id}"):
                    return
                try:
                    await db.execute(
                        """
                        INSERT INTO news_broadcasts (
                            briefing_id, group_id, payload_json, sent_as, broadcast_at
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(briefing_id) DO UPDATE SET
                            group_id = excluded.group_id,
                            payload_json = excluded.payload_json,
                            sent_as = excluded.sent_as,
                            broadcast_at = excluded.broadcast_at
                        WHERE news_broadcasts.sent_as LIKE 'skipped_%'
                           OR news_broadcasts.sent_as IN ('retry_pending', 'retry_exhausted')
                        """,
                        (
                            int(briefing_id),
                            int(group_id),
                            json.dumps(payload, ensure_ascii=False),
                            sent_as,
                            broadcast_at.isoformat(),
                        ),
                    )
                    await db.commit()
                    return
                except aiosqlite.OperationalError as error:
                    await self._rollback_safely(db)
                    if not self._is_database_locked_error(error) or attempt >= SQLITE_WRITE_RETRIES - 1:
                        raise
                except BaseException:
                    await self._rollback_safely(db)
                    raise
            await asyncio.sleep(SQLITE_WRITE_RETRY_BASE_DELAY * (attempt + 1))
