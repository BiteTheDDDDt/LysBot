"""Initialize/migrate the canonical memory schema on an owned connection.

The manager serializes initialization and performs the legacy session backfill.
Search projections continue to be installed by memory_index.
"""
import aiosqlite


async def initialize_schema(db: aiosqlite.Connection) -> None:
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=30000")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            messages TEXT,
            model TEXT,
            created_at TEXT,
            updated_at TEXT,
            compressed_count INTEGER DEFAULT 0
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            time_range TEXT,
            created_at TEXT NOT NULL
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS important_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            info_type TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS important_info_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_key TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            messages_json TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            next_retry_at TEXT NOT NULL,
            last_error TEXT,
            dead_letter_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    async with db.execute("PRAGMA table_info(important_info_jobs)") as cursor:
        important_info_job_columns = {row[1] for row in await cursor.fetchall()}
    if "dead_letter_at" not in important_info_job_columns:
        await db.execute(
            "ALTER TABLE important_info_jobs ADD COLUMN dead_letter_at TEXT"
        )
    # 明细消息表：用于 recall_history 工具按字段精确检索（与 sessions JSON 双写）
    await db.execute("""
        CREATE TABLE IF NOT EXISTS messages_detail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            name TEXT,
            sender_id INTEGER,
            content TEXT NOT NULL,
            is_thinking INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS daily_briefings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER NOT NULL,
            briefing_date TEXT NOT NULL,
            card_json TEXT NOT NULL,
            sent_as TEXT NOT NULL,
            published_at TEXT NOT NULL,
            UNIQUE(group_id, briefing_date)
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS news_broadcasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            briefing_id INTEGER NOT NULL UNIQUE,
            group_id INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            sent_as TEXT NOT NULL,
            broadcast_at TEXT NOT NULL
        )
    """)
    async with db.execute("PRAGMA table_info(messages_detail)") as cursor:
        message_detail_columns = {row[1] for row in await cursor.fetchall()}
    if "sender_id" not in message_detail_columns:
        await db.execute(
            "ALTER TABLE messages_detail ADD COLUMN sender_id INTEGER"
        )
    if "model_spec" not in message_detail_columns:
        await db.execute("ALTER TABLE messages_detail ADD COLUMN model_spec TEXT")
    # 元数据 kv 表（迁移标记等）
    await db.execute("""
        CREATE TABLE IF NOT EXISTS kv_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_info_session ON important_info(session_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_info_owner_type_created ON important_info(session_id,user_id,info_type,created_at DESC,id DESC)")
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_info_jobs_due ON important_info_jobs(next_retry_at, id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_info_jobs_active_due "
        "ON important_info_jobs(dead_letter_at, next_retry_at, id)"
    )
    await db.execute("CREATE INDEX IF NOT EXISTS idx_md_session_ts ON messages_detail(session_id, timestamp DESC)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_md_session_name ON messages_detail(session_id, name)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_md_session_sender ON messages_detail(session_id, sender_id, timestamp DESC)")
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_daily_briefings_group_time "
        "ON daily_briefings(group_id, published_at DESC)"
    )
    from .memory_index import install_schema
    await install_schema(db)
    await db.commit()
