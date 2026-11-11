"""表情包管理：采集、入库、检索、冷却。

设计：
- 6 个 topic（happy/sad/angry/funny/shock/cute），让 LLM 容易选对，冷启动可承受
- LLM 不知道库里具体有什么图，只指定 topic；后端命中 topic 后随机抽一张
- 文件按 sha256 hash 存盘，hash 同时是去重键
- 群级冷却 5 分钟，同一次 chat 回合最多发 1 张
"""
import asyncio
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiosqlite

logger = logging.getLogger(__name__)

# 受控 topic 词表 —— LLM 只能从中选
ALLOWED_TOPICS = {
    "happy",   # 开心 / 笑 / 赞同 / 兴奋
    "sad",     # 难过 / 哭 / 委屈 / 破防
    "angry",   # 生气 / 嫌弃 / 拒绝 / 不爽
    "funny",   # 搞笑 / 玩梗 / 嘚瑟 / 沙雕
    "shock",   # 震惊 / 疑惑 / 无语
    "cute",    # 可爱 / 卖萌 / 撒娇
}

# 群级冷却（秒）
GROUP_COOLDOWN_SEC = 300

# Shared by live collection and the explicit library review command.
MEME_CLASSIFICATION_RULES = (
    "meme_kind区分两种用途：reaction=可直接表达情绪、态度的反应图；joke=自成笑点、供人观看的网络梗图，不用来代替对人的回应。不是梗图时meme_kind=null。"
    "帖子截图即使确有完整笑点，也通常属于joke；普通资讯帖子、广告或纯信息截图仍为false，不因格式像梗图就收录。"
    "先判断图像的实际用途，再分类情绪；不能为了给它选情绪而倒推它是表情包。"
    "is_meme=true 必须有可见依据：图中文字/图文反差构成完整笑点，或画面集中表达明确的聊天反应。"
    "反应图例如举牌打分、捂脸崩溃、瞪眼质疑、白底委屈猫、夸张捧腹笑；不要求有字或后期加工。"
    "硬性上限：图片实际可见正文达到三句或更多，直接判is_meme=false、random_eligible=false、topic=null，即使有笑点也不收。"
    "按独立语句计数，不按排版换行；没有句末标点的独立聊天气泡/短句也计入，水印、用户名和界面按钮不计。"
    "用text_sentence_count返回实际正文句数，不数你自己的描述。少于三句的带字笑话、多格漫画、圈内梗和聊天截图仍按笑点与反应用途判断。"
    "普通插画、壁纸、头像、人物/宠物照片不能仅因可爱、脸红、微笑、近脸、Q版或温馨氛围就判为表情包。"
    "要分清画面主旨：展示角色和场景的插画不是反应图，角色偶有神情也不能让整幅插画自动变成meme。"
    "观者觉得角色萌，不等于角色正在表达聊天反应；吃东西、喂东西、走路等日常动作也不能仅因萌就收录。"
    "例如秋日两人依偎的精美插画、安静站立的萌系角色、普通宠物留影为false；"
    "主体集中、带有明确回应动作或强烈情绪的聊天贴纸可为true。"
    "普通新闻、活动公告、广告、商品页、文档、游戏界面、个人资料及视频介绍页为false。"
    "不能把商品名的一点谐音、标题夸张、采访日常字幕或评论有趣自动视为整图的笑点。"
    "只是讲述发生了什么、做过什么梦的普通对话字幕不构成笑话，须有实际的包袱、反转、反差或荒诞表达。"
    "例如含价格和套餐列表的‘Hold不住鸡排堡’商品页仍是商品页，不是梗图。"
    "这些形式若确实承载完整自足的笑话或被改造成反应图，也可为true；依据必须来自图中，不替图片编梗。"
    "无法指出具体反应表达或完整笑点时，is_meme=false，不用‘可以当作卖萌’等想象用途兜底。"
    "只根据实际可见的内容判断；看不清的字不要补写，不要假定图片年代或把正常日期当成穿越梗。"
    "水印、评论、零散叠字本身不足以把一张公告变成梗图，需要看整张图是否有明确笑点或反应用途。\n"
    "random_eligible 表示只按情绪随机发到不同话题是否合适，与是否收藏分开。"
    "文字本身能自成笑点（如‘虽然什么都没做，但还是辛苦我自己了’）可以为true；"
    "图内已经给齐笑话所需对话和信息，不算依赖外部上下文；不能仅因有影视出处或社交平台界面就排除随机使用。"
    "必须知道特定事件或紧接股票等特定话题才合适的图为false，仍可is_meme=true。"
    "例如‘好股票’图收藏为true、random_eligible=false；不是梗图时两者均false。\n"
    "topic 只能选 happy / sad / angry / funny / shock / cute；不是梗图时 topic=null。"
    "含义：happy=开心/笑/赞同；sad=难过/哭/委屈；angry=生气/嫌弃/拒绝；"
    "funny=搞笑/玩梗/嘚瑟/沙雕；shock=震惊/疑惑/无语；cute=撒娇/贴贴/讨好等互动反应，"
    "不是所有可爱图片的收纳桶。\n"
    "is_meme 和 random_eligible 必须是不带引号的 true 或 false。"
)

THOUGHT_PREFIX_RE = re.compile(r"^\s*<thought>\s*", re.IGNORECASE)
JSON_TAIL_RE = re.compile(r"\{[^{}]*\"is_meme\"[^{}]*\}\s*$", re.DOTALL)


def clean_meme_memory_description(description: str, max_chars: int = 120) -> str:
    """Return a compact, prompt-safe description for remembering a sent meme."""
    text = (description or "").strip()
    if not text:
        return ""

    text = JSON_TAIL_RE.sub("", text).strip()
    if re.search(r"</thought>", text, re.IGNORECASE):
        text = re.sub(r"(?is)^.*?</thought>", "", text).strip()
    text = THOUGHT_PREFIX_RE.sub("", text).strip()
    text = re.sub(r"</thought>", "", text, flags=re.IGNORECASE).strip()

    lines = []
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[*\-\d\.\)\s]+", "", line).strip()
        if not line:
            continue
        line = re.sub(
            r"^\**\s*(image content|content|image|scene|subject|main subject|description|natural language description|画面内容|图片内容)\s*\**\s*[:：]\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()
        if line:
            lines.append(line)
        if len(" ".join(lines)) >= max_chars:
            break

    cleaned = "；".join(lines) if lines else text
    cleaned = re.sub(r"\s+", " ", cleaned).strip("；; ，,。")
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "..."
    return cleaned


class MemeManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._db: Optional[aiosqlite.Connection] = None
        self._db_lock = asyncio.Lock()
        self._db_initialized = False

        base = Path(__file__).resolve().parents[3]
        self._db_path = str(base / "data" / "memes.db")
        self._files_dir = base / "data" / "memes"
        self._files_dir.mkdir(parents=True, exist_ok=True)

        # 群级冷却： {group_id: last_sent_ts}
        self._group_cooldown: Dict[int, float] = {}

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is not None and self._db_initialized:
            return self._db
        async with self._db_lock:
            if self._db is not None and self._db_initialized:
                return self._db
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS memes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash TEXT UNIQUE NOT NULL,
                    topic TEXT NOT NULL,
                    description TEXT,
                    memory_description TEXT,
                    file_path TEXT NOT NULL,
                    source_url TEXT,
                    source_user_id INTEGER,
                    source_group_id INTEGER,
                    created_at TEXT NOT NULL,
                    used_count INTEGER DEFAULT 0,
                    last_used_at TEXT
                )
            """)
            await self._ensure_schema()
            await self._db.execute("CREATE INDEX IF NOT EXISTS idx_memes_topic ON memes(topic)")
            await self._db.commit()
            self._db_initialized = True
            return self._db

    async def _ensure_schema(self):
        """Apply lightweight schema migrations for existing meme databases."""
        if self._db is None:
            return
        async with self._db.execute("PRAGMA table_info(memes)") as cur:
            columns = {row[1] for row in await cur.fetchall()}
        if "memory_description" not in columns:
            await self._db.execute("ALTER TABLE memes ADD COLUMN memory_description TEXT")
        if "random_eligible" not in columns:
            await self._db.execute("ALTER TABLE memes ADD COLUMN random_eligible INTEGER")
        for name, sql_type in (("meme_kind", "TEXT"), ("classification_reason", "TEXT"), ("text_sentence_count", "INTEGER")):
            if name not in columns:
                await self._db.execute(f"ALTER TABLE memes ADD COLUMN {name} {sql_type}")

        async with self._db.execute(
            "SELECT id, description FROM memes WHERE memory_description IS NULL OR trim(memory_description) = ''"
        ) as cur:
            rows = await cur.fetchall()
        for meme_id, description in rows:
            await self._db.execute(
                "UPDATE memes SET memory_description = ? WHERE id = ?",
                (clean_meme_memory_description(description), meme_id),
            )

    # -------- 采集 --------
    async def add_meme(
        self,
        image_data: bytes,
        topic: str,
        description: str,
        source_url: Optional[str] = None,
        source_user_id: Optional[int] = None,
        source_group_id: Optional[int] = None,
        ext: str = "jpg",
        random_eligible: Optional[bool] = None,
        meme_kind: Optional[str] = "reaction",
        classification_reason: str = "",
        text_sentence_count: Optional[int] = None,
    ) -> Optional[int]:
        """入库一张表情包。返回 id；hash 已存在则返回已有 id，不重复落盘。"""
        if topic not in ALLOWED_TOPICS:
            logger.warning(f"[meme] 拒绝入库：未知 topic={topic}")
            return None
        if meme_kind not in {"reaction", "joke"} or (type(text_sentence_count) is int and text_sentence_count >= 3):
            return None
        if type(random_eligible) is not bool:
            return None
        memory_description = clean_meme_memory_description(description)
        if not memory_description:
            logger.info(f"[meme] 拒绝入库：无法生成可记忆描述 topic={topic} desc={(description or '')[:60]}")
            return None

        sha = hashlib.sha256(image_data).hexdigest()
        db = await self._get_db()
        async with db.execute("SELECT id FROM memes WHERE hash = ?", (sha,)) as cur:
            row = await cur.fetchone()
            if row:
                await self.update_random_eligibility(image_data, random_eligible)
                await db.execute("UPDATE memes SET meme_kind=?, classification_reason=?, text_sentence_count=?, description=?, memory_description=? WHERE id=?",
                                 (meme_kind, classification_reason, text_sentence_count, description, memory_description, row[0]))
                await db.commit()
                return row[0]

        # 落盘：data/memes/<hash[:2]>/<hash>.<ext>
        sub = self._files_dir / sha[:2]
        sub.mkdir(parents=True, exist_ok=True)
        ext = (ext or "jpg").lstrip(".").lower()
        if ext not in {"jpg", "jpeg", "png", "gif", "webp"}:
            ext = "jpg"
        file_path = sub / f"{sha}.{ext}"
        try:
            await asyncio.to_thread(file_path.write_bytes, image_data)
        except Exception as e:
            logger.error(f"[meme] 写文件失败 hash={sha[:8]}: {e}")
            return None

        from datetime import datetime
        now = datetime.now().isoformat()
        try:
            cur = await db.execute(
                """INSERT INTO memes
                   (hash, topic, description, memory_description, file_path, source_url, source_user_id, source_group_id, created_at, random_eligible,meme_kind,classification_reason,text_sentence_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sha,
                    topic,
                    description or "",
                    memory_description,
                    str(file_path),
                    source_url,
                    source_user_id,
                    source_group_id,
                    now,
                    random_eligible,
                    meme_kind, classification_reason, text_sentence_count,
                ),
            )
            await db.commit()
            logger.info(f"[meme] 入库成功 id={cur.lastrowid} topic={topic} hash={sha[:8]} desc={memory_description[:30]}")
            await self.prune_topic(topic)
            return cur.lastrowid
        except aiosqlite.IntegrityError:
            # 并发竞争下的二次保险
            async with db.execute("SELECT id FROM memes WHERE hash = ?", (sha,)) as c2:
                row = await c2.fetchone()
                return row[0] if row else None

    async def update_random_eligibility(self, image_data: bytes, eligible: Optional[bool], *, classification=None):
        """Refresh a known image without deleting it; missing/invalid metadata changes nothing."""
        if type(eligible) is not bool:
            return
        db = await self._get_db()
        if isinstance(classification, dict) and classification.get("classification_reason"):
            count = classification.get("text_sentence_count")
            kind = classification.get("meme_kind")
            if classification.get("is_meme") is False or (type(count) is int and count >= 3):
                kind, eligible = None, False
            if kind in {"reaction", "joke"} or eligible is False:
                await db.execute("UPDATE memes SET meme_kind=?,classification_reason=?,text_sentence_count=? WHERE hash=?",
                                 (kind, classification["classification_reason"], count, hashlib.sha256(image_data).hexdigest()))
        await db.execute("UPDATE memes SET random_eligible=? WHERE hash=?",
                         (eligible, hashlib.sha256(image_data).hexdigest()))
        await db.commit()

    async def prune_topic(self, topic: str) -> int:
        """按 topic 上限淘汰最久未使用/最旧入库的表情包，返回删除数量。"""
        if topic not in ALLOWED_TOPICS:
            return 0
        from .config import config

        max_per_topic = int(getattr(config, "meme_max_per_topic", 100) or 0)
        if max_per_topic <= 0:
            return 0

        db = await self._get_db()
        async with db.execute(
            """SELECT id, file_path
               FROM memes
               WHERE topic = ?
               ORDER BY COALESCE(last_used_at, created_at) DESC, id DESC""",
            (topic,),
        ) as cur:
            rows = await cur.fetchall()

        victims = rows[max_per_topic:]
        if not victims:
            return 0

        victim_ids = [row[0] for row in victims]
        placeholders = ",".join("?" for _ in victim_ids)
        await db.execute(f"DELETE FROM memes WHERE id IN ({placeholders})", victim_ids)
        await db.commit()

        for _, file_path in victims:
            fp = Path(file_path)
            try:
                if fp.exists():
                    await asyncio.to_thread(fp.unlink)
            except OSError as e:
                logger.warning(f"[meme] 淘汰文件失败 {fp}: {e}")
        logger.info(f"[meme] topic={topic} 按上限 {max_per_topic} 淘汰 {len(victims)} 张")
        return len(victims)

    async def prune_all_topics(self) -> Dict[str, int]:
        """对所有 topic 执行上限淘汰。"""
        result: Dict[str, int] = {}
        for topic in ALLOWED_TOPICS:
            result[topic] = await self.prune_topic(topic)
        return result

    # -------- 检索 --------
    async def pick_random(self, topic: Optional[str], *, meme_kind: str = "reaction") -> Optional[Tuple[int, str, str]]:
        """在指定 topic 的合格图片中等概率随机抽取，不按使用次数或时间加权。"""
        if (topic is not None and topic not in ALLOWED_TOPICS) or meme_kind not in {"reaction", "joke"}:
            return None
        db = await self._get_db()
        async with db.execute(
            """SELECT id, file_path, COALESCE(NULLIF(memory_description, ''), description), description, random_eligible, hash
               FROM memes
               WHERE (? IS NULL OR topic = ?) AND random_eligible=1 AND meme_kind=?
               ORDER BY RANDOM()""",
            (topic, topic, meme_kind),
        ) as cur:
            rows = await cur.fetchall()
            for row in rows:
                path = Path(row[1])
                if not path.is_file():
                    # Resolve old host paths by content hash after a move to Docker.
                    matches = list((self._files_dir / row[5][:2]).glob(row[5] + ".*"))
                    if len(matches) != 1 or not matches[0].is_file():
                        continue
                    path = matches[0]
                return (row[0], str(path), row[2])
            return None

    async def mark_used(self, meme_id: int):
        from datetime import datetime
        db = await self._get_db()
        await db.execute(
            "UPDATE memes SET used_count = used_count + 1, last_used_at = ? WHERE id = ?",
            (datetime.now().isoformat(), meme_id),
        )
        await db.commit()

    # -------- 冷却 --------
    def is_on_cooldown(self, group_id: Optional[int]) -> bool:
        if not group_id:
            return False
        last = self._group_cooldown.get(group_id, 0)
        return (time.time() - last) < GROUP_COOLDOWN_SEC

    def mark_sent(self, group_id: Optional[int]):
        if group_id:
            self._group_cooldown[group_id] = time.time()

    # -------- 统计 --------
    async def stats(self) -> Dict[str, int]:
        db = await self._get_db()
        out = {t: 0 for t in ALLOWED_TOPICS}
        async with db.execute("SELECT topic, COUNT(*) FROM memes GROUP BY topic") as cur:
            async for topic, cnt in cur:
                if topic in out:
                    out[topic] = cnt
        return out

    async def recent(self, limit: int = 5) -> List[dict]:
        db = await self._get_db()
        rows = []
        async with db.execute(
            "SELECT id, topic, description, used_count, created_at FROM memes ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            async for r in cur:
                rows.append({"id": r[0], "topic": r[1], "description": r[2], "used_count": r[3], "created_at": r[4]})
        return rows

    async def delete(self, meme_id: int) -> bool:
        db = await self._get_db()
        async with db.execute("SELECT file_path FROM memes WHERE id = ?", (meme_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                return False
            fp = Path(row[0])
        await db.execute("DELETE FROM memes WHERE id = ?", (meme_id,))
        await db.commit()
        try:
            if fp.exists():
                await asyncio.to_thread(fp.unlink)
        except Exception as e:
            logger.warning(f"[meme] 删除文件失败 {fp}: {e}")
        return True

    async def close(self):
        if self._db is not None:
            try:
                await self._db.close()
            except Exception:
                pass
            self._db = None


meme_manager = MemeManager()
