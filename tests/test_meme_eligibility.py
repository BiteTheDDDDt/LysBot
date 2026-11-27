import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from src.plugins.llm_chat.memes import MemeManager


class MemeEligibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_reaction_and_joke_pools_are_independent(self):
        m = self.manager
        reaction = await m.add_meme(b"reaction", "funny", "捂脸", random_eligible=True, meme_kind="reaction")
        joke = await m.add_meme(b"joke", "funny", "完整两句笑话", random_eligible=True, meme_kind="joke", classification_reason="独立笑话", text_sentence_count=2)
        self.assertEqual((await m.pick_random("funny"))[0], reaction)
        self.assertEqual((await m.pick_random(None, meme_kind="joke"))[0], joke)
        self.assertIsNone(await m.add_meme(b"long", "funny", "长截图", random_eligible=True, meme_kind="joke", text_sentence_count=3))

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = object.__new__(MemeManager)
        self.manager._db = None
        self.manager._db_initialized = False
        self.manager._db_lock = asyncio.Lock()
        self.manager._db_path = str(Path(self.temp.name) / "memes.db")
        self.manager._files_dir = Path(self.temp.name) / "images"

    async def asyncTearDown(self):
        if self.manager._db:
            await self.manager._db.close()
        self.temp.cleanup()

    async def test_contextual_caption_is_collected_but_not_random_then_can_be_restored(self):
        m = self.manager
        meme_id = await m.add_meme(b"context", "funny", "好股票", random_eligible=False)
        self.assertIsNotNone(meme_id)
        self.assertIsNone(await m.pick_random("funny"))
        await m.update_random_eligibility(b"context", True)
        self.assertEqual((await m.pick_random("funny"))[0], meme_id)
        await m.update_random_eligibility(b"context", None)
        self.assertEqual((await m.pick_random("funny"))[0], meme_id)
        await m.update_random_eligibility(b"context", False)
        self.assertIsNone(await m.pick_random("funny"))
        db = await m._get_db()
        row = await (await db.execute("SELECT file_path FROM memes WHERE id=?", (meme_id,))).fetchone()
        self.assertTrue(Path(row[0]).exists())

    async def test_duplicate_reclassification_updates_eligibility_without_new_file(self):
        m = self.manager
        first = await m.add_meme(b"caption", "funny", "完整带字笑话", random_eligible=True)
        self.assertEqual((await m.pick_random("funny"))[0], first)
        second = await m.add_meme(b"caption", "funny", "完整带字笑话", random_eligible=False)
        self.assertEqual(first, second)
        self.assertIsNone(await m.pick_random("funny"))

    async def test_old_schema_migrates_without_dropping_images(self):
        image = Path(self.temp.name) / "old.jpg"
        image.write_bytes(b"old image")
        with closing(sqlite3.connect(self.manager._db_path)) as db:
            db.execute("""CREATE TABLE memes (
                id INTEGER PRIMARY KEY, hash TEXT UNIQUE, topic TEXT, description TEXT,
                file_path TEXT, source_url TEXT, source_user_id INTEGER, source_group_id INTEGER,
                created_at TEXT, used_count INTEGER DEFAULT 0, last_used_at TEXT)""")
            db.execute("INSERT INTO memes(id,hash,topic,description,file_path) VALUES(1,'old','funny','猫表情夸张',?)", (str(image),))
            db.commit()
        # Legacy descriptions are not an eligibility verdict, even if they
        # contain the old positive keywords. Keep the image for re-review.
        self.assertIsNone(await self.manager.pick_random("funny"))
        self.assertTrue(image.exists())
        db = await self.manager._get_db()
        await db.execute("UPDATE memes SET random_eligible=1, meme_kind='reaction' WHERE id=1")
        await db.commit()
        self.assertEqual((await self.manager.pick_random("funny"))[0], 1)

    async def test_migrated_path_resolves_by_hash_and_missing_files_are_skipped(self):
        m = self.manager
        meme_id = await m.add_meme(b"caption", "funny", "完整带字笑话", random_eligible=True)
        original = (await m.pick_random("funny"))[1]
        db = await m._get_db()
        await db.execute("UPDATE memes SET file_path=? WHERE id=?", ("Z:/old-host/meme.jpg", meme_id))
        await db.commit()
        self.assertEqual((await m.pick_random("funny"))[1], original)
        Path(original).unlink()
        self.assertIsNone(await m.pick_random("funny"))
