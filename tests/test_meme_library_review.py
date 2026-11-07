import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts import review_meme_library as review


def result(image_id, image_hash, eligible):
    return {"id": image_id, "hash": image_hash, "is_meme": eligible,
            "random_eligible": eligible, "topic": "funny" if eligible else None,
            "description": "带字笑话" if eligible else "活动公告", "reason": "图中实际可见内容"}


class MemeReviewTests(unittest.TestCase):
    def test_batch_must_match_images_and_have_real_booleans(self):
        good = result(1, "a", True)
        self.assertEqual(review.parse_results(json.dumps([good]), [1]), [good])
        for bad in ([good, good], [dict(good, id=2)], [dict(good, random_eligible="false")],
                    [dict(good, is_meme=False)], [dict(good, reason="")]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                review.parse_results(json.dumps(bad), [1])

    def test_apply_restore_preserve_rows_files_and_concurrent_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "memes.db"
            original = root / "original.jpg"
            original.write_bytes(b"preserved image")
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("CREATE TABLE memes(id PRIMARY KEY,hash,topic,file_path,random_eligible)")
                db.executemany("INSERT INTO memes VALUES(?,?,?, ?,?)", [
                    (1, "a", "funny", str(original), 1), (2, "b", "shock", str(original), None),
                    (3, "c", "sad", str(original), 1), (4, "d", "funny", str(original), 1)])
            report = root / "report"
            manifest = review.snapshot(database, report)
            for item in (result(1, "a", False), result(2, "b", True), result(3, "c", False), result(4, "d", False)):
                review.append(report / "results.jsonl", item)
            # Collection replaced row 3 since the snapshot. It must win.
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("UPDATE memes SET hash='replacement' WHERE id=3")
                db.execute("UPDATE memes SET random_eligible=0 WHERE id=4")
            with self.assertRaisesRegex(ValueError, "No completed apply"):
                review.apply(database, report, manifest, restore=True)
            review.apply(database, report, manifest)
            with closing(sqlite3.connect(database)) as db, db:
                self.assertEqual(db.execute("SELECT random_eligible FROM memes ORDER BY id").fetchall(),
                                 [(0,), (1,), (1,), (0,)])
            (report / "results.jsonl").unlink()  # Restore only uses the applied changes, not mutable model output.
            review.apply(database, report, manifest, restore=True)
            with closing(sqlite3.connect(database)) as db, db:
                self.assertEqual(db.execute("SELECT random_eligible FROM memes ORDER BY id").fetchall(),
                                 [(1,), (None,), (1,), (0,)])
            self.assertEqual(original.read_bytes(), b"preserved image")

    def test_incomplete_review_cannot_change_database(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            review.append(root / "results.jsonl", result(9, "unrelated", True))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                review.apply(root / "missing.db", root, [{"id": 1, "hash": "a"}])
            self.assertFalse((root / "missing.db").exists())

    def test_invalid_metadata_rolls_back_whole_apply(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "memes.db"
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("CREATE TABLE memes(id PRIMARY KEY,hash,topic,file_path,random_eligible)")
                db.executemany("INSERT INTO memes VALUES(?,?,'funny','unused',1)", [(1, "a"), (2, "b")])
            manifest = review.snapshot(database, root / "report")
            review.append(root / "report/results.jsonl", result(1, "a", False))
            review.append(root / "report/results.jsonl", dict(result(2, "b", False), random_eligible="false"))
            with self.assertRaises(ValueError):
                review.apply(database, root / "report", manifest)
            with closing(sqlite3.connect(database)) as db, db:
                self.assertEqual(db.execute("SELECT random_eligible FROM memes ORDER BY id").fetchall(), [(1,), (1,)])
