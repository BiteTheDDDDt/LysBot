import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock, patch

import nonebot
nonebot.init()
from src.plugins.llm_chat import memory_index


class SQLiteCleanupTests(unittest.TestCase):
    def test_failed_connection_setup_always_closes_connection(self):
        db = MagicMock()
        db.execute.side_effect = sqlite3.OperationalError('database is locked')
        db.total_changes = 0
        with patch.object(memory_index.sqlite3, 'connect', return_value=db):
            with self.assertRaises(sqlite3.OperationalError):
                with memory_index.connect('unused', operation='test.setup'):
                    self.fail('connection setup should not reach body')
        db.close.assert_called_once()

    def test_stale_vector_tables_are_dropped_but_active_version_is_kept(self):
        versions = {name: char * 16 for name, char in (('stale', 'a'), ('active', 'b'), ('current', 'c'))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.db'
            with memory_index.connect(path, operation='test.setup') as db:
                db.execute('CREATE TABLE kv_meta(key TEXT PRIMARY KEY, value TEXT)')
                db.execute("INSERT INTO kv_meta VALUES('memory_active_embedding',?)", (versions['active'],))
                for version in versions.values():
                    db.execute(f'CREATE TABLE memory_vec_{version}(item_id INTEGER PRIMARY KEY)')
                db.execute("CREATE TABLE memory_vec_not_a_version(item_id INTEGER PRIMARY KEY)")
            index = memory_index.MemoryIndex(path)
            index.version = versions['current']
            with memory_index.connect(path, operation='test.retire') as db:
                index._retire_vector_tables(db)
            with closing(sqlite3.connect(path)) as db:
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn(f"memory_vec_{versions['current']}", tables)
            self.assertIn(f"memory_vec_{versions['active']}", tables)
            self.assertNotIn(f"memory_vec_{versions['stale']}", tables)
            self.assertIn('memory_vec_not_a_version', tables)
