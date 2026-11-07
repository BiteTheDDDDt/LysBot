import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


class DatabasePathTests(unittest.TestCase):
    def paths(self):
        return runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/plugins/llm_chat/paths.py"))

    def test_database_override_keeps_other_assets_in_data_directory(self):
        with patch.dict(os.environ, {"LLM_CHAT_DATA_DIR": "/app/data", "LLM_CHAT_DATABASE_PATH": "/app/database/history.db"}):
            data_path = self.paths()["data_path"]
            self.assertEqual(data_path("history.db"), Path("/app/database/history.db"))
            self.assertEqual(data_path("health.json"), Path("/app/data/health.json"))
            self.assertEqual(data_path("models", "bge"), Path("/app/data/models/bge"))

    def test_empty_override_preserves_existing_database_location(self):
        with patch.dict(os.environ, {"LLM_CHAT_DATA_DIR": "/app/data", "LLM_CHAT_DATABASE_PATH": ""}):
            self.assertEqual(self.paths()["data_path"]("history.db"), Path("/app/data/history.db"))
