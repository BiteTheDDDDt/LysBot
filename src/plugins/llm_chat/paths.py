"""Runtime filesystem paths for the llm_chat plugin.

Defaults intentionally keep the existing behavior: paths are relative to the
process working directory. Set LLM_CHAT_DATA_DIR only when deploying with an
explicit runtime data directory.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("LLM_CHAT_DATA_DIR", "data"))


def data_path(*parts: str) -> Path:
    if parts == ("history.db",) and (database := os.environ.get("LLM_CHAT_DATABASE_PATH")):
        return Path(database)
    return DATA_DIR.joinpath(*parts)
