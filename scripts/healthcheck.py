"""Read readiness without importing the NoneBot plugin or modifying SQLite."""
import json
import os
from pathlib import Path
import socket
import sys
import time


def check(data_dir: Path, *, now=None, require_onebot=True) -> tuple[bool, str]:
    try:
        state = json.loads((data_dir / "health.json").read_text(encoding="utf-8"))
        age = (time.time() if now is None else now) - float(state["updated_at"])
        if age < -10 or age > 120:
            return False, "plugin heartbeat stale"
        if not state.get("plugin_ready"):
            return False, "plugin not ready"
        if not state.get("database_ok"):
            return False, "database unhealthy"
        if require_onebot and not state.get("onebot_connected"):
            return False, "OneBot disconnected"
        return True, "ready"
    except (OSError, ValueError, KeyError, TypeError):
        return False, "plugin heartbeat unavailable"


if __name__ == "__main__":
    ok, message = check(Path(os.environ.get("LLM_CHAT_DATA_DIR", "data")))
    if ok:
        try:
            with socket.create_connection(("127.0.0.1", int(os.environ.get("PORT", "8080"))), timeout=3):
                pass
        except OSError:
            ok, message = False, "HTTP listener unavailable"
    print(message)
    sys.exit(0 if ok else 1)
