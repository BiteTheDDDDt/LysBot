"""Deterministic score shared by /jrrp and the conversation tool."""
from datetime import datetime
import hashlib


def daily_luck_score(user_id: int, day_index: int | None = None) -> int:
    if day_index is None:
        day_index = int(datetime.now().timestamp() / 86400 + 0.5)
    seed = f"{user_id}{day_index}42".encode()
    # Keep the original 2bot rounding: JS converts the hash to a binary64 number before modulo.
    return int(float(int(hashlib.sha256(seed).hexdigest(), 16)) % 101)
