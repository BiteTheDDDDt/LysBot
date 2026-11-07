"""Environment setup without importing the NoneBot plugin."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def prepare_environment():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    master = os.environ.get("MASTER_QQ", "").strip()
    if not master.isdecimal() or int(master) <= 0:
        raise ValueError("Set MASTER_QQ to your administrator QQ account")
    os.environ["SUPERUSERS"] = json.dumps([master])
