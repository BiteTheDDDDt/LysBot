#!/usr/bin/env python3
"""QQ大模型机器人入口"""
import nonebot
import os
from pathlib import Path
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

from src.bootstrap import prepare_environment
prepare_environment()
nonebot.init()

# A previous process's readiness marker must not make a failed plugin import look healthy.
Path(os.environ.get("LLM_CHAT_DATA_DIR", "data"), "health.json").unlink(missing_ok=True)

driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)

nonebot.load_from_toml("pyproject.toml")
# Load extensions only after the core has registered its handlers.
nonebot.load_plugins("extensions")

if __name__ == "__main__":
    nonebot.run()
