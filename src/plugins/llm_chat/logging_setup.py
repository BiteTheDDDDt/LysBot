"""日志基础设施

把 nonebot 的 loguru logger 按 category 同时落到多个分类文件，
方便事后按维度检索（聊天/记忆/错误）。

约定:
    logger.bind(category="llm").info(...)      -> data/logs/llm/llm-YYYY-MM-DD.log
    logger.bind(category="memory").info(...)   -> data/logs/memory/...

所有日志（不论 category）也会进入 data/logs/all/all-YYYY-MM-DD.log；
ERROR+ 级别同时落入 data/logs/error/error-YYYY-MM-DD.log。

终端输出由 nonebot 默认配置控制，本模块不改动。
纯文本，不压缩，retention=3 天。
"""
from __future__ import annotations

import sys
from pathlib import Path

from nonebot import logger

from .paths import data_path

LOG_ROOT = data_path("logs")
CATEGORIES = ("llm", "memory")
ROTATION = "00:00"
RETENTION = "3 days"

DEFAULT_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | "
    "{extra[category]:<8} | {name}:{function}:{line} - {message}"
)

_initialized = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _patch_default_category(record: dict) -> None:
    """给没显式 bind(category=...) 的日志补一个默认 category，避免格式串 KeyError"""
    record["extra"].setdefault("category", "general")


def _ensure_utf8_stdio() -> None:
    """Windows 重定向 stdout/stderr 时默认可能是 GBK，会让 emoji 日志触发 Loguru 内部异常。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):
            # 某些宿主会禁止重配标准流；文件 sink 已是 UTF-8，终端失败不影响业务。
            continue


def setup_logging() -> None:
    """注册所有文件 sink。重复调用幂等。"""
    global _initialized
    if _initialized:
        return

    _ensure_utf8_stdio()
    _ensure_dir(LOG_ROOT)
    for cat in CATEGORIES:
        _ensure_dir(LOG_ROOT / cat)
    _ensure_dir(LOG_ROOT / "error")
    _ensure_dir(LOG_ROOT / "all")

    logger.configure(patcher=_patch_default_category)

    for cat in CATEGORIES:
        logger.add(
            LOG_ROOT / cat / (cat + "-{time:YYYY-MM-DD}.log"),
            rotation=ROTATION,
            retention=RETENTION,
            encoding="utf-8",
            enqueue=True,
            format=DEFAULT_FORMAT,
            filter=lambda record, c=cat: record["extra"].get("category") == c,
            level="DEBUG",
        )

    logger.add(
        LOG_ROOT / "error" / "error-{time:YYYY-MM-DD}.log",
        rotation=ROTATION, retention=RETENTION, encoding="utf-8",
        enqueue=True, format=DEFAULT_FORMAT, level="ERROR",
    )

    logger.add(
        LOG_ROOT / "all" / "all-{time:YYYY-MM-DD}.log",
        rotation=ROTATION, retention=RETENTION, encoding="utf-8",
        enqueue=True, format=DEFAULT_FORMAT, level="DEBUG",
    )

    _initialized = True
    logger.bind(category="general").info(
        f"[logging_setup] 日志分类落盘已启用 root={LOG_ROOT.resolve()} retention={RETENTION}"
    )


# 分类专用 logger 直接 import 用
llm_logger = logger.bind(category="llm")
memory_logger = logger.bind(category="memory")
general_logger = logger.bind(category="general")
