"""LLM聊天插件"""
import os
from pathlib import Path

from dotenv import load_dotenv
from nonebot.plugin import PluginMetadata

from .config import Config
from .logging_setup import setup_logging

# nonebot/pydantic-settings 只把已声明的 config 字段从 .env 读进来，
# MODEL_* 路由等自定义 key 不会进 os.environ；这里显式加载一次
# (override=False 保证已存在的 env 优先，避免覆盖容器/CI 注入的值)
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)

# 在最早期初始化日志（创建分类目录、注册 sink）
setup_logging()

# 启动校验：分层模型配置必须可解析，工具候选必须匹配协议能力。
from .roles import validate_all as _validate_role_specs
from .logging_setup import general_logger as _gl
_role_errors = _validate_role_specs()
if _role_errors:
    _msg = "Role 路由配置校验失败:\n  - " + "\n  - ".join(_role_errors)
    _gl.bind(category="llm").error(_msg)
    raise RuntimeError(_msg)
_gl.bind(category="llm").info("[startup] Role 路由配置校验通过")

from .chat import *
from .commands import *
from .events import *
from .briefing import *
from .news_broadcast import *
from .idle_memes import *


# ---- 资源清理：关闭 db / OpenAI 客户端，缓解 Ctrl+C 关停卡死 ----
import nonebot as _nb

_driver = _nb.get_driver()


@_driver.on_startup
async def _llm_chat_start_background_workers():
    from .model_identity import refresh_proxy_model_identity
    await refresh_proxy_model_identity()
    from .memory import memory_manager
    await memory_manager.start()
    from . import conversation, health
    conversation.start()
    health.start()
    _gl.bind(category="llm").info("[startup] 重要信息持久化 worker 已启动")


@_driver.on_shutdown
async def _llm_chat_cleanup():
    log = _gl.bind(category="llm")
    from . import conversation, health
    from .chat import close_chat_tasks, close_incoming_tasks
    from .sentiment import sentiment_analyzer
    from .multimodal import close_downloads
    from .proactive import proactive_manager
    await health.close()
    for group_id in list(proactive_manager._wait_tasks):
        proactive_manager.cancel_wait(group_id)
    await close_chat_tasks()
    await close_incoming_tasks()
    await sentiment_analyzer.close()
    await conversation.close()
    await close_downloads()
    from .models import model_manager
    await model_manager.close_pending()
    # 1. 关 aiosqlite 连接（它内部跑非守护线程，不关进程退不了）
    try:
        from .memory import memory_manager
        await memory_manager.close()
        log.info("[shutdown] 记忆后台任务和 aiosqlite 连接已关闭")
    except Exception as e:
        log.warning(f"[shutdown] 关 db 失败: {e}")

    # 2. 关所有缓存的 AsyncOpenAI / httpx 客户端
    try:
        from .models import model_manager
        for spec, adapter in list(model_manager._adapters.items()):
            client = getattr(adapter, "client", None)
            if client is not None and hasattr(client, "close"):
                try:
                    await client.close()
                except Exception as ce:
                    log.warning(f"[shutdown] 关 client {spec} 失败: {ce}")
        log.info(f"[shutdown] 已关闭 {len(model_manager._adapters)} 个 adapter client")
    except Exception as e:
        log.warning(f"[shutdown] 关 adapters 失败: {e}")

    from .llm_logger import drain_usage_tasks
    await drain_usage_tasks()

    # 3. 关 usage_db 的 aiosqlite 连接
    try:
        from .usage_db import close_db as _close_usage_db
        await _close_usage_db()
        log.info("[shutdown] usage_db 连接已关闭")
    except Exception as e:
        log.warning(f"[shutdown] 关 usage_db 失败: {e}")

    # 4. 关 memes db
    try:
        from .memes import meme_manager
        await meme_manager.close()
        log.info("[shutdown] memes db 连接已关闭")
    except Exception as e:
        log.warning(f"[shutdown] 关 memes db 失败: {e}")

__plugin_meta__ = PluginMetadata(
    name="LysBot",
    description="有记忆、能检索历史的磷叶石群友",
    usage="直接发送消息即可与AI对话，群聊需要触发词或@机器人",
    config=Config,
)
