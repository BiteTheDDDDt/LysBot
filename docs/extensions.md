# 扩展 LysBot

扩展是可信的本地 Python 代码，沿用 NoneBot 插件机制，不是沙箱。把插件放进 `extensions/`；此目录中的私人代码默认不进入 Git。安装依赖后重启 bot，加载失败会出现在 NoneBot 日志中。不要把插件安装失败当作加载成功。

LysBot 启动时先加载核心，再加载 `extensions/`，扩展使用 `lysbot` API 访问同一套能力目录。不要另建同名的 `llm_chat` 包。

## 向 planner 注册工具

```python
from lysbot import ToolContext, register_tool

async def project_status(context: ToolContext, project: str):
    return f"{project}: 暂无待办"

register_tool(
    "project_status",
    "查询项目状态",
    {"type": "object", "properties": {"project": {"type": "string"}}, "required": ["project"]},
    project_status,
    master_only=True,
    read_only=True,
)
```

上下文由程序注入，包含真实请求者与当前会话；工具参数不能指定管理员身份。`master_only=True` 同时控制工具可见性与执行时权限。重名和覆盖内置工具会被拒绝。注册应在插件加载时完成，不能在每条消息中重复注册。

工具共享现有回合预算、活动跟踪和执行策略。`read_only` 是插件作者声明的行为约定，不会对 Python 代码提供沙箱隔离。插件应自行限制外部访问、参数范围及返回体积，不把整个数据库或私密配置返回给模型。

## 定时任务和外部通知

投资通知这类功能不需要经过模型。使用 NoneBot 的事件、scheduler 或本地队列处理任务，通过统一接口通知 master：

```python
from nonebot import get_bots, require
from lysbot import notify_master

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler

@scheduler.scheduled_job("interval", minutes=10, max_instances=1)
async def check_jobs():
    # 从自己的持久化任务队列取得一个待发送结果。
    result = None
    if result:
        bot = next(iter(get_bots().values()), None)
        if bot:
            status, receipt = await notify_master(bot, result)
            # sent 表示收到 QQ 发送回执；unconfirmed 不应盲目重发。
```

`notify_master` 仅发送给配置中的唯一管理员，不接收模型提供的收件人。接口保留已发送／未确认状态；持久化队列、过期策略和任务级防重由插件负责。投递成功不代表对方已读。
