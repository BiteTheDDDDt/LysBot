"""Small extension API on top of NoneBot; no separate plugin runtime."""
import inspect
import json
import re
from copy import deepcopy
from dataclasses import dataclass

from nonebot import get_driver
from nonebot.adapters.onebot.v11 import MessageSegment


def is_master(user_id):
    return bool(user_id) and str(user_id) in get_driver().config.superusers


@dataclass(frozen=True)
class ToolContext:
    requester_id: str | None
    session_id: str

    @property
    def is_master(self):
        return is_master(self.requester_id)


async def notify_master(bot, text):
    """Send a private notification to the configured owner. No model call."""
    from src.plugins.llm_chat.delivery import deliver_once
    owners = get_driver().config.superusers
    if len(owners) != 1:
        raise ValueError("Exactly one MASTER_QQ is required")
    owner = int(next(iter(owners)))
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 4000:
        raise ValueError("Notification text must contain 1..4000 characters")
    return await deliver_once(
        lambda: bot.send_private_msg(user_id=owner, message=MessageSegment.text(text)),
        session_id=f"user_{owner}",
    )


def register_tool(name, description, parameters, handler, *, master_only=False, read_only=False):
    """Register an async handler(context, **arguments) during plugin loading."""
    from src.plugins.llm_chat import capabilities, tools
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or name in capabilities.CAPABILITIES:
        raise ValueError("Tool name invalid or already registered")
    if not inspect.iscoroutinefunction(handler):
        raise TypeError("Tool handler must be async")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError("Tool parameters must be an object schema")
    parameters = deepcopy(parameters)
    json.dumps(parameters)  # Fail at registration, rather than during a model call.
    schema = {"type": "function", "function": {
        "name": name, "description": description, "parameters": parameters,
    }}

    async def guarded(**arguments):
        from src.plugins.llm_chat.agent_runner import current_run
        run = current_run.get()
        if run is None:
            raise PermissionError("Extension tools require an active conversation")
        context = ToolContext(str(run.requester_id) if run.requester_id else None, run.session_id)
        if master_only and not context.is_master:
            raise PermissionError("Master permission required")
        inspect.signature(handler).bind(context, **arguments)
        return await handler(context, **arguments)

    capabilities.CAPABILITIES[name] = capabilities.Capability(schema, read_only=read_only)
    capabilities.BASE_CHAT_TOOLS.append(schema)
    capabilities.AVAILABLE_TOOLS.append(schema)
    if read_only:
        capabilities.READ_TOOLS.add(name)
    tools.TOOL_FUNCTIONS[name] = guarded
    _master_tools[name] = master_only


_master_tools = {}


def tool_visible(name):
    if not _master_tools.get(name):
        return True
    from src.plugins.llm_chat.agent_runner import current_run
    run = current_run.get()
    return bool(run and is_master(run.requester_id))
