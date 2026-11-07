"""Buffer a single OpenAI-compatible completion before executing any tool.

Antigravity repeats role/name metadata in streamed chunks. Accumulate only text
and argument fragments; repeated metadata must not turn into assistantassistant.
"""
import json

from openai.types.chat import ChatCompletion


async def collect_completion(raw_stream) -> ChatCompletion:
    first = None
    content, reasoning, refusal = [], [], []
    calls = {}
    finish_reason = None
    usage = None
    async for chunk in raw_stream:
        if first is None:
            first = chunk
        if chunk.usage is not None:
            usage = chunk.usage
        for choice in chunk.choices:
            if choice.index != 0:
                raise RuntimeError("模型流式响应返回了未请求的多个候选")
            delta = choice.delta
            if delta.role not in (None, "assistant"):
                raise RuntimeError("模型流式响应角色异常")
            for field, parts in (("content", content), ("reasoning_content", reasoning), ("refusal", refusal)):
                value = getattr(delta, field, None)
                if isinstance(value, str):
                    parts.append(value)
            for part in delta.tool_calls or []:
                call = calls.setdefault(part.index, {
                    "id": "", "type": "function", "function": {"name": "", "arguments": ""},
                })
                if part.type not in (None, "function"):
                    raise RuntimeError("模型流式响应工具类型异常")
                if part.id and part.id != call["id"]:
                    call["id"] += part.id
                if part.function:
                    name = part.function.name
                    if name and name != call["function"]["name"]:
                        call["function"]["name"] += name
                    call["function"]["arguments"] += part.function.arguments or ""
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    if first is None or finish_reason not in ("stop", "tool_calls"):
        raise RuntimeError("模型流式响应未完整结束")
    tool_calls = [calls[index] for index in sorted(calls)]
    for call in tool_calls:
        if not call["id"] or not call["function"]["name"]:
            raise RuntimeError("模型流式响应工具调用不完整")
        try:
            arguments = json.loads(call["function"]["arguments"])
        except (TypeError, ValueError) as error:
            raise RuntimeError("模型流式响应工具参数不完整") from error
        if not isinstance(arguments, dict):
            raise RuntimeError("模型流式响应工具参数不是对象")
    text = "".join(content) or "".join(refusal)
    if not text.strip() and not tool_calls:
        # The generic adapter fallback must never expose reasoning-only output.
        raise RuntimeError("模型流式响应没有最终回复或工具调用")
    message = {"role": "assistant", "content": text or None, "tool_calls": tool_calls or None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    return ChatCompletion(
        id=first.id, object="chat.completion", created=first.created, model=first.model,
        choices=[{"index": 0, "message": message, "finish_reason": finish_reason}], usage=usage,
    )
