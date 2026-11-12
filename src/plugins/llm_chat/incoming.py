"""Normalize OneBot messages, mentions, quotes and forwards within bounded budgets.

Owns only parsing and the mention cache; no handlers, model calls or memory writes.
"""
import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Message, MessageEvent


def format_sender_name(name: str) -> str:
    """格式化发送者名称，防止过长的诗句或提问干扰上下文"""
    if not name:
        return "未知用户"
    name = name.strip()
    if len(name) > 8:
        # 截断过长的名字
        return name[:8] + "..."
    return name

@dataclass
class ResolvedIncomingMessage:
    """One normalized view shared by record/planner/chat paths."""

    plain_text: str
    response_text: Optional[str] = None
    context_blocks: list[str] = field(default_factory=list)
    direct_image_urls: list[str] = field(default_factory=list)
    context_image_urls: list[str] = field(default_factory=list)
    unavailable_images: bool = False
    faces: list[str] = field(default_factory=list)

    @property
    def image_urls(self) -> list[str]:
        return list(
            dict.fromkeys([*self.direct_image_urls, *self.context_image_urls])
        )

    def compose(self, plain_text: Optional[str] = None) -> str:
        current_text = self.plain_text if plain_text is None else plain_text
        parts = [block.strip() for block in self.context_blocks if block.strip()]
        if current_text.strip():
            if parts:
                parts.append(f"[当前消息]\n{current_text.strip()}")
            else:
                parts.append(current_text.strip())
        return "\n\n".join(parts)


@dataclass
class _ExpansionBudget:
    group_id: Optional[int] = None
    self_id: str = ""
    mention_calls_left: int = 2
    known_names: dict[str, str] = field(default_factory=dict)
    unavailable_images: bool = False
    nodes_left: int = 30
    chars_left: int = 6000
    images_left: int = 6
    api_calls_left: int = 4
    max_depth: int = 2

    def take_text(self, value: str, *, preserve_whitespace: bool = False) -> str:
        text = str(value or "")
        if not preserve_whitespace:
            text = text.strip()
        if not text or self.chars_left <= 0:
            return ""
        if len(text) > self.chars_left:
            suffix = "…[内容过长，已截断]"
            keep = max(0, self.chars_left - len(suffix))
            text = text[:keep] + suffix
        self.chars_left -= len(text)
        return text

    def take_image(self, url: Any) -> Optional[str]:
        value = str(url or "").strip()
        if (
            not value
            or not value.startswith(("http://", "https://"))
            or self.images_left <= 0
        ):
            self.unavailable_images = True
            return None
        self.images_left -= 1
        return value


def object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def segment_type_and_data(segment: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(segment, dict):
        seg_type = segment.get("type")
        data = segment.get("data")
    else:
        seg_type = getattr(segment, "type", None)
        data = getattr(segment, "data", None)
    return str(seg_type or ""), data if isinstance(data, dict) else {}


def as_segment_list(message: Any) -> list[Any]:
    if message is None:
        return []
    if isinstance(message, dict) and "type" in message:
        return [message]
    if isinstance(message, (list, tuple, Message)):
        return list(message)
    if isinstance(message, str):
        return list(Message(message))
    return []


def _image_url_from_data(data: dict) -> Optional[str]:
    for value in (data.get("url"), data.get("file")):
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


# Names are scoped to the bot and group: a member can use different cards in different groups.
_mention_names: OrderedDict[tuple, tuple[float, str]] = OrderedDict()


def _remember_sender(sender: Any, budget: _ExpansionBudget, *, cache: bool = False) -> None:
    user_id = str(object_value(sender, "user_id", ""))
    name = object_value(sender, "card", "") or object_value(sender, "nickname", "")
    if user_id and name:
        budget.known_names[user_id] = format_sender_name(str(name))
        if not cache:
            return
        key = (budget.self_id, budget.group_id, user_id)
        _mention_names[key] = (time.monotonic(), format_sender_name(str(name)))
        _mention_names.move_to_end(key)
        while len(_mention_names) > 2048:
            _mention_names.popitem(last=False)


async def _render_mention(bot: Bot, data: dict, budget: _ExpansionBudget) -> str:
    qq = str(data.get("qq", "")).strip()
    if qq == "all":
        return "@全体成员"
    if not qq:
        return "@某人"
    if qq == budget.self_id:
        return f"@你(QQ:{qq})"
    name = data.get("name") or data.get("nickname") or budget.known_names.get(qq)
    key = (budget.self_id, budget.group_id, qq)
    cached = _mention_names.get(key)
    if not name and cached and time.monotonic() - cached[0] < 3600:
        name = cached[1]
    if not name and budget.group_id and qq.isdigit() and budget.mention_calls_left > 0:
        budget.mention_calls_left -= 1
        try:
            member = await asyncio.wait_for(bot.call_api(
                "get_group_member_info", group_id=budget.group_id, user_id=int(qq),
            ), timeout=2.0)
            name = object_value(member, "card", "") or object_value(member, "nickname", "")
            if name:
                _remember_sender({"user_id": qq, "nickname": name}, budget, cache=True)
        except Exception as exc:
            logger.debug(f"[message_expand] 获取提及对象失败 qq={qq}: {exc}")
    return f"@{format_sender_name(str(name))}(QQ:{qq})" if name else f"@QQ:{qq}"


async def _render_current_text(bot: Bot, event: MessageEvent, budget: _ExpansionBudget) -> tuple[str, str]:
    """Keep mentions in memory; only bot addressing is omitted from the response input."""
    parts, response_parts = [], []
    for segment in as_segment_list(event.message):
        seg_type, data = segment_type_and_data(segment)
        if seg_type == "text":
            text = str(data.get("text", ""))
        elif seg_type == "at":
            text = await _render_mention(bot, data, budget) + " "
        else:
            continue
        parts.append(text)
        if seg_type != "at" or str(data.get("qq", "")) != budget.self_id:
            response_parts.append(text)
    return "".join(parts).strip(), "".join(response_parts).strip()


def _sender_label(sender: Any, event: MessageEvent) -> str:
    user_id = object_value(sender, "user_id", "")
    if user_id and str(user_id) == str(event.self_id):
        return "你"
    raw_name = (
        object_value(sender, "card", "")
        or object_value(sender, "nickname", "")
        or str(user_id or "")
    )
    return format_sender_name(str(raw_name)) if raw_name else "某人"


async def _call_message_api(
    bot: Bot,
    api: str,
    budget: _ExpansionBudget,
    **kwargs: Any,
) -> Any:
    if budget.api_calls_left <= 0:
        raise RuntimeError("消息展开 API 调用次数已达上限")
    budget.api_calls_left -= 1
    return await asyncio.wait_for(bot.call_api(api, **kwargs), timeout=3.0)


def _extract_forward_nodes(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("messages", "content"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    nested = payload.get("data")
    if isinstance(nested, dict):
        return _extract_forward_nodes(nested)
    return []


def _is_forward_json(data: dict[str, Any]) -> tuple[bool, list[str]]:
    raw = data.get("data")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return False, []
    elif isinstance(raw, dict):
        payload = raw
    else:
        return False, []

    config_data = payload.get("config") if isinstance(payload, dict) else {}
    app = str(payload.get("app", "")) if isinstance(payload, dict) else ""
    is_forward = bool(
        (isinstance(config_data, dict) and config_data.get("forward"))
        or app == "com.tencent.multimsg"
    )
    if not is_forward:
        return False, []

    meta = payload.get("meta") if isinstance(payload, dict) else {}
    detail = meta.get("detail", {}) if isinstance(meta, dict) else {}
    previews = []
    if isinstance(detail, dict):
        news = detail.get("news")
        if isinstance(news, list):
            for item in news:
                text = object_value(item, "text", "")
                if text:
                    previews.append(str(text).strip())
    return True, previews


async def _render_forward_payload(
    bot: Bot,
    payload: Any,
    budget: _ExpansionBudget,
    *,
    depth: int,
) -> tuple[str, list[str]]:
    nodes = _extract_forward_nodes(payload)
    if not nodes:
        return "", []

    # Some implementations embed one ordinary message chain directly rather
    # than wrapping it in forward-node objects.
    if all(
        segment_type_and_data(node)[0]
        and segment_type_and_data(node)[0] != "node"
        for node in nodes
    ):
        return await _render_message_value(bot, nodes, budget, depth=depth)

    lines: list[str] = []
    image_urls: list[str] = []
    for node in nodes[:budget.nodes_left]:
        seg_type, seg_data = segment_type_and_data(node)
        node_data = seg_data if seg_type == "node" else node
        sender = object_value(node_data, "sender", {}) or {}
        _remember_sender(sender, budget)
        _remember_sender({
            "user_id": object_value(node_data, "uin", ""),
            "nickname": object_value(node_data, "name", "") or object_value(node_data, "nickname", ""),
        }, budget)
    for node in nodes:
        if budget.nodes_left <= 0 or budget.chars_left <= 0:
            lines.append("…[后续转发节点已省略]")
            break
        budget.nodes_left -= 1

        seg_type, seg_data = segment_type_and_data(node)
        node_data = seg_data if seg_type == "node" else node
        sender = object_value(node_data, "sender", {}) or {}
        _remember_sender(sender, budget)
        nickname = (
            object_value(sender, "card", "")
            or object_value(sender, "nickname", "")
            or object_value(node_data, "name", "")
            or object_value(node_data, "nickname", "")
            or str(
                object_value(sender, "user_id", "")
                or object_value(node_data, "uin", "")
                or "某人"
            )
        )
        content = object_value(node_data, "content", None)
        if content is None:
            content = object_value(node_data, "message", None)

        rendered, nested_images = await _render_message_value(
            bot,
            content,
            budget,
            depth=depth,
        )
        rendered = rendered or "[空消息]"
        lines.append(f"{format_sender_name(str(nickname))}: {rendered}")
        image_urls.extend(nested_images)

    return "\n".join(lines), image_urls


async def _render_message_value(
    bot: Bot,
    message: Any,
    budget: _ExpansionBudget,
    *,
    depth: int,
) -> tuple[str, list[str]]:
    if isinstance(message, str):
        if "[CQ:" in message:
            try:
                parsed = Message(message)
                if any(segment.type != "text" for segment in parsed):
                    return await _render_message_value(
                        bot,
                        parsed,
                        budget,
                        depth=depth,
                    )
            except Exception as exc:
                logger.debug(f"[message_expand] CQ 字符串解析失败: {exc}")
        return budget.take_text(message), []

    segments = as_segment_list(message)
    if not segments:
        return budget.take_text(str(message or "")), []

    parts: list[str] = []
    image_urls: list[str] = []
    for segment in segments:
        if budget.chars_left <= 0:
            break
        seg_type, data = segment_type_and_data(segment)
        if seg_type == "text":
            text = budget.take_text(data.get("text", ""), preserve_whitespace=True)
            if text:
                parts.append(text)
        elif seg_type == "at":
            parts.append(budget.take_text(await _render_mention(bot, data, budget)))
        elif seg_type == "image":
            parts.append(budget.take_text("[图片]"))
            url = budget.take_image(_image_url_from_data(data))
            if url:
                image_urls.append(url)
        elif seg_type == "face":
            face_id = str(data.get("id") or data.get("emoji_id") or "")
            face_name = FACE_EMOJI_MAP.get(face_id, f"表情{face_id}")
            parts.append(budget.take_text(f"[收到QQ表情：{face_name}]"))
        elif seg_type in {"record", "voice"}:
            parts.append(budget.take_text("[语音]"))
        elif seg_type == "video":
            parts.append(budget.take_text("[视频]"))
        elif seg_type == "file":
            name = data.get("name") or data.get("file") or ""
            parts.append(budget.take_text(f"[文件: {name}]" if name else "[文件]"))
        elif seg_type == "reply":
            reply_id = data.get("id")
            if not reply_id or depth >= budget.max_depth:
                parts.append(budget.take_text("[回复了一条消息]"))
                continue
            try:
                info = await _call_message_api(
                    bot,
                    "get_msg",
                    budget,
                    message_id=int(reply_id),
                )
                quoted, nested_images = await _render_message_value(
                    bot,
                    object_value(info, "message", None),
                    budget,
                    depth=depth + 1,
                )
                parts.append(f"[回复: {quoted or '空消息'}]")
                image_urls.extend(nested_images)
            except Exception as exc:
                logger.debug(f"[message_expand] 展开嵌套回复失败 id={reply_id}: {exc}")
                parts.append(budget.take_text("[回复了一条无法读取的消息]"))
        elif seg_type == "forward":
            if depth >= budget.max_depth:
                parts.append(budget.take_text("[嵌套合并转发已省略]"))
                continue
            forward_id = data.get("id")
            payload = data.get("content")
            try:
                if not isinstance(payload, list) and forward_id:
                    payload = await _call_message_api(
                        bot,
                        "get_forward_msg",
                        budget,
                        id=str(forward_id),
                    )
                rendered, nested_images = await _render_forward_payload(
                    bot,
                    payload,
                    budget,
                    depth=depth + 1,
                )
                parts.append(
                    f"[嵌套合并转发]\n{rendered}"
                    if rendered
                    else "[无法读取嵌套合并转发]"
                )
                image_urls.extend(nested_images)
            except Exception as exc:
                logger.debug(f"[message_expand] 展开嵌套转发失败 id={forward_id}: {exc}")
                parts.append(budget.take_text("[无法读取嵌套合并转发]"))
        elif seg_type == "json":
            is_forward, previews = _is_forward_json(data)
            if is_forward:
                preview = "\n".join(previews)
                parts.append(
                    budget.take_text(
                        f"[合并转发预览]\n{preview}"
                        if preview
                        else "[合并转发卡片]"
                    )
                )
            else:
                parts.append(budget.take_text("[JSON 卡片]"))
        elif seg_type:
            parts.append(budget.take_text(f"[{seg_type}]"))

    return "".join(parts).strip(), image_urls


async def _resolve_reply_context(
    bot: Bot,
    event: MessageEvent,
    budget: _ExpansionBudget,
) -> tuple[Optional[str], list[str]]:
    reply_info = getattr(event, "reply", None)
    reply_id = None

    if reply_info is None:
        for source in (
            getattr(event, "original_message", None),
            getattr(event, "message", None),
        ):
            for segment in as_segment_list(source):
                seg_type, data = segment_type_and_data(segment)
                if seg_type == "reply":
                    reply_id = data.get("id")
                    break
            if reply_id:
                break

        if not reply_id:
            return None, []
        try:
            reply_info = await _call_message_api(
                bot,
                "get_msg",
                budget,
                message_id=int(reply_id),
            )
        except Exception as exc:
            logger.debug(f"[message_expand] get_msg 失败 id={reply_id}: {exc}")
            budget.unavailable_images = True
            return "[引用回复]\n原消息：[无法读取，可能已撤回]", []

    sender = object_value(reply_info, "sender", {}) or {}
    _remember_sender(sender, budget)
    replied_user_id = object_value(sender, "user_id", "")
    if replied_user_id and str(replied_user_id) == str(event.self_id):
        event.to_me = True
    name = _sender_label(sender, event)
    quoted, image_urls = await _render_message_value(
        bot,
        object_value(reply_info, "message", None),
        budget,
        depth=0,
    )
    return f"[引用回复]\n被回复者：{name}\n原消息：{quoted or '[空消息]'}", image_urls


async def _resolve_forward_contexts(
    bot: Bot,
    event: MessageEvent,
    budget: _ExpansionBudget,
) -> tuple[list[str], list[str]]:
    blocks: list[str] = []
    image_urls: list[str] = []

    for segment in as_segment_list(getattr(event, "message", None)):
        seg_type, data = segment_type_and_data(segment)
        if seg_type == "forward":
            forward_id = data.get("id")
            payload = data.get("content")
            try:
                if not isinstance(payload, list) and forward_id:
                    payload = await _call_message_api(
                        bot,
                        "get_forward_msg",
                        budget,
                        id=str(forward_id),
                    )
                rendered, nested_images = await _render_forward_payload(
                    bot,
                    payload,
                    budget,
                    depth=1,
                )
                blocks.append(
                    f"[合并转发]\n{rendered}"
                    if rendered
                    else "[合并转发]\n[无法读取内容]"
                )
                image_urls.extend(nested_images)
            except Exception as exc:
                logger.debug(f"[message_expand] get_forward_msg 失败 id={forward_id}: {exc}")
                blocks.append("[合并转发]\n[无法读取内容]")
        elif seg_type == "json":
            is_forward, previews = _is_forward_json(data)
            if not is_forward:
                continue
            rendered = ""
            nested_images: list[str] = []
            message_id = getattr(event, "message_id", None)
            if message_id is not None:
                try:
                    payload = await _call_message_api(
                        bot,
                        "get_forward_msg",
                        budget,
                        message_id=str(message_id),
                    )
                    rendered, nested_images = await _render_forward_payload(
                        bot,
                        payload,
                        budget,
                        depth=1,
                    )
                except Exception as exc:
                    logger.debug(
                        f"[message_expand] JSON 转发展开失败 message_id={message_id}: {exc}"
                    )
            if not rendered:
                rendered = "\n".join(
                    text for text in (budget.take_text(item) for item in previews) if text
                )
            blocks.append(
                f"[合并转发]\n{rendered}"
                if rendered
                else "[合并转发]\n[只能识别到转发卡片，无法读取内容]"
            )
            image_urls.extend(nested_images)

    return blocks, image_urls


async def resolve_incoming_message(
    bot: Bot,
    event: MessageEvent,
) -> ResolvedIncomingMessage:
    """Expand OneBot reply/forward data once and cache it on the event."""
    cached = getattr(event, "_llm_resolved_incoming_message", None)
    if isinstance(cached, ResolvedIncomingMessage):
        return cached

    budget = _ExpansionBudget(group_id=getattr(event, "group_id", None), self_id=str(event.self_id))
    _remember_sender(getattr(event, "sender", {}), budget, cache=True)
    reply_block, reply_images = await _resolve_reply_context(bot, event, budget)
    forward_blocks, forward_images = await _resolve_forward_contexts(
        bot,
        event,
        budget,
    )
    plain_text, response_text = await _render_current_text(bot, event, budget)
    direct_images = await get_images_from_message(event)
    unavailable_direct = any(
        seg_type == "image" and not _image_url_from_data(data)
        for seg_type, data in map(segment_type_and_data, as_segment_list(event.message))
    )
    result = ResolvedIncomingMessage(
        plain_text=plain_text,
        response_text=response_text,
        context_blocks=[
            *([reply_block] if reply_block else []),
            *forward_blocks,
        ],
        direct_image_urls=direct_images,
        context_image_urls=[*reply_images, *forward_images],
        unavailable_images=budget.unavailable_images or unavailable_direct,
        faces=await get_faces_from_message(event),
    )
    try:
        setattr(event, "_llm_resolved_incoming_message", result)
    except Exception as exc:
        logger.debug(f"[message_expand] 无法缓存规范化事件: {exc}")
    return result


async def get_images_from_message(event: MessageEvent) -> list[str]:
    """从消息中提取图片URL"""
    images = []
    for seg in as_segment_list(event.message):
        seg_type, data = segment_type_and_data(seg)
        if seg_type == "image":
            url = _image_url_from_data(data)
            # 过滤无效URL（QQ内置表情等）
            if url:
                images.append(url)
    return images


# QQ 内置表情映射（常见表情）
FACE_EMOJI_MAP = {
    "0": "微笑", "1": "偷笑", "2": "大笑", "3": "抠鼻", "4": "擦汗",
    "5": "鼓掌", "6": "啤酒", "7": "瓢虫", "8": "睡", "9": "爱心",
    "10": "玫瑰", "11": "小星星", "12": "月亮", "13": "太阳", "14": "奖杯",
    "15": "蛋糕", "16": "炸弹", "17": "刀", "18": "足球", "19": "便便",
    "20": "咖啡", "21": "音乐", "22": "灯泡", "23": "纸巾", "24": "药水",
    "25": "口罩", "26": "电话", "27": "时钟", "28": "包裹", "29": "书本",
    "30": "铅笔", "31": "考试", "32": "画板", "33": "追", "34": "期待",
    "35": "甩", "36": "得意", "37": "恼火", "38": "石化", "39": "擦汗",
    "40": "无奈", "41": "流汗", "42": "呕吐", "43": "抓狂", "44": "囧",
    "45": "尴尬", "46": "微笑", "47": "难过", "48": "伤心", "49": "哭了",
    "50": "生气", "51": "白眼", "52": "左哼", "53": "右哼", "54": "哈欠",
    "55": "折磨", "56": "悠闲", "57": "惊讶", "58": "疑问", "59": "冻僵",
    "60": "恐惧", "61": "发呆", "62": "挥手", "63": "再见", "64": "亲亲",
    "65": "抱抱", "66": "举爪", "67": "商务", "68": "NO", "69": "OK",
    "70": "握手", "71": "献花", "72": "献玫瑰", "73": "差评", "74": "好评",
    "75": "加油", "76": "点赞", "77": "weak", "78": "good", "79": "抱拳",
    "80": "查看", "81": "发怒", "82": "爱心", "83": "心碎", "84": "允悲",
    "85": "托腮", "86": "吃瓜", "87": "加油", "88": "打call", "89": "崇拜",
    "90": "比心", "91": "点赞", "92": "社会社会", "93": "旺柴", "94": "好的",
    "95": "打工人", "96": "加油打工人", "97": "我裂开了", "98": "摊手",
    "99": "ohhh", "100": "呜呜", "101": "好累", "102": "药丸", "103": "老爷爷",
    "104": "吸入", "105": "抛出", "106": "妙啊", "107": "梭哈", "108": "精神",
    "109": "蔫了", "110": "嗯嗯", "111": "no", "112": "甩锅", "113": "无奈",
    "114": "嗨呀", "115": "谢谢你", "116": "夸夸", "117": "么么哒", "118": "收下",
    "119": "比心", "120": "胡瓜", "121": "发箍", "122": "二哈", "123": "雕塑",
    "124": "铃铛", "125": "便捷", "126": "便捷", "127": "二维码", "128": "文件夹",
    "129": "芯片", "130": "放大镜", "131": "云", "132": "及时雨", "133": "太阳",
    "134": "小中风", "135": "脑瓜疼", "136": "吃瓜", "137": "996", "138": "加油",
}


async def get_faces_from_message(event: MessageEvent) -> list[str]:
    """从消息中提取QQ内置表情"""
    faces = []
    for seg in event.message:
        if seg.type == "face":
            emoji_id = seg.data.get("id") or seg.data.get("emoji_id")
            face_name = FACE_EMOJI_MAP.get(emoji_id, f"表情{emoji_id}")
            faces.append(face_name)
    return faces




def clear_mention_cache() -> None:
    """Release cached names when the chat lifecycle stops."""
    _mention_names.clear()
