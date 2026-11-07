"""Static capability declarations shared by tool exposure and execution policy.

No provider clients, database connections or plugin lifecycle are created here.
Dynamic context checks and resource limits remain in the existing implementations.
"""

from dataclasses import dataclass
from .execution import ExecutionPolicy


@dataclass(frozen=True)
class Capability:
    schema: dict
    read_only: bool = False
    media: bool = False
    memory_scoped: bool = False
    agentic_only: bool = False
    runner_managed: bool = False

    @property
    def name(self):
        return self.schema["function"]["name"]

    @property
    def execution(self):
        from .config import config
        from .network_retry import retry_budget
        return ExecutionPolicy(
            self.name, permission="current" if self.read_only else "send",
            timeout=lambda: float(config.chat_media_turn_timeout if self.media else
                                  retry_budget(config.agent_tool_timeout) if self.name == "search_web" else config.agent_tool_timeout),
            check_result=self.read_only,
        )


SEARCH_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "搜索公开资料，返回真实来源 ID、URL、标题和片段。由搜索模型使用原生联网能力完成查证，返回结论、来源和未确认项；有具体缺口时补充搜索，不为复核调用 read_web。适用于公开昵称、作者/选手的身份、作品和经历；群聊中的公开活动线索可用于消歧，群内转述不代替公开来源。"
        "只发送查证必要的公开主题，不把整段群聊或群成员私人信息发送给搜索引擎。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索的关键短语，必须简短精准，比如 '2024年3月国际金价 实时报价' 或 '今日天气 深圳'",
                },
                "domains": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                "freshness_days": {"type": "integer", "minimum": 1, "maximum": 365},
                "limit": {"type": "integer", "minimum": 1, "maximum": 15},
                "image_refs": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                               "description": "辨认图片角色、物品或出处时附上当前会话的图片引用ID，例如 img_…；搜索模型会直接看原图再联网核对，不必只靠外观描述猜测。"},
            },
            "required": ["query"],
        },
    },
}

POKE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_poke",
        "description": "当用户要求你戳一戳某人、或者你想用戳一戳的方式提醒/调侃/互动某人时使用此工具。戳一戳是QQ的一种轻量级互动方式，会向对方发送一个震动提醒。",
        "parameters": {
            "type": "object",
            "properties": {
                "target_id": {
                    "type": "string",
                    "description": "要戳一戳的目标用户QQ号，可以是具体的QQ号（数字）或者 'sender' 表示戳发送消息的人",
                },
                "reason": {
                    "type": "string",
                    "description": "戳一戳的原因或场景，比如 '提醒回消息'、'调侃'、'打招呼' 等，这将影响你的回复语气",
                },
            },
            "required": ["target_id"],
        },
    },
}

VOICE_INTENT_POLICY = (
    "结合语境判断声音意图：点播、配音、来一段或‘为某人播放某某’，若是在邀请 bot 用声音接梗，"
    "应使用 voice/send_voice，不要求用户说出‘TTS’或‘生成语音’。例如在出行玩笑中‘为某人播放 See You Again’，"
    "若有出行/送别等玩梗上下文且用户没有明确要求原曲，选择 voice/send_voice，用一小段贴合送别语境的原创台词生成语音。"
    "这类请求的核心是用声音接梗，不要仅因出现歌名就推断必须播放原曲、改成文字解释不能放歌。"
    "reply_hint 保留对象、梗和声音意图，不必预写固定台词。"
    "现有工具只能合成语音，不能播放原曲、视频或已有音频，也不保证演唱；不能把歌名念一遍冒充放歌，"
    "不能声称正在播放原曲。明确要求原曲/视频，或仅讨论歌曲、播放器时，不以 TTS 偷换请求；"
    "必要时简短说明具体能力边界，不为普通接梗附加免责声明。"
)


VOICE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_voice",
        "description": VOICE_INTENT_POLICY + "当用户明确索要语音，或表达只有听到你的声音才能满足的自然意图时使用。例如‘想听你的声音’、‘叫我一声宝宝’、‘说句晚安给我听’、‘哄我睡觉’都应发送语音，即使用户没有说出‘生成语音’四个字。普通的‘你说说’、‘告诉我’仍按文字回复。调用后系统会直接生成并发送一条QQ语音；不要用于普通文字回复，也不要擅自把每条回复都转成语音。成功后系统会把真实执行结果交还给你；是否补充文字由本轮回复规划决定，不要重复朗读内容。",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "真正要朗读的文本，不要加入发送说明或引号。"},
                "emotion": {
                    "type": "string",
                    "enum": ["neutral", "cheerful", "sad", "angry", "whisper", "excited", "sarcastic"],
                    "description": "朗读语气，默认 neutral。",
                },
            },
            "required": ["text"],
        },
    },
}

IMAGE_GENERATION_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "当用户明确要求画图，或自然地索要一个需要视觉作品才能满足的画面时使用。例如‘宝宝我要看你的自拍’、‘给我看看你长什么样’、‘来张你的照片’、‘想看你穿汉服的样子’都应生成图片，即使用户没有说出‘生成图片’四个字；此时不要回答自己没有自拍或照片。仅讨论图片、评价已有图片、询问会不会画画时不要调用。调用后系统会立即生成并发送一张QQ图片；不要用于普通聊天，也不要擅自替用户画图。成功后系统会把真实画面描述交还给你；是否补充文字由本轮回复规划决定，不要机械地说‘画好了’，也不要完整复述图片描述。",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "完整、具体的画面描述，忠实保留用户指定的主体、风格、构图和文字，不要添加用户未要求的角色或元素。用户索要你的自拍、照片、画像或某种装扮时，主体就是当前人设，外貌使用角色配置，结合用户指定的服装、动作、场景和构图。",
                },
                "orientation": {
                    "type": "string",
                    "enum": ["auto", "square", "landscape", "portrait"],
                    "description": "图片方向：自动、方图、横图或竖图，默认 auto。",
                },
                "reference_image_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "参考或修改已有图片时，选择上下文中的图片引用 "
                    "ID，按参考顺序填写。prompt "
                    "写修改要求和需保留的内容，系统会传入原图，无需凭描述重建。省略默认使用本轮附图/引用图；仅当用户要求全新绘图时传 "
                    "[]。使用历史图片必须明确填写 "
                    "ID。原图不可用时请用户引用或重发，不要改为纯文字重画。",
                },
            },
            "required": ["prompt"],
        },
    },
}

RECALL_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "recall_history",
        "description": "搜索当前群/私聊的历史，用于群内称呼对应、发言和事件依据；公开人物介绍缺口可转 search_web，不能仅因群聊提到此人就限于历史。自己决定查询方式、人物与时间条件；根据结果改写查询或用 read_history 读原文。hybrid "
        "混合关键词和向量；keyword 查原话/型号；semantic 查近义概念；timeline 按时间浏览；aggregate 对 messages "
        "做精确计数，可按 author/day/role "
        "分组。摘要和旧提取只作线索。author_ids=谁说的，subject_ids=重要信息说的是谁；同名须先消歧。不得把计划、转述当作当前事实。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "自然语言语义问题或少量关键词；timeline/aggregate 可留空。"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["messages", "summaries", "important_info", "generated_media"]},
                    "description": "检索来源；默认 messages、summaries、important_info。",
                },
                "sender_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "按历史昵称模糊筛选，可选；同名时不如 sender_ids 可靠。",
                },
                "since": {
                    "type": "string",
                    "description": "起始时间，相对值如 '24h' / '7d' / '30d'，或绝对 ISO 日期如 '2025-01-01'。可选。",
                },
                "until": {
                    "type": "string",
                    "description": "截止时间，相对值或绝对 ISO 日期。用于限定某天/某段时间，可选。",
                },
                "roles": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["user", "assistant", "system"]},
                    "description": "限定角色；查群友用 user，查 Bot 旧回复用 assistant。",
                },
                "match_mode": {
                    "type": "string",
                    "enum": ["all", "any"],
                    "description": "all=所有关键词都命中，any=命中任意关键词；默认 all。",
                },
                "sort": {
                    "type": "string",
                    "enum": ["relevance", "newest", "oldest"],
                    "description": "排序方式；默认 relevance。",
                },
                "limit": {
                    "type": "integer",
                    "description": "每页结果数，默认 8，最多 30。",
                    "minimum": 1,
                    "maximum": 30,
                },
                "mode": {"type": "string", "enum": ["hybrid", "keyword", "semantic", "timeline", "aggregate"]},
                "author_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "subject_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "exact_phrase": {"type": "string", "description": "按原文连续字串精确匹配，适合型号和原话。"},
                "cursor": {"type": "string", "description": "上次相同查询返回的 next_cursor；其余参数保持一致。"},
                "group_by": {"type": "string", "enum": ["author", "day", "role"]},
            },
            "required": [],
        },
    },
}

READ_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "read_history",
        "description": "按检索返回的 ID 读取原文与来源链；给原消息 anchor_id 可展开前后对话。保留每条说话人与时间。",
        "parameters": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 30,
                    "description": "检索返回的完整引用 ID；SQL 查询用 source_id 列，例如 "
                    "h:0:messages:123，不能使用数字 id。",
                },
                "anchor_id": {"type": "string"},
                "before": {"type": "integer", "minimum": 0, "maximum": 20},
                "after": {"type": "integer", "minimum": 0, "maximum": 20},
                "cursor": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30},
            },
            "required": [],
        },
    },
}

READ_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "read_web",
        "description": "仅阅读用户明确提供的链接，使用用户地址目录中的来源 ID。不得打开搜索返回的来源来复核；公开资料查证由 search_web 完成。需要时分页继续。",
        "parameters": {
            "type": "object",
            "properties": {"source_id": {"type": "string"}, "focus": {"type": "string"}, "cursor": {"type": "string"}},
            "required": ["source_id"],
        },
    },
}

QUERY_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_history",
        "description": "自由编写 SQLite SELECT：支持 JOIN、CTE、子查询、窗口函数、JSON、自定义分组统计。仅当前会话视图，自动隔离，无需群号。sql "
        "留空获取字段说明和示例。messages(id,source_id,sender_id,name,role,content,timestamp)；summaries(id,source_id,content,time_range,created_at,provenance_json)；facts(id,source_id,user_id,info_type,content,created_at,provenance_json)。参数可用 "
        ":name 或 ?。语义查询先 recall_history，再结合发现的 ID、人物、时间自由查询。SQL 结果用 columns+rows 返回；选择 "
        "source_id 列时自动核验并附最多 8 条原消息短片段到 items，更多原文用 read_history。语法错误会返回原因，可修正重查；耗时 8 "
        "秒，长结果请自行缩小条件和稳定排序翻页。",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "一条只读 SQLite 查询；空字符串返回逻辑表结构。"},
                "parameters": {
                    "type": "string",
                    "description": "可省略。JSON 编码的命名参数对象或位置参数数组，例如 "
                    '{"kind":"hardware"}；值为字符串、数值、布尔值或 '
                    "null。",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "最多返回多少行，默认 30；不改变聚合的统计范围。",
                },
            },
            "required": [],
        },
    },
}

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_memory",
            "description": "查看当前群某人的记忆和变更记录，或某次回答注入/检索/引用的依据。追问刚才为何这样说时不传参数；run_id "
            "来自工作上下文。旧记录仅供审计，不能当作当前事实。",
            "parameters": {
                "type": "object",
                "properties": {"owner_id": {"type": "integer"}, "run_id": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revise_memory",
            "description": "仅用户明确要求纠正/撤回记忆时使用。preview 指定一条 important_info 引用及变更；先展示前后内容与 "
            "confirm_text，用户发送后 apply。undo "
            "按操作ID撤销，需用户发送撤销口令。本人只能改自己的记忆，管理员可改其他人。归属不是原消息作者；转述不能当成本人确认。普通聊天和玩笑不写入纠正。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["preview", "apply", "undo"]},
                    "ref": {"type": "string"},
                    "content": {"type": "string"},
                    "owner_id": {"type": "integer"},
                    "assertion": {
                        "type": "string",
                        "enum": ["asserted", "negated", "planned", "reported", "uncertain", "withdrawn"],
                    },
                    "reason": {"type": "string"},
                    "operation_id": {"type": "string"},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    },
]

PROGRESS_TOOL = {
    "type": "function",
    "function": {
        "name": "report_progress",
        "description": "长检索中的可选进展通知。提供已取得的来源ID和简短进展，不能提前宣布最终结论。系统限频，无需反复尝试。",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}, "source_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["summary", "source_ids"],
            "additionalProperties": False,
        },
    },
}

CAPABILITIES = {
    cap.name: cap
    for cap in (
        Capability(SEARCH_WEB_TOOL, read_only=True),
        Capability(POKE_TOOL),
        Capability(VOICE_TOOL, media=True),
        Capability(IMAGE_GENERATION_TOOL, media=True),
        Capability(RECALL_HISTORY_TOOL, read_only=True, memory_scoped=True),
        Capability(READ_HISTORY_TOOL, read_only=True, memory_scoped=True, agentic_only=True),
        Capability(READ_WEB_TOOL, read_only=True, agentic_only=True),
        Capability(QUERY_HISTORY_TOOL, read_only=True, memory_scoped=True, agentic_only=True),
        Capability(MEMORY_TOOLS[0], read_only=True, memory_scoped=True, agentic_only=True),
        Capability(MEMORY_TOOLS[1], agentic_only=True),
        Capability(PROGRESS_TOOL, runner_managed=True),
    )
}
READ_TOOLS = {name for name, cap in CAPABILITIES.items() if cap.read_only}
MEDIA_TOOLS = {name for name, cap in CAPABILITIES.items() if cap.media}
MEMORY_READ_TOOLS = {name for name, cap in CAPABILITIES.items() if cap.memory_scoped}
BASE_CHAT_TOOLS = [cap.schema for cap in CAPABILITIES.values() if not cap.media]
MEDIA_CHAT_TOOLS = {name: cap.schema for name, cap in CAPABILITIES.items() if cap.media}
AVAILABLE_TOOLS = [cap.schema for cap in CAPABILITIES.values()]


def bind_handlers(namespace):
    """Fail at startup if a declared ordinary capability lacks its implementation."""
    handlers = {}
    for name, cap in CAPABILITIES.items():
        if cap.runner_managed:
            continue
        handler = namespace.get(name)
        if not callable(handler):
            raise ValueError(f"Missing capability handler: {name}")
        handlers[name] = handler
    return handlers


def capability_available(name):
    from .roles import role_available
    role = {"search_web": "search", "send_voice": "tts", "generate_image": "image_gen"}.get(name)
    return role_available(role) if role else True
