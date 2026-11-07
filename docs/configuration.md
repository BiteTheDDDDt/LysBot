# 配置与运行

在 `.env` 配置 `MASTER_QQ`、OneBot 的 `ACCESS_TOKEN`、`MODEL_GENERAL` 与 provider 凭据。master 在启动前映射为唯一的 NoneBot superuser，旧的 `SUPERUSERS` 不会扩大管理员范围。

## 模型

模型配置写作 `provider:model_id`，备用链使用 ` | ` 分隔。当前适配器包括 `openai`、`deepseek`、`dashscope`、`google`、`anthropic`、`nvidia`、`openrouter`、`siliconflow`、`tencentmaas`、`volc`、`xiaomimimo` 和 `cliproxyapi`。具体模型可用性取决于账号与上游服务；不需要安装 CLIProxyAPI 才能运行。

模型系列、免费/试用/订阅/按量计费的区别及专用能力白名单见 [provider 指南](providers.md)。已有可用 Codex 账号时，优先考虑 [CLIProxyAPI 部署方案](cliproxyapi.md)，Google TTS 单独接入。全部 provider 的凭据名和默认地址见 [.env.example](../.env.example)。

- `MODEL_GENERAL` 必填，负责基础聊天、规划、摘要、记忆提取及普通文本任务。
- `MODEL_CHAT`、`MODEL_PLANNER`、`MODEL_SUMMARY`、`MODEL_IMPORTANT_INFO` 为可选覆盖，未填继承 general。
- `MODEL_FALLBACK` 可不填；填写后追加到普通文本任务的候选链。
- `MODEL_SEARCH`、`MODEL_VISION`、`MODEL_TTS`、`MODEL_IMAGE_GEN` 未填即无对应能力；专用模型只使用各自的候选链。
- 搜索实现支持 DashScope 原生搜索及已适配的 GPT Responses 原生搜索；当前不是任意模型都能开启联网。
- 识图仍有协议能力白名单，见 `providers.py`；语音当前使用 Google 原生 TTS，生图使用 OpenAI/CLIProxyAPI Images API。

显式配置了错误 provider、协议不支持的型号或无可用凭据时，启动会报告配置错误。未配置的能力从工具目录移除，后台入口和执行入口也不发起对应模型调用。

## 候选链与自动 fallback

每个 `MODEL_*` 都可以在同一行用 ` | ` 配置多个候选，左边优先。调用失败时按顺序尝试后面的模型，可以跨 provider：

```dotenv
MODEL_SUMMARY="cliproxyapi:gpt-6-luna | google:gemini-3.5-flash-lite"
MODEL_FALLBACK="google:gemini-3.5-flash-lite | openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"
```

该例的摘要候选顺序为：GPT → Google → Google（全局备用）→ OpenRouter。链中重复项保留，表示再次尝试；不会自动去重。没有配置凭据、能力不匹配或仍在接口冷却期的候选会跳过。整个任务的超时预算耗尽或被取消后停止，不保证每次都尝试完整条链。

- `MODEL_GENERAL`、`MODEL_CHAT`、`MODEL_PLANNER`、`MODEL_SUMMARY`、`MODEL_IMPORTANT_INFO` 等普通文本角色：自身候选链后追加 `MODEL_FALLBACK`。
- 未单独填写的 chat、planner、summary、important_info：继承 `MODEL_GENERAL` 的完整候选链，再追加全局备用。
- `MODEL_SEARCH`、`MODEL_VISION`、`MODEL_TTS`、`MODEL_IMAGE_GEN`：只尝试各自显式配置的链，不追加 `MODEL_FALLBACK`。每个候选都要符合该专用能力的协议限制。

完整多模型组合见 [.env.models.example](../.env.models.example)：GPT 用于 planner，多 provider chat 按顺序回退，识图也配置多个候选，TTS 单独使用 Google。示例只包含模型路由，需要与 [.env.example](../.env.example) 的 QQ 配置及 provider 凭据合并使用。

修改 `.env` 后用 `docker compose up -d --force-recreate bot` 应用，再在 QQ 用 `/status` 检查实际角色链。`/model` 只临时覆盖当前会话的聊天模型，不替代持久配置。

## 人设和群聊行为

`PERSONA_FILE` 默认是 `personas/phosphophyllite.toml`，文件包含角色名称、性格与口吻、外貌和声音。通用执行规则由程序维护。更换人设后重启 bot。

主动接话、复读、表情、冷群梗图、jrrp 和戳一戳保留为内置互动行为。静音命令和掉线邮件已从新版移除。

`ENABLE_DAILY_BRIEFING`、`ENABLE_DAILY_NEWS_BROADCAST`、`ENABLE_MODEL_CARD` 默认为 false。新闻要求前两项同时开启并配置搜索模型；群像仅在生图可用时生成。配置冲突会在启动时说明。

## 本地检索

模型固定为 BGE small 中文 ONNX 工件，512 维，固定 revision、SHA-256 和分段策略。不需要 embedding API key，不支持通过环境变量更换模型身份。`scripts/prepare_local_embedding.py` 准备文件；运行时只读本地权重。

向量、FTS5、原文、摘要和来源关系均保存在 SQLite 中。缺少权重或索引未就绪时保留关键词查询。模型身份或策略在将来版本变化时需要重建索引，不能把同维度视作兼容。

默认数据目录是 `data/`。Compose 把它挂载到 `/app/data`，权重在 `data/models/`；可以通过 `LLM_CHAT_DATABASE_PATH` 继续使用旧部署的独立数据库卷。查询状态：

```bash
docker compose exec bot python scripts/agentic_search_ops.py status --database /app/data/history.db
```

实际数据库在其他位置时替换参数。在运行库所在的系统内执行查询与 SQLite backup，避免跨 Windows/Linux 同时打开一个 WAL 数据库。

## 命令

完整命令、别名、参数、权限及作用范围见 [QQ 使用手册](commands.md)。
