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

可选的 `image_reference_document` 指向知识库根目录下的相对 Markdown 路径。生成当前角色的自拍或装扮时，工具默认加载该文档前两张可用的原图，仍受参考图数量限制。用户附图和显式选择的图片引用优先；显式传空列表可以取消默认参考。

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

## 静态知识库

动态知识库来自实际群聊、摘要与记忆，使用 `recall_history/read_history/query_history` 检索。静态知识库由部署者预设，使用 `search_knowledge/read_knowledge` 检索；两者并列，按问题需要结合，不互相覆盖。

planner 常驻的元信息只有固定存储范围：DB 包含聊天原文及作者、时间、引用关系、摘要、长期记忆与来源、生成媒体记录；knowledge 默认分为项目信息、背景知识、skill 方法。可通过 `KNOWLEDGE_OVERVIEW` 用一句话说明自己部署的静态知识类别。新增文档无需更新概述，只有类别变化时再改；不会自动扫描生成目录、统计数据量或时间覆盖，也不注入正文，具体资料和变化交给实际检索确认。

默认从 `knowledge/` 读取 UTF-8 Markdown／文本文件，包含 LysBot 项目信息。可让 AI 在 `knowledge/local/` 添加领域资料、术语或 skill，例如：

```text
knowledge/
├── project.md
└── local/
    ├── 游戏攻略.md
    └── skills/
        └── 排查问题.md
```

文件用 `# 标题` 开头，正文说明适用范围、事实或步骤即可，不要求特殊元数据格式。skill 是可检索的方法资料，不会注册新工具、执行脚本或授予管理员权限。默认忽略自定义资料，不提交 Git；内置 `project.md` 仍受版本控制。

Compose 已把 `./knowledge` 只读挂载到 `/app/knowledge`。本地 Python 默认读取当前工作目录的 `knowledge`。可用 `KNOWLEDGE_DIR` 指定容器内／进程内的目录；使用自有宿主目录时，由 AI 修改 bot 的 Compose 挂载源并保留只读模式。设置 `KNOWLEDGE_DIR=` 可同时关闭两个静态知识工具。

角色资料可在 Markdown 中用独立一行 `![早期外观参考](phos.png)` 关联同目录或子目录的本地图像。每份文档最多读取前三个图片链接，支持 PNG、JPEG、WebP，单图上限 4 MiB。`read_knowledge` 返回 `image_refs`，其 `id` 可以放入 `generate_image.reference_image_ids`，系统沿用参考图生图接口上传原图。角色自拍会优先查找匹配的设定资料和参考图；用户明确指定的原图、服装、动作及场景按本轮要求处理。

**此目录中的资料会向所有群和私聊开放，并可能发送到配置的模型服务。** 不放 `.env`、账号资料、聊天备份或仅供某个群访问的内容；这类内容不属于共享静态知识库。工具拒绝隐藏文件、目录穿越、符号链接与目录联接，不接受模型传入任意系统路径。

搜索采用本地关键词，不调用 embedding 或联网服务。空查询可分页浏览目录；搜索返回标题、片段和带内容版本的 `document_id`。读取正文时用该 ID，长文按 `next_offset` 继续。文件修改后下一次搜索生效，无需重建索引；旧 ID 会返回版本变化并要求重新搜索，跨轮使用的静态来源也会回读校验。

每次最多扫描 4096 个目录项、512 篇文档和 8 MiB 文本，单文件上限 256 KiB；搜索每页最多 20 条、读取每页最多 6000 字符。超限、文件不可读或无效编码会返回 partial／warning，不把扫描不完整当作没有资料。适合项目说明和小型知识集，大批资料可先按主题整理。

## 命令

完整命令、别名、参数、权限及作用范围见 [QQ 使用手册](commands.md)。

## 群聊参与与回复长度

planner 按语境选择 `tiny`（极短反应，一个词或两个短词组合）、`sentence`（普通短句）或 `detailed`（需要展开的回答）。极短模式用于接梗、惊讶、确认等单个反应，倾向不带标点，不强制凑满字数，也不在发送时裁剪输出。

未引用、未 @ 的接话需承接 bot 的具体内容，同一个人继续同一话题不自动构成追问。轻微吐槽可以沉默、发表情或极短回应，不默认生成正式认错。主动规划后若发现没有适合发送的内容，可以放弃本轮参与：这是发送前保持旁听，不是撤回已发送消息；直接请求仍会回应或必要澄清。

每日简报的“今日群友画像”根据本人当天的说话方式、互动和具体表现动态生成人格标签与短评，不使用固定模型或人格分类。来源继续绑定本人当日消息，标签不写回长期人物记忆。

群像为动画风横图，个人画像为黑白漫画横图，两块都以 1600 × 900 的 16:9 画幅嵌入。个人画像最多四位，按有效人数一次生成，一人一格，分镜可斜切或大小错落。短评用于画面构思，画内放昵称和标签。两次生图各有独立超时；画像生成失败或返回不适合横幅的比例时，简报使用文字画像。

`BRIEFING_REFERENCE_DOCUMENTS` 可选，使用 JSON 对象配置 `ensemble`（多人剧照）、`interaction`（互动剧照）和 `comic`（漫画分镜）。每个值是相对 `KNOWLEDGE_DIR` 的 Markdown 文档路径数组，复用知识库的本地图片附件。同一集的两类剧照文档使用相同文件名，选图时会避开同集配对。每次最多传两张，同一群同一天的重试复用选择。
