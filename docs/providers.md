# Provider、模型能力与费用

本页介绍 LysBot 支持的 provider、模型能力和费用。具体型号与额度以服务商目录和账号权限为准。

## 怎么选

优先考虑 **CLIProxyAPI + Codex OAuth**：已有可用 Codex 账号时，用一个本地网关承接聊天、规划、摘要、识图，以及账号支持的原生搜索与生图。语音另配 Google API key。部署步骤见 [CLIProxyAPI 指南](cliproxyapi.md)。不需要语音就不配 Google；先只配置 `MODEL_GENERAL` 也能跑。

推荐顺序是本项目的部署选型倾向：

1. **CLIProxyAPI**：优先复用 Codex 套餐；还可接 Grok、Antigravity 的可用免费模型。Antigravity 可以提供 Claude / Gemini，额度较少时优先考虑放在 `MODEL_CHAT`，把有限额度用于最终表达。
2. **NVIDIA**：有免费开发体验模型；可用性和延迟波动很大，选型时要实测，不能只看模型本身的能力。
3. **OpenRouter**：不定期有短期免费模型，也有一些较长期提供的免费型号，适合试用与备用；以实时免费目录为准。
4. **DashScope / 腾讯 TokenHub / 火山方舟**：同一推荐层级，可按所选模型开通或领取试用额度，适合逐个体验模型；有效期、账号内共享关系和用尽后行为分别核对控制台。

其他已接入服务列在表格后半部分，已有 API key 时也可直接使用。免费层、试用额度、订阅权益是不同概念，群聊高频运行前应核对限额。

## 全部 provider

“LysBot 能力”描述当前适配器，**不是服务商全部产品能力**。所有通用模型都需要实际支持工具调用。具体凭据与默认 URL 已逐项写在 [.env.example](../.env.example)。

| provider | 模型选择方向 / 示例 | LysBot 当前能力 | 免费与付费方式、官方入口 |
| --- | --- | --- | --- |
| `cliproxyapi` | Codex 套餐中的 GPT（如 `gpt-6-luna`）、GPT Image；也可接 Grok 与 Antigravity 的 Claude / Gemini | 文本；白名单识图、GPT 原生搜索、GPT Image 生图/编辑 | 首选账号授权网关：Codex 使用套餐额度；Grok / Antigravity 有部分免费可用模型，Antigravity 额度较少，可优先分配给 chat。精确型号与权益以账号为准。[上游项目](https://github.com/router-for-me/CLIProxyAPI)、[Antigravity 型号与套餐](https://antigravity.google/docs/models/)、[Codex 计费](https://learn.chatgpt.com/docs/pricing) |
| `nvidia` | NIM 目录中的 Kimi / Qwen / Nemotron / Gemma 等，以目录完整 ID 为准 | 文本；Gemma 或已适配 Nemotron Omni 识图 | 有免费开发体验模型。可用性和延迟非常不稳定，建议实测并准备备用路由；生产使用另核对许可/订阅，自部署有 GPU 等成本。[NIM FAQ](https://docs.api.nvidia.com/nim/docs/product)、[NIM 提供方式](https://docs.nvidia.com/nim/large-language-models/latest/day-0/index.html) |
| `openrouter` | 多家模型；ID 含厂商前缀；目录中实际存在的 `:free` 变体或 `openrouter/free` | 文本、工具调用；未接其搜索/视觉/生图专用能力 | 不定期有短期免费模型，也有较长期提供的免费型号；免费模型单独限流，部分需购入 credits 才有较高配额；付费模型充值 credits 后按实际用量扣除，购买 credits 有平台手续费。[免费目录](https://openrouter.ai/collections/free-models/)、[FAQ 与计费](https://openrouter.ai/docs/faq) |
| `dashscope` | 百炼 Qwen 系列，按成本选择 Flash / Plus / Max 等；搜索可选支持联网的 Qwen | 文本、原生搜索 | 部分地域和模型有新人限时额度；实时 API 通常按量付费，也有资源包/套餐。免费额度用尽可能转付费，可开启用完即停。套餐专属 key/URL 不能和通用 API 混用。[免费额度](https://help.aliyun.cn/zh/model-studio/new-free-quota)、[账单与计费模式](https://help.aliyun.cn/zh/model-studio/bill-query-and-cost-management) |
| `tencentmaas` | 腾讯云 TokenHub 的 DeepSeek / 混元等模型，复制控制台 model ID | 文本、工具调用；平台其他多模态产品尚未接入专用角色 | 可按模型领取新人体验额度；额度有效期及共享规则以控制台为准。普通语言 API 按输入、输出、缓存 token 后付费，另有独立套餐。[新人体验包](https://cloud.tencent.com/document/product/1823/130053)、[计费方式](https://cloud.tencent.com/document/product/1823/130054)、[模型入口](https://cloud.tencent.com/product/tokenhub) |
| `volc` | 火山方舟 Doubao / DeepSeek 等；可使用开通的模型 ID 或推理接入点 ID | 文本、工具调用；不是平台所有多模态模型都可直接接入 | 提供模型粒度的免费推理试用额度；用尽后按开通的计费方式处理，也有资源包或独立订阅。[试用额度](https://docs.volcengine.com/docs/ark/free-inference-quota?lang=zh&redirect=1)。标准 API、专属部署和 Coding/Agent Plan 的计费及适用范围不同。[模型价格](https://docs.volcengine.com/docs/ark/model-pricing?lang=zh&redirect=1) |
| `openai` | GPT 系列用于文本/工具；GPT Image 系列用于图片 | 文本、白名单 GPT 原生搜索、GPT Image 生图/编辑；直连识图尚未加入当前白名单 | API 按输入、输出、缓存及工具等计费；不要把 ChatGPT 订阅直接当作通用 API 余额，符合条件的订阅 API credits 另有领取及适用范围。[API 定价](https://developers.openai.com/api/docs/pricing)、[订阅 API credits](https://developers.openai.com/api/docs/guides/chatgpt-api-credits) |
| `deepseek` | `deepseek-flash`、`deepseek-v4-pro` | 文本、工具调用；本项目尚未接其原生搜索/视觉接口 | API 按 token 用量计费，输入缓存等可能分价；聊天网页免费不能推导为 API 免费。赠额以账户为准。[模型与价格入口](https://api-docs.deepseek.com/quick_start/pricing) |
| `google` | Gemini 文本系列；Gemma 视觉型号；`gemini-3.1-flash-tts-preview` 等 TTS 型号 | 文本、Gemma 白名单识图、原生 TTS；未接 Google 原生搜索和生图 | 部分型号有受限免费层，包括所列 TTS 的免费层；付费层按文本/音频 token 等计费。型号、地区和免费层数据使用条款须核对。[Gemini API 定价](https://ai.google.dev/gemini-api/docs/pricing) |
| `anthropic` | Claude Sonnet / Haiku / Opus 等系列，分别权衡费用、速度与质量 | 文本、工具调用，走 Messages 协议；当前未开放直连识图角色 | API 按 token、缓存等计费；不要把 Claude 网页套餐直接当 API key。赠额或订阅 API 权益按平台规则核对，不承诺常驻免费 API。[模型与定价](https://platform.claude.com/docs/en/about-claude/pricing) |
| `siliconflow` | DeepSeek / Qwen / GLM 等；如 `deepseek-ai/DeepSeek-V4.1-Flash`；视觉选 Qwen-VL / GLM-V 型号 | 文本、白名单识图 | 价格页有免费及付费型号；同系列或带 `Pro/` 的条目不一定同价。付费按模型输入/输出/缓存等用量计费，赠送余额只是试用额度。[当前模型价格](https://www.siliconflow.cn/pricing) |
| `xiaomimimo` | 小米 MiMo 文本系列；当前平台还提供多模态/语音产品 | 文本、工具调用；本项目未接其 TTS/搜索/生图 | 普通 API key 按实际 token 消耗账户余额；Token Plan 是另一套订阅额度，不互通。免费 TTS 等活动不等于本项目已适配或文本 API 永久免费。[API 定价](https://mimo.mi.com/docs/en-US/pricing)、[平台及套餐](https://platform.xiaomimimo.com/) |

Antigravity 的 Claude / Gemini 型号与免费额度随套餐变化，部署时查询账号可用列表。[官方型号表](https://antigravity.google/docs/models/)

## 文本角色怎么分配

优先把强模型用在 **`MODEL_PLANNER` 和 `MODEL_CHAT`**，两者可以是不同模型：

| 配置 | 选型重点 |
| --- | --- |
| `MODEL_PLANNER` | 行为决策的逻辑性：判断是否接话、回应谁、是否等待补充，优先选择推理和指令遵循可靠的模型 |
| `MODEL_CHAT` | 最终输出的语言风格：人设、语气、自然程度和表达质量，优先选择语言表现好的强模型 |
| `MODEL_GENERAL` | 通用默认路由；按能力与预算选，不必让所有任务都消耗最贵模型的额度 |
| `MODEL_SUMMARY` / `MODEL_IMPORTANT_INFO` | 摘要和记忆提取；质量达标后可考虑更经济的模型 |

额度较少、语言风格好的 Antigravity Claude / Gemini 可以优先给 chat；planner 另选逻辑性强且可用性合适的模型。NVIDIA 或短期免费模型是否适合放入关键角色，要先看实际延迟和成功率。角色不填时仍继承 `MODEL_GENERAL`，最简部署不增加必填项。

## 专用能力究竟能填什么

能力限制由 [providers.py](../src/plugins/llm_chat/providers.py) 统一判断。以下是**代码接受规则**，还要满足上游模型真实能力和账号权限；通过规则不等于已经做过真实联网验证。

| 角色 | 当前接受的 provider / 型号范围 | 协议与注意事项 |
| --- | --- | --- |
| `MODEL_GENERAL` 及普通文本角色 | 全部已注册 provider，所选模型须支持工具调用 | OpenAI chat、Anthropic Messages 或方舟适配器；低价/免费模型也要测试工具返回 |
| `MODEL_VISION` | `cliproxyapi`：ID 含 `gpt-5.`、`gpt-6-luna`、`gemini`、`claude`，排除 image；`google`：Gemma；`nvidia`：Gemma 或 `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`；`siliconflow`：Qwen / VL / GLM-4.6V / GLM-4.5V 规则 | 图片输入通过现有 chat 适配器；必须选真正有视觉能力的型号，纯文本 Qwen 也可能通过子串规则，不能据此启用 |
| `MODEL_SEARCH` | `dashscope` 的实际支持搜索型号；`openai` / `cliproxyapi` 的 `gpt-6-astra`、`gpt-6-luna`、`gpt-5.6-sol`、`gpt-5.6-terra`、`gpt-5.6-luna`、`gpt-5.5`，允许日期后缀 | 百炼原生搜索或 GPT Responses `web_search`；列表包含兼容旧部署的 ID，不保证仍可申请；更新型号需先适配，不能任意替换 |
| `MODEL_TTS` | `google`，型号 ID 含 `tts` | Google 原生 `generateContent` 音频输出；独立 `GOOGLE_TTS_BASE_URL`，不能填 OpenAI-compatible URL |
| `MODEL_IMAGE_GEN` | `openai` / `cliproxyapi`，型号 ID 含 `gpt-image` | `/images/generations` 与 `/images/edits`，仅支持 Responses 内置画图工具的网关不够 |

例如新的 GPT 文本型号即使能用于 general，也不一定已进入搜索或视觉规则。模型被拒绝时先检查这里与 doctor 输出，不要通过改别名伪装成另一个模型。读取群友给出的网页 URL 不依赖 `MODEL_SEARCH`。

## 推荐配置组合

### 多 provider 与自动 fallback

完整组合见 [.env.models.example](../.env.models.example)，包含 GPT planner、多 provider chat、识图候选链、Google TTS 和生图。把需要的路由合并到 `.env`，并填写对应 provider 凭据。

例如该组合中的摘要和全局备用：

```dotenv
MODEL_SUMMARY="cliproxyapi:gpt-6-luna | google:gemini-3.5-flash-lite"
MODEL_FALLBACK="google:gemini-3.5-flash-lite | openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"
```

每次摘要先尝试 GPT，再尝试 Google；自身链耗尽后再进入全局备用。候选不会去重，因此这里 Google 会再次出现。未配置凭据或正在接口冷却中的候选会跳过；总时间预算耗尽或任务取消时不再继续尝试。

` | ` 同样适用于 `MODEL_CHAT`、`MODEL_PLANNER`、`MODEL_GENERAL` 及其他模型配置。专用能力也可配多个候选，但不追加普通文本的 `MODEL_FALLBACK`；每个候选须符合该能力的协议限制。完整规则见 [配置手册](configuration.md#候选链与自动-fallback)。

### 不部署网关，先用一个普通 API

```dotenv
MODEL_GENERAL=deepseek:deepseek-flash
DEEPSEEK_API_KEY=replace-with-your-deepseek-key
```

其他 provider 同理，通常只改 `MODEL_GENERAL` 和对应 key。专用能力可跨 provider 混搭，不需要让聊天、搜索和 TTS 属于同一家。

## 部署 AI 的选型与费用检查

1. 先复用用户已有账号或 API，不替用户开通付费服务。确认需要哪些能力；基础聊天不要求配齐四个专用模型。
2. 从控制台或服务端模型列表确认精确 ID、工具调用、权限和地域；聚合平台上的同名模型可能来自不同渠道。
3. 区分免费层（限流且可能调整）、新人赠额（有有效期）、订阅额度（固定周期重置）和按量计费。套餐通常有专属 key、URL 或用途，不能假定任意 bot 调用都能抵扣。
4. 普通聊天也包含规划、摘要、记忆提取与工具后续调用；一条 QQ 消息不一定只产生一次模型请求。缓存、推理输出、图片、音频、联网工具可能分别收费。用 `/usage` 观察调用量，最终金额以服务商账单为准。
5. 先执行 `python scripts/doctor.py --json` 检查本地配置，再以虚构短文本和测试图片分别验证真实接口。doctor 不发模型请求，不证明余额、OAuth 或专用接口可用；验证请求可能产生用量。

本地 BGE 向量推理不产生远程 embedding API 费用，仍有下载、磁盘和 CPU 开销。
