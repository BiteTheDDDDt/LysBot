# Provider、模型能力与费用

核对日期：2026-11-07。本页覆盖 LysBot 注册的全部 12 个 provider。模型、活动和账号权限会变化；下面列选型方向与例子，不是完整型号目录或长期价格承诺。部署 AI 应核对官方目录及当前账号，再执行实际验证。

## 怎么选

优先考虑 **CLIProxyAPI + Codex OAuth**：已有可用 Codex 账号时，用一个本地网关承接聊天、规划、摘要、识图，以及账号支持的原生搜索与生图。语音另配 Google API key。部署步骤见 [CLIProxyAPI 指南](cliproxyapi.md)。不需要语音就不配 Google；先只配置 `MODEL_GENERAL` 也能跑。

没有适用的 Codex 账号，可以直接选择下面任一支持工具调用的文本模型。希望先免费试用，可看 Google 的免费层、OpenRouter 的免费模型、SiliconFlow 标价免费的模型，或 NVIDIA 的开发试用。免费层、试用额度、订阅权益是不同概念，群聊高频运行前应核对限额。

## 全部 provider

“LysBot 能力”描述当前适配器，**不是服务商全部产品能力**。所有通用模型都需要实际支持工具调用。具体凭据与默认 URL 已逐项写在 [.env.example](../.env.example)。

| provider | 模型选择方向 / 示例 | LysBot 当前能力 | 免费与付费方式、官方入口 |
| --- | --- | --- | --- |
| `cliproxyapi` | Codex 账号实际列出的 GPT；如 `gpt-6-luna`；Images 路由如 `gpt-image-2`。也可汇聚其他已授权渠道 | 文本；白名单识图、GPT 原生搜索、GPT Image 生图/编辑 | 代理软件开源免费；上游用量受账号套餐、额度和授权渠道限制，不等于免费模型或无限调用。[上游项目](https://github.com/router-for-me/CLIProxyAPI)、[Codex 计费](https://learn.chatgpt.com/docs/pricing) |
| `openai` | GPT 系列用于文本/工具；GPT Image 系列用于图片 | 文本、白名单 GPT 原生搜索、GPT Image 生图/编辑；直连识图尚未加入当前白名单 | API 按输入、输出、缓存及工具等计费；不要把 ChatGPT 订阅直接当作通用 API 余额，符合条件的订阅 API credits 另有领取及适用范围。[API 定价](https://developers.openai.com/api/docs/pricing)、[订阅 API credits](https://developers.openai.com/api/docs/guides/chatgpt-api-credits) |
| `deepseek` | `deepseek-flash`、`deepseek-v4-pro` | 文本、工具调用；本项目尚未接其原生搜索/视觉接口 | API 按 token 用量计费，输入缓存等可能分价；聊天网页免费不能推导为 API 免费。赠额以账户为准。[模型与价格入口](https://api-docs.deepseek.com/quick_start/pricing) |
| `dashscope` | 百炼 Qwen 系列，按成本选择 Flash / Plus / Max 等；搜索可选支持联网的 Qwen | 文本、原生搜索 | 部分地域和模型有新人限时额度；实时 API 通常按量付费，也有资源包/套餐。免费额度用尽可能转付费，可开启用完即停。套餐专属 key/URL 不能和通用 API 混用。[免费额度](https://help.aliyun.cn/zh/model-studio/new-free-quota)、[账单与计费模式](https://help.aliyun.cn/zh/model-studio/bill-query-and-cost-management) |
| `google` | Gemini 文本系列；Gemma 视觉型号；`gemini-3.1-flash-tts-preview` 等 TTS 型号 | 文本、Gemma 白名单识图、原生 TTS；未接 Google 原生搜索和生图 | 部分型号有受限免费层，包括所列 TTS 的免费层；付费层按文本/音频 token 等计费。型号、地区和免费层数据使用条款须核对。[Gemini API 定价](https://ai.google.dev/gemini-api/docs/pricing) |
| `anthropic` | Claude Sonnet / Haiku / Opus 等系列，分别权衡费用、速度与质量 | 文本、工具调用，走 Messages 协议；当前未开放直连识图角色 | API 按 token、缓存等计费；不要把 Claude 网页套餐直接当 API key。赠额或订阅 API 权益按平台规则核对，不承诺常驻免费 API。[模型与定价](https://platform.claude.com/docs/en/about-claude/pricing) |
| `nvidia` | NIM 目录中的 Kimi / Qwen / Nemotron / Gemma 等，以目录完整 ID 为准 | 文本；Gemma 或已适配 Nemotron Omni 识图 | Developer Program 提供开发体验；速率、许可和可用性有限制，生产服务需核对订阅/授权。自行部署还要承担 GPU 等费用，不能按“永久免费生产 API”规划。[NIM FAQ](https://docs.api.nvidia.com/nim/docs/product)、[NIM 提供方式](https://docs.nvidia.com/nim/large-language-models/latest/day-0/index.html) |
| `openrouter` | 多家模型；ID 含厂商前缀；目录中实际存在的 `:free` 变体或 `openrouter/free` | 文本、工具调用；未接其搜索/视觉/生图专用能力 | 免费模型有单独限流，部分需购入 credits 才有较高配额；付费模型充值 credits 后按实际用量扣除，购买 credits 有平台手续费。[FAQ 与计费](https://openrouter.ai/docs/faq) |
| `siliconflow` | DeepSeek / Qwen / GLM 等；如 `deepseek-ai/DeepSeek-V4.1-Flash`；视觉选 Qwen-VL / GLM-V 型号 | 文本、白名单识图 | 价格页有免费及付费型号；同系列或带 `Pro/` 的条目不一定同价。付费按模型输入/输出/缓存等用量计费，赠送余额只是试用额度。[当前模型价格](https://www.siliconflow.cn/pricing) |
| `tencentmaas` | 腾讯云 TokenHub 的 DeepSeek / 混元等模型，复制控制台 model ID | 文本、工具调用；平台其他多模态产品尚未接入专用角色 | 普通语言 API 按输入、输出、缓存 token 后付费；另有套餐，赠额和适用端点以账户为准，不承诺普遍免费。[计费方式](https://cloud.tencent.com/document/product/1823/130054)、[模型入口](https://cloud.tencent.com/product/tokenhub) |
| `volc` | 火山方舟 Doubao / DeepSeek 等；可使用开通的模型 ID 或推理接入点 ID | 文本、工具调用；不是平台所有多模态模型都可直接接入 | 按量计费，部分有资源包、试用或独立订阅。标准 API、专属部署和 Coding/Agent Plan 的计费及适用范围不同。[模型价格](https://docs.volcengine.com/docs/ark/model-pricing?lang=zh&redirect=1) |
| `xiaomimimo` | 小米 MiMo 文本系列；当前平台还提供多模态/语音产品 | 文本、工具调用；本项目未接其 TTS/搜索/生图 | 普通 API key 按实际 token 消耗账户余额；Token Plan 是另一套订阅额度，不互通。免费 TTS 等活动不等于本项目已适配或文本 API 永久免费。[API 定价](https://mimo.mi.com/docs/en-US/pricing)、[平台及套餐](https://platform.xiaomimimo.com/) |

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

### Codex 网关承担大部分能力，Google 承担语音

以下是账号可用时的候选示例；按 [网关指南](cliproxyapi.md) 逐项验证，不支持的项留空。

```dotenv
MODEL_GENERAL=cliproxyapi:gpt-6-luna
MODEL_VISION=cliproxyapi:gpt-6-luna
MODEL_SEARCH=cliproxyapi:gpt-6-luna
MODEL_IMAGE_GEN=cliproxyapi:gpt-image-2
CLIPROXYAPI_BASE_URL=http://cliproxyapi:8317/v1
CLIPROXYAPI_API_KEY=replace-with-your-local-proxy-key
MODEL_TTS=google:gemini-3.1-flash-tts-preview
GOOGLE_API_KEY=replace-with-your-google-key
```

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
