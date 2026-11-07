# 推荐部署：CLIProxyAPI + Codex，Google TTS 单独接入

这是交给部署 AI 的操作说明。已有可用 Codex 账号时，优先用 CLIProxyAPI 把上游能力统一接入 LysBot：聊天、规划、摘要、识图，以及实际可用的搜索和生图。TTS 单独走 Google 原生 API。没有适用账号或偏好直连 API 时，按 [provider 指南](providers.md) 选普通 API 即可。

同一网关也可接入 Grok、Antigravity 的可用免费模型。Antigravity 提供的 Claude / Gemini 额度较少时，可优先用于 `MODEL_CHAT`；`MODEL_PLANNER` 另选逻辑性强的模型。具体型号、免费范围与角色分配见 [provider 指南](providers.md)。新增渠道由 AI 按固定版本的上游登录说明完成授权并查询模型列表，不能直接沿用 Codex 的登录参数。

CLIProxyAPI 是第三方开源网关。它通过账号授权使用上游权益，不把订阅变成无限额度；LysBot 也不提供模型额度。[上游项目](https://github.com/router-for-me/CLIProxyAPI) 与 [OpenAI 账号授权说明](https://learn.chatgpt.com/docs/auth) 分别描述代理实现和官方登录方式，不能据此推定第三方网关获得官方背书。账号可用型号和限制以当前权益为准；订阅、普通 API 计费及可领取的 API credits 也应分别核对，见 [官方计费](https://learn.chatgpt.com/docs/pricing) 和 [API credits 范围](https://developers.openai.com/api/docs/guides/chatgpt-api-credits)。

## 1. 由 AI 准备私有配置

先读根目录 `AGENTS.md`、`QUICKSTART.md`。已有网关时优先复用其地址，不覆盖配置或复制其他项目的 OAuth 文件。新建时：

1. 在 `data/cliproxyapi/` 创建目录，创建 `auth/` 子目录；将 `deploy/cliproxyapi/config.example.yaml` 复制成其中的 `config.yaml`。已有文件只编辑必要项。
2. 使用密码学随机数生成长本地访问密钥，替换 `access.api-keys` 的占位值，并填入 `.env` 的 `CLIPROXYAPI_API_KEY`；不在终端输出密钥。此 key 与 `ACCESS_TOKEN` 各自生成，作用不同。
3. 私有配置与授权目录都在 Git 忽略的 `data/` 内；限制为部署账号可读。不要把 OAuth JSON、授权回调 URL 或浏览器会话粘贴到报告中。

模板采用 CLIProxyAPI v8 配置。覆盖文件固定为 `v8.0.59`，已核对上游配置结构与镜像标签；升级前先对照[同版本上游配置](https://github.com/router-for-me/CLIProxyAPI/blob/v8.0.59/config.example.yaml)，不要直接用不同大版本模板。模板关闭管理面板和请求正文日志，仅保留本地调用所需设置；`disable-image-generation: chat` 保留 Images API，避免给普通聊天注入画图工具。

## 2. 首次 Codex 授权

在 LysBot 根目录执行（这一步是一次性容器，退出后保留 auth 卷）：

```bash
docker compose -f compose.yaml -f compose.cliproxyapi.yaml run --rm --no-deps -p 127.0.0.1:1455:1455 cliproxyapi ./CLIProxyAPI --config /CLIProxyAPI/config.yaml --codex-login --no-browser
```

由 AI 读取本次进程输出的授权入口，私下交给账号所有者在浏览器登录并确认授权；AI 继续处理命令和回调，人的操作停留在账号登录确认。OAuth 登录不可由 QQ bot 代替所有者确认。完成后确认进程成功退出并生成 auth 文件，不输出文件内容。

浏览器和 Docker 位于同一台电脑时可使用 localhost 回调。远程服务器部署时，AI 先为本次 OAuth 建立本地端口到服务器 `127.0.0.1:1455` 的 SSH 转发，再让所有者授权；不要把回调端口直接公开到公网。若端口被占用或登录流程变更，先检查该固定版本 `./CLIProxyAPI --help` 与上游实现，不停止其他服务。不要擅自读取既有 Codex 客户端的认证文件来代替这次授权。

## 3. 启动网关与选择模型

```bash
docker compose -f compose.yaml -f compose.cliproxyapi.yaml up -d cliproxyapi
```

AI 用本地访问密钥向 `http://127.0.0.1:8317/v1/models` 发 Bearer 认证请求，只提取模型 ID；不要输出请求头或完整诊断对象。网关端口仅绑定宿主机回环地址，容器内则通过 Compose 网络访问。

`.env` 先配置基础聊天（若列表没有示例 ID，选择真实可用且支持工具调用的模型）：

```dotenv
MODEL_GENERAL=cliproxyapi:gpt-6-luna
CLIPROXYAPI_BASE_URL=http://cliproxyapi:8317/v1
CLIPROXYAPI_API_KEY=replace-with-your-local-proxy-key
```

Bot 在宿主机 Python 环境运行时 URL 改为 `http://127.0.0.1:8317/v1`。Bot 在另一个 Compose 项目时要配置共同网络或可达网关地址，不能直接用本项目服务名，也不能在容器里用 `127.0.0.1` 指向宿主机。使用 `PROVIDER_PROXY_ALL` 时为本地网关设置 `PROVIDER_PROXY_CLIPROXYAPI=direct`，避免绕到外部网络代理。

## 4. 按实际验证结果开启专用能力

候选示例：

```dotenv
MODEL_VISION=cliproxyapi:gpt-6-luna
MODEL_SEARCH=cliproxyapi:gpt-6-luna
MODEL_IMAGE_GEN=cliproxyapi:gpt-image-2
# 语音独立配置 Google；不需要语音就省略这两项。
MODEL_TTS=google:gemini-3.1-flash-tts-preview
GOOGLE_API_KEY=replace-with-your-google-key
```

由 AI 依次验证：

| 能力 | 验证目标 |
| --- | --- |
| 文本 | 使用无敏感内容的短请求确认工具调用往返可用，不能只测普通问答 |
| 识图 | 发送一张无隐私的测试图，确认返回基于图片的描述 |
| 搜索 | 用该模型的 Responses 原生 `web_search` 验证联网与来源；普通聊天成功不代表搜索成功 |
| 生图/编辑 | 分别验证 `/v1/images/generations` 和 `/v1/images/edits`；模型列表有 ID、ChatGPT 网页能画图都不能代替接口验证 |
| TTS | Google 原生 `generateContent` 返回有效音频；此处使用 `GOOGLE_API_KEY`，不是网关 key，也不是 Google Cloud TTS 的另一套 API |

真实验证会消耗上游额度；使用短请求、小图，记录成功/失败类别，不回显凭据。若某能力不可用，说明限制并留空对应项，不把基础聊天部署卡在可选能力上。完整白名单和版本限制见 [provider 指南](providers.md)。

## 5. 接回常规部署与维护

回到 [QUICKSTART](../QUICKSTART.md) 完成 build、setup、doctor、NapCat 登录与实际 QQ 回复验证。网关是独立的可选服务，普通 `docker compose up -d` 仍可启动 Bot 与 NapCat；首次和重启网关使用上面的双文件命令。

更新网关时先保留私有 config/auth 的备份，修改 `compose.cliproxyapi.yaml` 的版本并检查新配置，再拉取和重建这个服务；不要批量更新其他生产服务。升级后重新验证实际启用的接口。429 先检查上游额度和限流，401 先检查本地 key 与 OAuth 状态；连接失败先检查服务名、网络和端口。部署结果分开报告网关、模型能力、NapCat、QQ 回复及本地向量状态。

## 本仓库已验证的范围

2026-11-07：合并 Compose 配置校验通过；使用固定 `v8.0.59` 镜像、模板配置和虚构密钥，在无外网的临时容器启动成功，认证后的模型列表返回 200，未认证请求返回 401。推荐 MODEL 组合通过本地 doctor 与能力规则检查。验证未使用真实 OAuth、QQ 或模型额度，因此实际账号授权、搜索、生图及 TTS 仍需部署 AI 按上面步骤验证。
