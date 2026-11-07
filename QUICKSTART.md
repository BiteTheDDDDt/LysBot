# 运行 LysBot

把本文件交给 AI 助手时，可以说：

> 阅读 AGENTS.md 和 QUICKSTART.md，帮我在独立目录部署 LysBot。保留已有配置和数据，不回显密钥；自动完成 NapCat 配置，把二维码私下展示给我扫码。验证实际 QQ 回复，并分别报告模型、连接和向量索引的状态。

下面的命令供 AI 执行；人只需提供或填写 `.env`，并在手机 QQ 中完成首次登录确认。

**给部署 AI：优先评估 CLIProxyAPI + Codex 方案。** 已有可用 Codex 账号时，按 [网关部署指南](docs/cliproxyapi.md) 配好本地代理，让它承接聊天、识图、原生搜索及账号支持的生图，TTS 单独接 Google。首次 OAuth 由账号所有者登录确认，其余配置和验证由 AI 完成。用户已有 API 或没有适用账号时直接走普通 API，不要求购买订阅。全部 provider、免费层和付费方式见 [provider 指南](docs/providers.md)。

## 1. 准备环境与配置

首次部署先克隆仓库并进入目录；已经在正确的 checkout 中则跳过，不重复克隆或覆盖。

```bash
git clone https://github.com/BiteTheDDDDt/LysBot.git
cd LysBot
```

推荐 Docker Compose（Windows 可使用 Docker Desktop 的 Linux 容器）。在 LysBot 根目录执行 `docker version` 和 `docker compose version`，确认引擎可用。

没有 `.env` 时复制 `.env.example`：PowerShell 使用 `Copy-Item .env.example .env`，Linux/macOS 使用 `cp .env.example .env`。已有文件直接编辑，不覆盖。

只需填写四类信息：

| 配置 | 内容 |
| --- | --- |
| `MASTER_QQ` | 你的管理账号，不是 bot 小号 |
| `ACCESS_TOKEN` | 一段自己生成的长随机字符串，稍后填到 NapCat |
| `MODEL_GENERAL` | `provider:model_id`，例如自己的 `openai:模型ID` |
| Provider 凭据 | 对应的 API key；CLIProxyAPI 使用本地网关访问密钥与 `CLIPROXYAPI_BASE_URL`，直连服务使用各自的 key / URL |

不用填写搜索、视觉、语音、生图或向量模型配置。通用模型需支持工具调用。不要把 `.env`、数据库、登录状态或二维码提交到 Git、粘贴到 issue 或排错报告中。

## 2. 构建并准备本地向量模型

```bash
docker compose build
docker compose run --rm setup
docker compose run --rm --no-deps bot python scripts/doctor.py --json
```

setup 下载固定的 BGE 中文模型并校验哈希，数据保存在 `data/models/`，模型本体约 95 MB。有效文件会被复用，运行 bot 时不联网下载。如果下载失败，检查网络；可先继续启动，关键词与 SQL 查询仍可用。后续准备成功后重启 bot 即可。

诊断中先修复 error；缺少向量工件的 warning 不妨碍基础聊天，可继续安装并稍后补齐。

## 3. 启动与登录 NapCat

```bash
docker compose up -d
```

先让 AI 检查配置和 NapCat 状态：

```bash
docker compose run --rm --no-deps bot python scripts/doctor.py --json
docker compose run --rm --no-deps bot python scripts/napcat.py status
```

首次启动若 NapCat 尚未生成 `webui.json`，等其初始化后再检查。未登录时由 AI 获取二维码：

```bash
docker compose run --rm --no-deps bot python scripts/napcat.py qr
```

脚本只输出图片路径，不输出 QR URL 或凭据。二维码在宿主机 `data/private/qq-login.png`；AI 将该私有图片展示给账号所有者，由其使用手机 QQ 扫码并确认。不要要求用户进入 WebUI 填写连接表单，也不要把二维码发到公开群或报告里。

扫码后再次检查 status，确认 `logged_in=true`，然后自动连接：

```bash
docker compose run --rm --no-deps bot python scripts/napcat.py connect
```

脚本只更新本项目的 WebSocket，保留其他连接，开启心跳／重连并核对回读结果。默认地址为 `ws://bot:8080/onebot/v11/ws`，token 读取 `.env` 的 `ACCESS_TOKEN`。bot 无需对宿主机发布端口。

WebUI 后端接口随 NapCat 版本可能变化；脚本报错时由 AI 按 [运维说明](docs/ai-operations.md) 排查或通过本地 WebUI 完成同样操作，不把页面配置工作转交给人。只有实际接通后才报告部署完成。

## 4. 验证

```bash
docker compose ps
docker compose exec bot python scripts/healthcheck.py
```

健康检查会同时检查插件、数据库、OneBot 连接及监听端口。首次扫码之前因尚未连接而 unhealthy 属于未完成接入，不能据此反复重建容器。

在测试群 @ bot 发送文字，确认 QQ 收到实际回复。master 可发送 `/status` 检查路由、`/usage` 查看用量。语义检索还需等待后台索引，状态检查见 [配置说明](docs/configuration.md)。没有填专用模型时，相应能力关闭是正常行为。

修改 `.env` 后执行 `docker compose up -d --force-recreate bot`；修改源码后执行 `docker compose up -d --build bot`。不要用 `docker compose down -v` 清理数据。

## 排错

- 无法启动：检查 master、general 模型、凭据及 provider 名称；配置错误会在插件启动时说明。
- NapCat 连不上：检查客户端地址、token、消息格式及两个服务是否处于同一 Compose 网络。
- 群聊不回复：先确认 OneBot 连接，再检查实际模型调用错误；日志中不能有凭据原文。
- 向量未就绪：先检查 setup 是否成功；基础聊天无需等索引完成。
- 开了新闻却启动失败：新闻需要同时开启简报并配置 `MODEL_SEARCH`。

## 本地 Python 开发

使用 Python 3.11+ 虚拟环境，运行 `python -m pip install -e ".[dev]"`。在根目录执行 `python scripts/prepare_local_embedding.py` 准备模型，再执行 `python bot.py`。NapCat 与 bot 同宿主机时客户端地址改为 `ws://127.0.0.1:8080/onebot/v11/ws`。常规验证使用 `python scripts/run_tests.py`，不连接 QQ 或调用真实模型。
