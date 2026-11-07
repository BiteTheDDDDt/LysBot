# AI 操作入口

LysBot 的人机边界：人在 QQ 里聊天和管理，最多编辑 `.env`；安装、运维和开发交给 AI。仓库根目录的 `AGENTS.md` 是代码助手入口，`QUICKSTART.md` 是部署入口，不提供要求人维护的独立控制台。

## 结构化诊断

```bash
docker compose run --rm --no-deps bot python scripts/doctor.py --json
docker compose exec bot python scripts/doctor.py --json --runtime
```

诊断只读配置与本地状态，不调用模型，也不打印凭据。JSON 包含 `schema_version`、`ok`、`capabilities` 和逐项 `checks`。每项有 `status`、稳定 `code` 和 `hint`。向量未准备好是 warning，基础聊天仍可运行；缺少 master、通用模型或凭据是 error。

配置通过不等于模型服务请求成功，诊断明确返回 `network_verified=false`。实际 QQ 回复与运行健康状态分别验证。

## NapCat 自动接入

```bash
docker compose run --rm --no-deps bot python scripts/napcat.py status
docker compose run --rm --no-deps bot python scripts/napcat.py qr
docker compose run --rm --no-deps bot python scripts/napcat.py connect
```

脚本从共享数据目录的私有 `napcat/config/webui.json` 读取 WebUI 凭据，通过 NapCat 后端接口认证，不需要人进入页面填写连接表单。`qr` 将二维码写入 `/app/data/private/qq-login.png`，对应宿主机的 `data/private/qq-login.png`；AI 仅向账号所有者展示这张图片，由其用手机 QQ 扫码并确认。不要公开 QR URL、二维码图片或原始认证响应。

登录后运行 `connect`，脚本保留其他连接，只更新名为 `lysbot` 的反向 WebSocket，使用 `.env` 的 `ACCESS_TOKEN`，写入后回读核对。反复执行应不产生重复连接。默认连接 `ws://bot:8080/onebot/v11/ws`，本地 Python 部署可通过 `--ws-url` 显式指定地址。

输出为 JSON：退出码 0 表示操作完成，1 表示失败，2 表示需要本人扫码。`status` 查询成功时仍应检查 `logged_in`、`core_ready`，不能把 HTTP 成功当作 QQ 已在线。

这些接口依据 [NapCat WebUI 后端](https://github.com/NapNeko/NapCatQQ/tree/main/packages/napcat-webui-backend/src) 实现，已有模拟接口测试；新接入真实 NapCat 版本时应实际验证，接口变化需明确报错，不能假装配置成功。

## 维护与开发

仓库保留以下运维入口。参数以各脚本的 `--help` 为准；在 Docker 内对数据操作时使用容器内路径。

| 脚本 | 用途与数据影响 |
| --- | --- |
| `doctor.py` | 只读配置、工件与可选运行状态，不调用模型 |
| `napcat.py` | 查询登录、保存私有二维码、配置本项目的反向 WebSocket |
| `healthcheck.py` | Compose 自动调用，检查进程心跳、数据库和 OneBot 状态；不需要人定期执行 |
| `prepare_local_embedding.py` | setup 服务调用，下载并校验固定 BGE 权重；已有有效文件时复用 |
| `database_maintenance.py` | SQLite 完整性检查、一致性备份、在新副本上重建普通索引；不替换运行数据库 |
| `agentic_search_ops.py` | 查询全文/向量索引进度；索引补建、失败任务重试和副本索引导入会写数据，按需执行 |
| `review_meme_library.py` | 历史图库重审；review 调用视觉模型并写报告，apply 修改随机发送资格，restore 按记录恢复。日常采集由 bot 自动处理，无需部署时重审 |
| `run_news_broadcast.py` | 新闻预览，不向 QQ 发消息、不消费简报；会调用真实模型并保存预览图片，调试新闻链路时使用 |

例如备份数据库：

```bash
docker compose exec bot python scripts/database_maintenance.py snapshot /app/data/history.db /app/data/backups/history-before-upgrade.db
```

目标文件必须不存在。在实际运行库所在的系统内执行，不从另一个 Windows/Linux 环境跨系统打开正在使用 WAL 的数据库。普通 SQLite `REINDEX` 不等于重新生成向量；向量状态与维护使用 `agentic_search_ops.py`。

开发测试和可复现评估入口见 [贡献指南](../CONTRIBUTING.md#开发与评估脚本)。

- 状态和用量优先由 master 在 QQ 使用 `/status`、`/usage` 查询。
- AI 排错先读结构化诊断，再取必要且脱敏的日志；不把完整环境或群聊数据库放入答复。
- 更新代码由 AI 构建并替换对应 bot 服务；生产数据、`.env` 和私人插件留在部署侧。
- 测试通过 `scripts/run_tests.py` 和 `scripts/smoke_test.py` 验证。后者始终使用临时源码副本与本地模拟服务，不读取项目 `.env` 或加载私人扩展。
- 需要扩展功能时使用 `lysbot` API 与 NoneBot 生命周期，不要求用户再学习另一套插件平台。
