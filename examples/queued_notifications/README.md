# 任务完成后通知 master

一个可选的本地文件队列插件：构建任务、定时报告、投资分析脚本或服务检查将结果放入队列，LysBot 私聊通知 `MASTER_QQ`。通知直接调用 `notify_master`，不消耗模型额度。

## 安装

让部署 AI 把本目录复制到**实际挂载的** `extensions/queued_notifications/`，然后重启 bot。默认部署的操作如下；目标目录已有内容时先检查，不覆盖定制代码。

```powershell
# Windows PowerShell，在 LysBot 根目录
Copy-Item -Recurse examples/queued_notifications extensions/queued_notifications
docker compose restart bot
```

```bash
# Linux / macOS，在 LysBot 根目录
cp -R examples/queued_notifications extensions/queued_notifications
docker compose restart bot
```

生产部署使用独立插件目录时，让 AI 按 Compose 挂载找到该目录并复制。插件随 `extensions/` 挂载加载，无需重新构建镜像。确认日志显示 `queued_notifications` 加载成功；本例等待唯一一个 QQ 账号在线后处理任务。

## 提交一条通知

安装后可以直接使用容器内的 Python，不要求宿主机安装依赖。下面的命令会向 master 发送一条私聊：

```bash
docker compose exec bot python /app/extensions/queued_notifications/enqueue.py --directory /app/data/queued_notifications --id build-demo-001 --text "示例项目构建完成，测试通过。" --ttl 3600
```

外部程序也可直接调用同一个脚本，只需 Python 3.11+ 标准库，不需要 API key 或 QQ 号：

```bash
python examples/queued_notifications/enqueue.py --directory data/queued_notifications --id report-demo-001 --text "每日报告已生成。" --ttl 3600
```

外部程序必须能访问 bot 使用的同一队列目录。默认 Compose 的 `/app/data/queued_notifications` 对应宿主机 `data/queued_notifications`；自定义挂载时替换 `--directory`。远程 CI 需要共享目录或在部署机器执行提交命令。

`--id` 使用稳定的任务编号，例如构建 ID 或报告日期；生产者重试时复用它。每条新通知使用新 ID。重复提交同一 ID 不覆盖待发送内容，也不触发第二次发送。`--ttl` 是有效秒数，默认一小时；每条文本最多 1800 字符。

## 队列与回执

队列在 `LLM_CHAT_DATA_DIR/queued_notifications/` 下，默认每 10 秒处理一次：

```text
queued_notifications/
├── pending/     # 已完整写入的任务，文件名为任务 ID 的 SHA-256
└── receipts/    # 持久化发送状态，也是防重复依据
```

- `sent`：QQ 返回了 `message_id`，不代表对方已读。
- `expired` / `rejected`：任务已过期或格式不合法，没有发送。
- `sending` / `unknown`：已认领但结果未确认，可能发送过，也可能没有发送；不会自动重发。

这是**每个 ID 至多尝试一次**的示例，不保证每条通知必达。进程退出、超时、断网或回执写入失败时，宁可保留待核对状态。不要自动删除回执重试；确认确实需要补发后再用新任务 ID 提交。备份或迁移时一起保留任务与回执。

任务先写临时文件，再通过同目录硬链接原子发布；使用支持硬链接的本地文件系统（如 NTFS、ext4）。收件人始终由 LysBot 的 `MASTER_QQ` 决定，任务不能指定其他收件人。只给可信的本地程序写队列权限，不把它当作公开网络接口。
