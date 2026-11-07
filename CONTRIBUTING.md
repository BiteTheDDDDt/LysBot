# 开发指南

使用 Python 3.11+ 虚拟环境，在仓库根目录运行：

```bash
python -m pip install -e ".[dev]"
python scripts/run_tests.py
python scripts/smoke_test.py
```

测试入口使用虚构凭据和临时数据库，禁用 HTTPX 传输，不需要 `.env`、QQ 或真实模型服务。smoke test 在临时源码副本中启动真实 bot，以本地模型桩和模拟 OneBot 验证消息闭环；它不使用 QQ 账号或外部模型。按文件运行单元测试可追加 `-p test_lysbot_config.py`。

| 位置 | 用途 |
| --- | --- |
| `bot.py`、`src/bootstrap.py` | 启动和 master 身份映射 |
| `src/plugins/llm_chat/chat.py`、`planner.py`、`turns.py` | 消息处理、参与与输出编排 |
| `agent_runner.py`、`capabilities.py`、`execution.py` | 工具循环、目录和执行预算（位于 llm_chat） |
| `memory.py`、`memory_index.py`、`retrieval.py`、`history_sql.py` | 存储、混合检索和来源核验 |
| `memory_changes.py`、`memory_trace.py` | 记忆纠错与使用记录 |
| `delivery.py`、`delivery_outbox.py` | 发送回执及持久化防重 |
| `lysbot/`、`extensions/` | 公共扩展 API 与私人插件目录 |
| `personas/` | 可修改的角色配置 |

保持变更范围明确。涉及会话隔离、取消、来源、记忆修改或发送语义时补充回归验证。测试使用合成资料，不加入群聊原文、截图、真实账号或密钥。

贡献前运行相关测试和 `git diff --check`。存量文件沿用原有格式，不要求为小改动格式化整个项目。

`python scripts/check_release.py` 检查已跟踪文件和历史中的已知敏感路径及常见密钥格式。它不扫描全部历史内容，也不能代替聊天资料与素材来源的人工审查。

本轮执行过的验证及其边界见 [验证记录](docs/validation.md)。AI 操作规范见根目录 `AGENTS.md`。
