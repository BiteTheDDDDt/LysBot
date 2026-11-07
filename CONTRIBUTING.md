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

自动测试结果见仓库的 [GitHub Actions](https://github.com/BiteTheDDDDt/LysBot/actions)。AI 操作规范见根目录 `AGENTS.md`。

## 开发与评估脚本

| 脚本 | 使用场景 |
| --- | --- |
| `scripts/run_tests.py` | 日常回归与 CI，隔离配置、数据和 HTTP 请求；`-p` 选择测试文件 |
| `scripts/smoke_test.py` | 在临时目录启动 bot，以模拟模型和 OneBot 验证完整消息闭环 |
| `scripts/evaluate_agentic_search.py` | 调整历史检索时，使用 `tests/fixtures/agentic_search_eval.json` 的合成资料比较检索或 agent 效果；不是部署步骤 |
| `scripts/evaluate_local_embedding.py` | 下载权重后验证真实本地 BGE 的语义召回、来源、隔离和延迟，禁止网络请求；不是每次单测的依赖 |
| `scripts/check_release.py` | 发布前检查已跟踪文件中的常见密钥及私有路径 |

评估脚本先用 `--help` 查看参数。检索评估的 `prepare`、`retrieve` 使用独立的 `agentic-eval` 数据库；`agent` 会调用配置的真实模型并消耗额度，不能当作离线单测。BGE 评估使用单独的 `agentic-local-eval*` 目录与已有权重。所有结果留在被忽略的 `data/` 下，不提交一次运行的报告。

部署与维护脚本的用途及数据影响见 [AI 操作说明](docs/ai-operations.md)。新增脚本应有明确使用场景、可用入口和必要验证；临时排查、一次性迁移及制作素材的过程文件留在本地，不作为公共项目接口提交。
