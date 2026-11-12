# 开发指南

使用 Python 3.11+ 虚拟环境，在仓库根目录运行：

```bash
python -m pip install -e ".[dev]"
python scripts/run_tests.py
python scripts/smoke_test.py
```

测试入口使用虚构凭据和临时数据库，禁用 HTTPX 传输，不需要 `.env`、QQ 或真实模型服务。smoke test 在临时源码副本中启动真实 bot，以本地模型桩和模拟 OneBot 验证消息闭环；它不使用 QQ 账号或外部模型。按文件运行单元测试可追加 `-p test_lysbot_config.py`。

本地 Python 环境用于开发调试和离线测试。准备部署时使用与普通用户相同的 Dockerfile 和 Compose 流程，见 [构建与更新](docs/ai-operations.md#构建与更新)；项目自身的生产环境也使用该流程，私人功能通过插件接入。

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

## 发行版

正式 Release 是普通用户默认部署和升级的版本，main 用于持续开发。版本使用 `v主版本.次版本.修订号` tag，与 `pyproject.toml` 保持一致；0.x 阶段仍可能调整配置与扩展接口，变化须在发行说明中交代。需要提前测试的版本标为 prerelease，不作为默认部署目标。

只有用户明确要求发版时才创建发行 tag／GitHub Release。持续部署的授权仅用于更新指定生产实例，不自动发布新版本。

发布前确认目标提交的 Linux／Windows、Python 3.11／3.13 CI 全部通过，完成发布检查及相关功能验证。tag 固定到该提交，发布后不移动 tag；修复另发版本。发行说明写清功能、已知限制、升级所需操作和 QUICKSTART 入口。首版使用 v0.1.0，后续版本根据改动递增。

发行版交付所选 tag 的源码，继续使用仓库 Dockerfile 和 Compose 构建部署，不要求上传运行数据、私人配置或额外制作二进制包。

## 核心模块边界

核心仍是单进程 NoneBot 应用。按职责定位修改，不在辅助模块创建第二套 manager、数据库连接或后台任务。

| 模块 | 职责与边界 |
| --- | --- |
| `chat.py` | 事件处理、群聊聚合、参与规划与回复投递的衔接 |
| `incoming.py` | OneBot 消息规范化、提及、引用和转发展开；拥有有界昵称缓存，不调用模型或写记忆 |
| `models.py` | 模型路由、adapter 缓存、备用链与视觉请求；保留 `model_manager` 和已有调用入口 |
| `model_types.py` | 共享消息／响应类型、用量及会话 ContextVar；其他模块需要数据类型时直接依赖这里 |
| `model_adapters.py` | Provider 请求格式和协议；不决定业务模型路由 |
| `model_turn.py` | 回合期限、Agent 生命周期及 planner 到 chat 的交接；通过传入的 manager 请求模型 |
| `memory.py` | 会话缓存、数据库连接生命周期、共享写锁、摘要和信息抽取任务；保留原有记忆 API |
| `memory_schema.py` | 在传入的连接上建表、升级旧列及安装检索 schema；不创建连接 |
| `memory_queries.py` | 按会话隔离的 SQL 历史查询；通过回调取得当前连接 |
| `digest_repository.py` | 简报、新闻及其来源查询和持久化；复用记忆连接、写锁和回滚逻辑 |

`models.py` 保留共享类型与 adapter 的兼容导出，不能另建同名数据类型或 ContextVar。记忆仓储不保存某次打开的连接，确保关闭后重新连接仍然有效；写入继续遵守会话失效检查、取消回滚和同一连接的事务互斥。

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
