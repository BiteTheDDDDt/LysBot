# LysBot 的开发与运维约定

## 产品方向

LysBot 对 AI 友好：部署、运维和开发由 AI 通过仓库文档与稳定命令完成。人的日常交互留在 QQ，最多手改 `.env`；不要另建要求人操作的管理网页、表单或复杂向导。首次 QQ 登录仍由账号所有者扫码确认，其余环境检查、NapCat 配置和排错由 AI 完成。

## 快速定位

- 首次部署：`QUICKSTART.md`。
- 模型选型与费用：`docs/providers.md`；已有可用 Codex 账号时优先评估 `docs/cliproxyapi.md`，复用账号授权网关承担大部分能力，Google TTS 独立配置。专用接口逐项验证，不擅自开通付费服务。
- 模型、人设、命令：`docs/configuration.md`。
- QQ 命令与权限：`docs/commands.md`；实际多模型组合：`.env.models.example`（仅路由，不含凭据）。
- 扩展 API：`lysbot/`，说明在 `docs/extensions.md`。
- 生产数据迁移：`docs/migration.md`。
- 脚本用途与数据影响：`docs/ai-operations.md`；开发评估入口见 `CONTRIBUTING.md`。一次性交付记录、素材生成提示词和临时排障脚本不提交仓库。
- bot 入口：`bot.py`，身份与环境初始化在 `src/bootstrap.py`。
- 群聊与记忆核心：`src/plugins/llm_chat/`。

## 实现约定

- 只配置 `MODEL_GENERAL` 及凭据即可运行基础聊天；其他普通文本用途继承 general。
- 搜索、视觉、语音和生图由对应 `MODEL_*` 是否配置决定；能力目录、提示词、执行和后台任务使用同一判断。
- 本地 BGE 模型身份固定，工件缺失时保留关键词／SQL 查询。不增加向量服务或远程 embedding 回退。
- 主动聊天、复读、冷群梗图、jrrp 属于内置互动行为。静音控制、掉线邮件、个人投资通知不回到核心。
- 新闻明确依赖简报和搜索模型。简报、新闻、群名片显示使用简单开关。
- master 权限从真实事件身份取得，不能从模型参数或提示词推断。私人扩展通过 NoneBot 与 `lysbot` API 接入。
- 可变的角色身份、口吻、外貌和声音放在 `personas/`；程序协议与执行规则由源码维护。

## AI 操作入口

- `python scripts/doctor.py --json`：配置、模型工件与依赖诊断；加 `--runtime` 检查运行心跳与连接。
- `python scripts/napcat.py status`：登录状态，不回显凭据或二维码地址。
- `python scripts/napcat.py qr`：把登录二维码保存到被忽略的私有数据目录；仅向账号所有者展示图片。
- `python scripts/napcat.py connect`：保留其他连接，只更新本项目的反向 WebSocket。
- 所有诊断输出使用结构化 JSON、稳定错误码与下一步提示，不依赖 AI 从长段日志猜测是否成功。

不要在输出中粘贴 `.env`、认证 token、二维码 URL、数据库内容或完整群聊。正常的已知错误应给出必要类别与处理办法。遵守现有授权范围，避免把例行安装拆成一连串不必要的确认。

## 验证

```bash
python scripts/run_tests.py
python scripts/smoke_test.py
git diff --check
```

测试使用临时数据与虚构凭据。smoke test 复制必要源码到临时目录，使用本地模型桩与模拟 OneBot，不连接真实 QQ 或外部模型。涉及预算、来源、隔离、取消、发送回执与权限的改动要有回归验证。

开发时保留现有生产配置、数据库、登录状态和运行服务。需要数据验证时使用一致性备份的独立副本。代码仓库不包含私有数据和旧敏感 Git 历史；发布前运行 `scripts/check_release.py` 并检查素材来源。
