<p align="center"><img src="assets/lysbot-header.png" alt="LysBot · 磷叶石 Bot" width="100%"></p>

# LysBot · 磷叶石 Bot

一个有长期记忆、能查历史、会接梗的 QQ 群友。基于 NoneBot2 和 NapCat，默认人设是《宝石之国》的磷叶石（法斯）。

LysBot 把群聊原文、摘要和重要信息保存在 SQLite 中，可以按语义找线索、用 SQL 查统计，再回读原话。记忆支持查看、确认纠正和撤销。主动接话、表情包、复读、冷群梗图、戳一戳和 jrrp 都属于日常群聊互动。

部署、运维和开发面向 AI：人主要在 QQ 使用和管理，最多编辑 `.env`。AI 从 [AGENTS.md](AGENTS.md) 和 quickstart 进入，通过结构化诊断及操作脚本完成其余工作。

## 跑起来

源码仓库：[BiteTheDDDDt/LysBot](https://github.com/BiteTheDDDDt/LysBot)。首次使用先按 [quickstart](QUICKSTART.md) 克隆仓库。

准备 Docker Compose，复制 `.env.example` 为 `.env`，填写 master QQ、连接 token、`MODEL_GENERAL` 和模型服务凭据。

```bash
docker compose build
docker compose run --rm setup
docker compose up -d
```

随后由 AI 获取登录二维码并自动配置 NapCat 连接，账号所有者只需在手机 QQ 扫码确认。完整步骤见 [QUICKSTART.md](QUICKSTART.md)，操作接口见 [AI 运维说明](docs/ai-operations.md)。

只填 `MODEL_GENERAL` 就能使用基础聊天、规划、摘要和记忆提取。固定的本地 BGE 模型由 setup 准备，日常 CPU 离线推理，不需要向量服务或 GPU。没准备好模型时仍保留关键词和 SQL 查询。

## 按配置增加能力

| 配置 | 能力 |
| --- | --- |
| `MODEL_VISION` | 识图与表情包采集 |
| `MODEL_SEARCH` | 原生联网搜索 |
| `MODEL_TTS` | 语音生成 |
| `MODEL_IMAGE_GEN` | 生图与参考图编辑 |

没有填写就不提供对应能力。专用模型不自动继承 general；通用模型需要支持工具调用。可选的 `MODEL_CHAT`、`MODEL_PLANNER`、`MODEL_SUMMARY`、`MODEL_IMPORTANT_INFO` 用于覆盖高用量任务，备用链使用 ` | ` 分隔。

每日群聊简报、新闻和群名片模型显示各有开关，默认关闭。新闻依赖简报和搜索模型。简报没有生图模型时不绘制群像。

## 人设与扩展

默认人设位于 [personas/phosphophyllite.toml](personas/phosphophyllite.toml)，身份、性格、外貌和声音集中配置。工具协议、事实与记忆规则由程序维护。设置 `PERSONA_FILE` 可使用自己的角色文件。

`MASTER_QQ` 是唯一管理员，拥有模型切换、数据清理等管理权限。自定义工具、定时任务和外部通知可以通过可信的本地 NoneBot 插件接入，见 [扩展指南](docs/extensions.md)。私人插件保存在被 Git 忽略的 `extensions/` 中。

## 开发与迁移

```bash
python -m pip install -e ".[dev]"
python scripts/run_tests.py
```

- [开发指南](CONTRIBUTING.md)
- [从旧部署迁移](docs/migration.md)
- [配置与运行说明](docs/configuration.md)

聊天内容会写入本地数据库，并按实际配置发送给模型服务。部署者应向群友说明记录和模型调用方式。当前首版面向 NapCat，其他协议端尚未验证；离线测试不能保证模型服务或 QQ 上游始终可用。

代码采用 [Apache License 2.0](LICENSE)。角色名称和形象属于各自权利人，头图来源及适用范围见 [素材说明](assets/README.md)。
