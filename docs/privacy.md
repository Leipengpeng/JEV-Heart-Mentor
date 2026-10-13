# 数据与脱敏设计

## 公开源码包含什么

仅包含自写 Python / Swift / Web 代码、测试、固定动作表、提示词和原创合成示例。演示截图来自 `chatlens demo`，图中的联系人、聊天、文案和分数均为人为构造。

真实聊天库、联系人清单、语音、数据库密钥、会话 token、模型权重、第三方仓库、历史回测及失败输出不随源码分发。

## 本机运行会保存什么

默认数据目录：

| 平台 | 默认位置 |
| --- | --- |
| macOS | `~/Library/Application Support/ChatLensOSS/` |
| Windows | `%LOCALAPPDATA%/ChatLens/` |
| Linux | `$XDG_DATA_HOME/chatlens/` 或 `~/.local/share/chatlens/` |

可使用 `CHATLENS_DATA_DIR` 指定私有目录。请将该目录放在仓库外。目录权限设置为 700、关键文件设置为 600；Windows 的访问控制应另按本机权限管理。

- `memory.sqlite3`：导入的聊天、稳定消息 ID、按联系人隔离的摘要和可选语音转写。
- `analysis-approvals.json`：各账号下已授权分析的联系人。
- `bridge-session.json`：本次本地 HTTP 会话凭据。
- `active-source.json` / `runtime-candidates.txt`：仅在自行接入外部微信适配器时存在的数据源配置和密钥。
- `decision-events.jsonl`：联系人/消息哈希、动作和情感标签、分数及耗时，不保存对话原文或回复草稿。哈希用于本地关联，不能视为匿名数据。
- `diagnostics/` / `reports/`：本机状态与运行元数据。
- 部分旧实验接口可能保存上下文压缩缓存；它也属于私密数据。

默认不保存完整失败输入/模型输出及最后一张卡片。只有显式设置 `CHATLENS_DEBUG_CHAT=1` 才开启完整私密调试记录；这些记录可能包含聊天、生成文案和模型输入，不能用于公开 Issue。

原生 UI 的 OCR 回退只处理可见微信窗口，主要用于点击位置定位；决策输入来自数据库/JSON 历史。权限与回退能力因微信版本而异。合成演示不启动原生监听，不读取数据目录，不申请系统权限。

## 网络与授权

真实 API 仅监听回环地址，要求 token、校验 Host / Origin。模型加载使用本地文件，推理服务阻止外网 socket 连接。`download-models` 是独立联网下载命令。

联系人列表展示不等于授权分析。新联系人先显示空白卡片，由用户点击“分析”才允许其历史进入分析流程。`CHATLENS_TARGETS` 只提供 UI 别名，不自动授权联系人。

只读取所选消息及之前的上下文；不读取未来消息、不跨联系人检索。回复为建议，程序没有自动发送微信消息的接口。

## 发布检查

```sh
python scripts/check_release.py
python scripts/export_release.py
```

检查器使用源码白名单，拒绝私密文件、权重、生成物、符号链接、常见凭据、个人绝对路径和账号 ID；即使 `git add -f` 强行加入忽略文件也会检查。输出仅含路径、行号、类别。

可另提供本机私有的禁止词文件：`--deny-words-file /path/to/private-deny-list.txt`。文件本身不要放进仓库。通用扫描无法保证识别所有自然语言隐私，新增截图和示例仍需确认全部为合成数据。
