# 设置指南

先体验合成演示，再按需安装模型和接入自己的历史。示例数据都是虚构内容。

## 合成演示

需要 Python 3.10+；建议使用 Python 3.11 或 3.12。

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e .
chatlens demo --open
```

Windows 的激活命令是 `.venv\Scripts\activate`。浏览器访问的是本机临时端口；终端按 `Ctrl+C` 停止。演示无需模型、微信、数据库密钥或系统权限，评分和文案是预置示例。

## Apple Silicon 上运行模型

当前完整模型和原生窗口路径面向 Apple Silicon、macOS 14+。需要 Xcode Command Line Tools；如未安装，可执行 `xcode-select --install` 并完成系统安装。

在已经激活的虚拟环境与仓库根目录中运行：

```sh
pip install -e '.[models]'
chatlens download-models
python scripts/build_macos.py
chatlens import-json examples/conversation.json
chatlens launch --main
```

`download-models` 联网下载三个固定版本：Typical 决策权重、Qwen3 决策骨干、MiniCPM 的 MLX 权重。下载完成后会写入清单，并校验上游 LFS 文件哈希；权重不随代码仓库发布。固定 revision 见 [README](../README.md#运行本地模型)。

`build_macos.py` 只构建并本地签名 `build/ChatLens.app` 和点击观察器，不启动应用，也不授予系统权限。`launch --main` 打开含历史列表的主窗口，并按需启动本地服务；首次模型加载进度在窗口中显示。

选择导入的联系人，点击“分析”后选择一条对方消息。JSON 导入本身不授予联系人分析权限。以后可用 `chatlens launch` 打开伴随卡片，或用 `chatlens launch --restart --main` 重启服务与窗口。

```sh
chatlens doctor
```

此命令检查平台、权重目录和可选适配器目录，不读取聊天。目录存在不等于文件完整；模型的实际加载仍需在应用内验证。

Windows、Linux 和 Intel Mac 可以运行合成演示、JSON 导入和纯 Python 规则模块，当前不支持完整 MLX 模型及原生窗口流程。

## JSON 历史格式

最小文件如下：

```json
{
  "contacts": [
    {
      "id": "example-friend-a",
      "name": "示例好友",
      "messages": [
        {"role": "self", "text": "周六三点一起去看展？", "timestamp": 1735689600},
        {"role": "other", "text": "好呀，我来找见面的地方。", "timestamp": 1735689660}
      ]
    }
  ]
}
```

```sh
chatlens import-json examples/conversation.json
```

`timestamp` 为非负整数 Unix 秒，角色只能是 `self`、`other`、`unknown`。联系人 `id` 是稳定身份，`name` 用于显示。请准确标注双方角色；程序不会通过句子语气猜测发送者。

每条消息还可提供非负整数 `local_id`、`sort_seq`、`kind`，以及 `server_id` 和 `source`。默认 `kind=1`、`server_id="0"`、`source="json-import"`，默认本地 ID 与顺序来自数组位置。分批导入时应提供稳定的消息 ID；没有服务端 ID 时，用稳定 `source + local_id` 避免把不同消息当作同一条覆盖。

单文件不超过 20 MB，最多 1000 个联系人，单个联系人最多 100000 条消息。导入前先校验整个文件；导入后保存在本机 SQLite。更大的历史可分批整理，不要把真实导出文件放进仓库的 `examples/`。

## 数据与配置

主要环境变量如下，需在启动命令之前设置。`.env.example` 只作配置参考，程序不会自动加载 `.env`：

| 变量 | 默认值 / 用途 |
| --- | --- |
| `CHATLENS_DATA_DIR` | 私密运行状态目录，即 `DATA_DIR` |
| `CHATLENS_MODELS_DIR` | `~/.cache/chatlens/models`，三个模型的本地目录 |
| `CHATLENS_BUILD_DIR` | 仓库 `build/`，原生构建输出 |
| `CHATLENS_PROJECT_DIR` | 源码开发时通常自动推断的仓库根目录 |
| `CHATLENS_WXEXPORT_PATH` | 可选外部适配器的项目根目录，需包含 `src/wxexport` |
| `CHATLENS_TARGETS` | 可选 UI 联系人别名 JSON 数组，不代表授权分析 |
| `CHATLENS_DEBUG_CHAT` | 默认 `0`；设为 `1` 才保存包含聊天内容的调试结果 |

`DATA_DIR` 默认位置：

- macOS：`~/Library/Application Support/ChatLensOSS`
- Windows：`%LOCALAPPDATA%\ChatLens`
- Linux：`${XDG_DATA_HOME:-~/.local/share}/chatlens`

历史数据库、联系人授权、会话凭据、解密快照和诊断输出应始终保留在这些私密目录，不随源码发布。要使用自定义位置，例如 macOS/Linux 下：

```sh
export CHATLENS_DATA_DIR="$HOME/.local/share/chatlens-private"
export CHATLENS_MODELS_DIR="$HOME/.cache/chatlens/models"
```

不要把已有版本的私密目录直接复制进这个仓库，也不要提交带 token 的服务输出。提交问题时仅提供合成复现与去标识的错误类型。

## 可选：接入微信本地数据

这部分需要兼容的外部 `wxexport` 模块、当前账号已存在的有效数据库密钥，以及当前微信数据库目录。项目不包含微信登录、获取密钥或绕过系统权限的自动流程，也不保证兼容全部微信版本。

外部模块应提供 `wxexport.keys.map`、`wxexport.decrypt` 和 `wxexport.msgparse` 所用接口。把其项目根目录配置到 `CHATLENS_WXEXPORT_PATH`，例如：

```sh
export CHATLENS_WXEXPORT_PATH="/path/to/wxexport-project"
```

在 `DATA_DIR` 中自行配置两个文件：

- `active-source.json`：对象包含 `db_dir`，指向自己的 `db_storage` 目录。
- `runtime-candidates.txt`：已有数据库密钥，每行 64 个十六进制字符（32 字节）。不同数据库可能需要不同密钥。

配置形式：

```json
{"db_dir": "/path/to/your/db_storage"}
```

仅配置路径不代表数据可读。适配器会先验证密钥，复制数据库与 WAL，解密副本并检查完整性，再导入精确联系人。原始微信数据库不被写入。未配置或密钥不匹配时，先使用 JSON 路径验证其余功能。

原生点击定位需要已有的 macOS 辅助功能权限；启用 OCR 回退时还需要屏幕录制权限。权限必须由用户在系统设置中管理。不同微信版本对辅助功能树的支持不同；如果点击定位不可用，可以在历史主窗口精确选句。

卡片是独立伴随窗口，可从应用菜单选择自由拖动或贴靠微信。它不会作为聊天消息发送，也不会修改官方微信客户端。
