# 第三方组件与模型

ChatLens 自写源码使用 [MIT](LICENSE)。模型权重、依赖库与外部工具遵循各自上游许可证，不受本仓库 MIT 替代。仓库及源码 ZIP 不附带权重、第三方源码、第三方二进制或聊天数据。

| 组件 | 上游声明 | 来源 |
| --- | --- | --- |
| Typical-small 决策权重 | Apache-2.0 | [OzLabs/typical-small 模型卡](https://huggingface.co/OzLabs/typical-small) |
| Qwen3-1.7B-Base 底座 | Apache-2.0 | [固定版本 LICENSE](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/LICENSE) |
| MiniCPM5-2B-MLX 权重 | Apache-2.0 | [模型卡](https://huggingface.co/openbmb/MiniCPM5-2B-MLX)、[项目 LICENSE](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) |
| typical-ai 运行库 | Apache-2.0 | [LICENSE](https://github.com/ozlabsai/typical/blob/main/LICENSE) |
| MLX / MLX-LM | MIT | [MLX](https://github.com/ml-explore/mlx/blob/main/LICENSE)、[MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE) |
| lm-format-enforcer | MIT | [LICENSE](https://github.com/noamgat/lm-format-enforcer/blob/main/LICENSE) |
| PyTorch / Transformers 等安装依赖 | 见各发行包的许可证与 notices | [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE)、[Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| 可选 wxexport 数据适配器 | WTFPL | [LICENSE](https://github.com/TimFang4162/wechat4-macos-frida-export/blob/main/LICENSE) |
| 可选 pilk 语音解码组件 | GPL v3 | [LICENSE](https://github.com/foyoux/pilk/blob/main/LICENSE) |

上表记录发布整理时核对的上游声明。Typical 模型卡另披露部分训练数据集许可证未声明；本项目没有独立核验训练数据授权。

模型下载固定到源码中的 revision，每个 LFS 权重文件核对上游 SHA-256；下载者需保留对应模型目录内的 LICENSE / 模型卡。Typical 是 JEV 类开源替代物，并非官方 JEV 权重。

语音为实验性可选功能，基础安装不包含 `pilk`。若自行安装或再分发包含 GPL 组件的组合，需按该组件条款处理；不要把整个组合笼统标为 MIT。

`knowledge/galgame/` 内动作释义、场景示例和提示词是本项目原创，发布包不包含商业游戏脚本、抓取页面正文或研究摘录。风格参考不代表获得游戏内容的再分发许可。
