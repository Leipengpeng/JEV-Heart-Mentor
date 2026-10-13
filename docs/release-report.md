# 0.1.0 发布整理与验证

日期：2026-10-12。整理后的项目为独立源码仓库，原实验目录与私人运行数据未迁入。

## 已完成

- `src/chatlens` 标准 Python 包、CLI、构建配置和代码格式化。
- `macos` 自写原生窗口与点击定位代码；`tests`、`examples`、`docs`、`.github` 各自独立。
- 微信风格卡片整理，三个主区块、前三项情感、10 分制显示；改进窄窗口、长文、空白、加载和键盘焦点。
- 连续点击、搜索/分页、切换联系人及旧后台结果分别做顺序保护；实际推理槽独立于 UI 卡片状态。
- API 输入和错误返回加固；诊断保存失败不影响成功结果；默认关闭完整私密调试输出。
- 新增合成演示、JSON 导入与离线历史授权路径；无需微信密钥即可分析已导入的 JSON。
- MIT 源码许可证、第三方 notices、隐私说明、源码扫描、源码 ZIP 和 Python 构建产物。

## 检查结果

| 检查 | 结果 |
| --- | --- |
| Python 回归 | 249 项通过；本机包含可选 tokenizer / MLX 检查 |
| JavaScript 选择与显示回归 | 52 项通过 |
| Swift 合成自检 | 47 项通过；原生代码编译通过 |
| 浏览器界面 | 真实 Chrome 检查通过；切句更新、切人清空、三个主区块、情感前三项、窄屏无横向溢出；无浏览器错误 |
| 干净虚拟环境 | 基础安装、CLI、JSON 导入、无模型 demo 通过 |
| Python 打包 | wheel 与 sdist 构建通过 |
| 脱敏 | 源码白名单/常见凭据/机器路径/账号 ID 扫描无命中；本机额外私密姓名与长聊天片段交叉复核无命中 |

Python、JS、Swift 合计 348 项检查通过。Linux CI 会跳过仅 Apple Silicon 可执行的格式解码检查；未下载 tokenizer 时也跳过对应可选检查。

## 真实本地模型联调

使用公开包代码和现有本机权重，只分析仓库内四条原创合成对话。没有读取真实聊天，没有调用外部模型。

| 项目 | 单次结果 |
| --- | ---: |
| 模型服务启动至可用 | 4.83 秒 |
| 完整分析 | 7.65 秒 |
| MiniCPM 生成 | 6.13 秒，1 次调用 |
| JEV 式检索 + 动作 + 情感 | 0.72 秒 |
| 参与评分的动作 / 情感 | 31 / 9 |

所选消息锚点保留，MiniCPM 没有预筛动作，推理后 worker 正常释放，默认没有产生完整失败输入/输出日志。

这是一条短合成对话的集成检查，不能据此推断长历史延迟、情感准确率或回复效果。已有 GPU 计算在切换联系人后会继续运行到结束；结果被忽略，新任务等待实际 worker 退出。

原生编译与自检未替代真人鼠标操作、系统授权和不同微信版本的兼容测试。项目尚未经过独立安全审计。

## 可复现命令

```sh
pip install -e '.[dev,test]'
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m unittest discover -s tests
node --test tests/*.js
python scripts/check_release.py
python -m build
python scripts/export_release.py
```

源码 ZIP 仅来自通过检查的白名单文件；不包含模型、数据库、密钥、第三方仓库、运行日志、历史报告或本机 `.app`。GitHub Actions 已写好，尚未在远端执行。

## 公开仓库

项目公开仓库为 [Leipengpeng/JEV-Heart-Mentor](https://github.com/Leipengpeng/JEV-Heart-Mentor)。本地继续提交的命令：

```sh
git add .
python scripts/check_release.py
git diff --cached --stat
git commit -m "Initial open-source release"
git push -u origin main
```

只发布已脱敏源码、合成示例、文档和测试；私人运行目录与模型权重不上传。
