"""Small command surface; heavy model dependencies load only on demand."""

import argparse
import json
import platform
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="chatlens", description="ChatLens 本地对话决策卡")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="使用合成对话体验界面，无需模型或微信")
    demo.add_argument("--port", type=int, default=0)
    demo.add_argument("--open", action="store_true", help="打开默认浏览器")
    sub.add_parser("serve", help="启动真实本地模型和历史 API")
    sub.add_parser("doctor", help="检查平台和本地权重，不读取聊天")
    sub.add_parser("download-models", help="联网下载固定版本的公开权重")
    launch = sub.add_parser("launch", help="打开已构建的 macOS 伴随窗口")
    launch.add_argument("--restart", action="store_true")
    launch.add_argument("--main", action="store_true", help="显示历史主窗口")
    importer = sub.add_parser("import-json", help="导入标准 JSON 历史；分析授权在界面单独确认")
    importer.add_argument("path")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import main as demo_main

            demo_main(args.port, args.open)
        elif args.command == "serve":
            from .server import main as serve_main

            serve_main()
        elif args.command == "download-models":
            from .download_models import main as download_main

            download_main()
        elif args.command == "launch":
            from .launcher import launch as launch_main

            launch_main(restart=args.restart, main=args.main)
        elif args.command == "import-json":
            from .import_history import import_json

            print(json.dumps(import_json(args.path), ensure_ascii=False))
        elif args.command == "doctor":
            from .config import MODELS_DIR, WXEXPORT_PATH

            print(
                json.dumps(
                    {
                        "python": platform.python_version(),
                        "platform": sys.platform,
                        "machine": platform.machine(),
                        "native_supported": sys.platform == "darwin" and platform.machine() == "arm64",
                        "models": {
                            name: (MODELS_DIR / name).is_dir()
                            for name in ("typical-small-1.7b", "qwen3-1.7b-base", "minicpm5-2b-mlx")
                        },
                        "external_wechat_adapter": WXEXPORT_PATH is not None
                        and (WXEXPORT_PATH / "src/wxexport").is_dir(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"ChatLens: {type(exc).__name__}: {exc}\n")
