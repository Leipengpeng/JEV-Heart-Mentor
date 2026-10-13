"""Build the two first-party macOS binaries; does not launch or request access."""

import platform
import plistlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chatlens.config import BUILD_DIR


def main():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("The current native/model build targets Apple Silicon macOS 14+.")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    app = BUILD_DIR / "ChatLens.app"
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(exist_ok=True)
    common = ["xcrun", "swiftc", "-swift-version", "5", "-O"]
    frameworks = ["-framework", "Cocoa", "-framework", "ScreenCaptureKit", "-framework", "Vision"]
    subprocess.run(
        common
        + [str(ROOT / "macos/ChatLens.swift"), "-o", str(contents / "MacOS/ChatLens")]
        + frameworks
        + ["-framework", "WebKit"],
        check=True,
    )
    subprocess.run(
        common
        + [str(ROOT / "macos/WeChatClickObserver.swift"), "-o", str(BUILD_DIR / "WeChatClickObserver")]
        + frameworks
        + ["-framework", "ApplicationServices"],
        check=True,
    )
    info = {
        "CFBundleIdentifier": "org.chatlens.opensource",
        "CFBundleName": "ChatLens",
        "CFBundleDisplayName": "ChatLens",
        "CFBundleExecutable": "ChatLens",
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "1",
        "CFBundleShortVersionString": "0.1.0",
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        "NSScreenCaptureUsageDescription": "在辅助功能无法定位消息时，本机识别微信可见窗口，不上传聊天。",
    }
    with (contents / "Info.plist").open("wb") as handle:
        plistlib.dump(info, handle)
    subprocess.run(
        ["codesign", "--force", "--sign", "-", "--identifier", info["CFBundleIdentifier"], str(app)], check=True
    )
    print("Built build/ChatLens.app and build/WeChatClickObserver (not launched).")


if __name__ == "__main__":
    main()
