"""Launch only owned processes; never write session credentials to console logs."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

from .config import BUILD_DIR, DATA_DIR, PACKAGE_DIR, PROJECT_DIR, contact_aliases, prepare_private_dirs


def read_session(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    port, token = value.get("port"), value.get("token")
    if type(port) is not int or not 0 < port < 65536 or not isinstance(token, str) or len(token) != 64:
        return None
    if any(c not in "0123456789abcdef" for c in token):
        return None
    return {"port": port, "token": token}


def service_available(session):
    if not session:
        return False
    request = urllib.request.Request(
        f"http://127.0.0.1:{session['port']}/api/status", headers={"X-ChatLens-Token": session["token"]}
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status == 200 and isinstance(json.load(response), dict)
    except (OSError, ValueError):
        return False


def stop_owned(path, needle):
    try:
        pid = json.loads(path.read_text())["pid"]
        if type(pid) is not int or pid <= 1 or pid == os.getpid():
            return
        args = subprocess.check_output(["ps", "-p", str(pid), "-o", "args="], text=True)
        if needle not in args:
            return
        os.kill(pid, signal.SIGCONT)
        os.kill(pid, signal.SIGTERM)
        for _ in range(40):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        raise RuntimeError("旧进程仍在退出，请稍后再启动")
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        return


def _write_state(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.chmod(0o600)
    os.replace(temporary, path)


def launch(restart=False, main=False):
    if sys.platform != "darwin":
        raise RuntimeError("原生伴随窗口目前仅支持 macOS；可先运行 chatlens demo")
    app = BUILD_DIR / "ChatLens.app"
    observer = BUILD_DIR / "WeChatClickObserver"
    if not (app / "Contents/MacOS/ChatLens").is_file() or not observer.is_file():
        raise RuntimeError("请先在项目目录运行 python scripts/build_macos.py")
    os.umask(0o077)
    prepare_private_dirs()
    import fcntl

    with (DATA_DIR / "launch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("另一个启动操作正在进行") from None
        session_path = DATA_DIR / "bridge-session.json"
        session = read_session(session_path)
        environment = dict(os.environ)
        environment.update(
            CHATLENS_DATA_DIR=str(DATA_DIR),
            CHATLENS_PRIVATE=str(DATA_DIR),
            CHATLENS_BUILD_DIR=str(BUILD_DIR),
            CHATLENS_PYTHON=sys.executable,
            CHATLENS_DIAGNOSTICS=str(DATA_DIR / "diagnostics"),
            CHATLENS_SCREENSHOTS=str(DATA_DIR / "screenshots"),
            CHATLENS_ALL_CONTACTS="1",
            CHATLENS_TARGETS=json.dumps(contact_aliases(), ensure_ascii=False),
        )
        environment["PYTHONPATH"] = str(PACKAGE_DIR.parent) + os.pathsep + environment.get("PYTHONPATH", "")
        if restart:
            stop_owned(DATA_DIR / "backend-process.json", "-m chatlens serve")
            session = None
        if not service_available(session):
            if session:
                environment.update(CHATLENS_PORT=str(session["port"]), CHATLENS_TOKEN=session["token"])
            else:
                environment.pop("CHATLENS_PORT", None)
                environment.pop("CHATLENS_TOKEN", None)
            # stdout contains the native bootstrap token: intentionally discard it.
            child = subprocess.Popen(
                [sys.executable, "-m", "chatlens", "serve"],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            _write_state(DATA_DIR / "backend-process.json", {"pid": child.pid, "owner": "chatlens"})
            for _ in range(120):
                if child.poll() is not None:
                    raise RuntimeError("本地服务退出，请运行 chatlens serve 查看错误")
                session = read_session(session_path)
                if service_available(session):
                    break
                time.sleep(0.25)
            else:
                child.terminate()
                raise RuntimeError("本地服务启动超时")
        environment.update(CHATLENS_PORT=str(session["port"]), CHATLENS_TOKEN=session["token"])
        stop_owned(DATA_DIR / "native-process.json", str(app / "Contents/MacOS/ChatLens"))
        stop_owned(BUILD_DIR / "click-observer-process.json", str(observer))
        child = subprocess.Popen(
            [str(observer)],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _write_state(BUILD_DIR / "click-observer-process.json", {"pid": child.pid, "owner": "chatlens"})
        command = [
            "/usr/bin/open",
            "-n",
            str(app),
            "--args",
            str(PROJECT_DIR),
            str(DATA_DIR),
            sys.executable,
            str(BUILD_DIR),
        ]
        if not main:
            command.append("--in-wechat")
        subprocess.run(command, env=environment, check=True)
        native_pid = None
        for _ in range(40):
            found = subprocess.run(["pgrep", "-x", "ChatLens"], capture_output=True, text=True)
            for candidate in found.stdout.splitlines():
                detail = subprocess.run(["ps", "-p", candidate, "-o", "args="], capture_output=True, text=True)
                if str(app / "Contents/MacOS/ChatLens") in detail.stdout:
                    native_pid = int(candidate)
                    break
            if native_pid:
                break
            time.sleep(0.1)
        if not native_pid:
            raise RuntimeError("原生窗口未启动，本地服务仍可使用")
        _write_state(DATA_DIR / "native-process.json", {"pid": native_pid, "owner": "chatlens"})
    print("ChatLens 已启动；模型加载进度可在窗口查看。")
