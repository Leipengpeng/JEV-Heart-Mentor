"""Paths and private state live outside the installed source tree."""

import json
import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("CHATLENS_PROJECT_DIR", PACKAGE_DIR.parents[1])).expanduser().resolve()


def default_data_dir():
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/ChatLensOSS"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "ChatLens"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "chatlens"


DATA_DIR = Path(os.environ.get("CHATLENS_DATA_DIR", default_data_dir())).expanduser().resolve()
BUILD_DIR = Path(os.environ.get("CHATLENS_BUILD_DIR", PROJECT_DIR / "build")).expanduser().resolve()
MODELS_DIR = Path(os.environ.get("CHATLENS_MODELS_DIR", Path.home() / ".cache/chatlens/models")).expanduser().resolve()
WXEXPORT_PATH = (
    Path(os.environ["CHATLENS_WXEXPORT_PATH"]).expanduser().resolve()
    if os.environ.get("CHATLENS_WXEXPORT_PATH")
    else None
)


def contact_aliases():
    """Optional UI aliases never imply consent to analyze a contact."""
    raw = os.environ.get("CHATLENS_TARGETS", "[]")
    try:
        names = json.loads(raw)
    except (ValueError, TypeError):
        return ()
    if not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() or len(n) > 100 for n in names):
        return ()
    return tuple(dict.fromkeys(n.strip() for n in names))


def prepare_private_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    DATA_DIR.chmod(0o700)
    for name in ("diagnostics", "screenshots", "reports"):
        folder = DATA_DIR / name
        folder.mkdir(exist_ok=True, mode=0o700)
        folder.chmod(0o700)


def private_debug_enabled():
    return os.environ.get("CHATLENS_DEBUG_CHAT", "0") == "1"
