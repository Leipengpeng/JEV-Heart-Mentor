"""Loopback-only UI/API. Chat content remains in memory; one inference worker."""

import json
import os
import secrets
import sys
import threading
import time
import socket
import hashlib
import copy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .analysis_engine import Engine, validate_messages
from .history_store import HistoryStore, DEFAULT_ANALYSIS_HISTORY_LIMIT
from .history_connector import runtime_status, sync_history, history_source_signature
from .card_engine import VERSION as CARDS_VERSION
from .click_selection import resolve as resolve_click
from .contact_catalog import ContactCatalog
from .decision_audit import record_decision

from .config import PACKAGE_DIR, DATA_DIR, BUILD_DIR, contact_aliases, prepare_private_dirs, private_debug_enabled

ROOT = PACKAGE_DIR
SUPPORTED_TARGETS = contact_aliases()
CONTACT_STATE = DATA_DIR / "contact-state.json"
try:
    saved_contact = json.loads(CONTACT_STATE.read_text())
    initial_target = saved_contact.get("target") if isinstance(saved_contact, dict) else None
except (OSError, ValueError):
    initial_target = None
if not isinstance(initial_target, str) or not initial_target:
    initial_target = None
TOKEN = os.environ.get("CHATLENS_TOKEN") or secrets.token_hex(32)
LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-inference")
STATE = {
    "ready": False,
    "phase": "正在准备本地模型",
    "error": None,
    "job": None,
    "cards_version": CARDS_VERSION,
    "worker_job_id": None,
    "worker_busy": False,
    "selection": None,
    "click_observer": None,
    "active_target": initial_target,
    "contact_revision": secrets.token_hex(12),
    "display_name": initial_target,
    "contact_observed_at": 0,
    "pending_locator": None,
    "contact_status": "loading",
    "enabling_revision": None,
}
RESULT_CACHE = OrderedDict()
ENGINE = None
HISTORY = None
HISTORY_LOCK = threading.Lock()
CATALOG = None
CATALOG_LOCK = threading.Lock()


def history_store():
    global HISTORY
    with HISTORY_LOCK:
        if HISTORY is None:
            HISTORY = HistoryStore(DATA_DIR / "memory.sqlite3")
    return HISTORY


def contact_catalog():
    global CATALOG
    with CATALOG_LOCK:
        if CATALOG is None:
            CATALOG = ContactCatalog(DATA_DIR, ())
        CATALOG.refresh(history_store())
        catalog = CATALOG
    with LOCK:
        target = STATE.get("active_target")
        if target and target not in catalog.records:
            contact, method = catalog.resolve(target, store=history_store())
            if contact:
                STATE.update(active_target=contact["id"], display_name=contact["name"], contact_status="recognized")
            else:
                STATE.update(
                    active_target=None, display_name=str(target), selection=None, job=None, contact_status=method
                )
    return catalog


def contact_fields_locked():
    catalog = CATALOG
    target = STATE.get("active_target")
    record = catalog.records.get(target, {}) if catalog else {}
    enabled = bool(catalog and catalog.enabled(target))
    return {
        "active_target": target,
        "display_name": STATE.get("display_name") or "",
        "contact_revision": STATE["contact_revision"],
        "contact_observed_at": STATE.get("contact_observed_at", 0),
        "analysis_enabled": enabled,
        "requires_consent": bool(target and not enabled),
        "can_analyze": bool(target and record.get("supported", False)),
        "status": STATE.get("contact_status", "loading"),
        "contact_status": STATE.get("contact_status", "loading"),
        "contact_catalog_revision": catalog.revision if catalog else "",
        "contact_source": catalog.source if catalog else "",
    }


def set_contact_locked(target, clear=False, display_name=None, status="recognized", observed=True):
    name = display_name or ((CATALOG.records.get(target) or {}).get("name") if CATALOG else None) or target or ""
    changed = target != STATE.get("active_target") or (target is None and name != STATE.get("display_name"))
    if changed or clear:
        STATE.update(
            active_target=target,
            display_name=name,
            contact_revision=secrets.token_hex(12),
            selection=None,
            job=None,
            pending_locator=None,
            enabling_revision=None,
            contact_status=status,
        )
        if observed:
            STATE["contact_observed_at"] = time.time()
        CONTACT_STATE.parent.mkdir(parents=True, exist_ok=True)
        CONTACT_STATE.write_text(json.dumps({"target": target}, ensure_ascii=False))
        CONTACT_STATE.chmod(0o600)
    else:
        STATE.update(display_name=name, contact_status=status)
    return contact_fields_locked()


def resolve_target(value):
    catalog = contact_catalog()
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError("联系人格式错误")
    contact, method = catalog.resolve(value.strip(), store=history_store())
    if contact is None:
        raise ValueError("联系人未找到或名称重复，请选择唯一联系人")
    return contact["id"]


def stale_contact(request, target):
    revision = request.get("contact_revision")
    return revision is not None and (revision != STATE["contact_revision"] or target != STATE.get("active_target"))


def make_selection(resolved, target, request):
    # A clicked message anchors the other person's analysis. Own messages are
    # background, never silently substituted as the object of judgement.
    if resolved.get("status") == "matched":
        role = (resolved.get("message") or {}).get("role")
        if role != "other":
            resolved = {**resolved, "status": "self_message" if role == "self" else "unknown_role"}
    location = request.get("location")
    if location is not None:
        if not isinstance(location, dict) or any(
            not isinstance(location.get(k), (int, float)) or not (-100000 < location[k] < 100000)
            for k in ("x", "y", "wechat_pid")
        ):
            raise ValueError("点击位置格式错误")
        location = {k: location[k] for k in ("x", "y", "wechat_pid")}
    return {
        **resolved,
        "revision": secrets.token_hex(12),
        "target": target,
        "location": location,
        "selected_at": time.time(),
        "contact_revision": STATE["contact_revision"],
    }


def install_local_network_guard():
    original = socket.socket.connect
    original_ex = socket.socket.connect_ex

    def check(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise RuntimeError("本地分析服务禁止外部网络连接")

    def connect(sock, address):
        check(sock, address)
        return original(sock, address)

    def connect_ex(sock, address):
        check(sock, address)
        return original_ex(sock, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex


def record_check(**details):
    # Only integration metadata; never persist message text or generated replies.
    path = BUILD_DIR / "live-check-report.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            previous = json.loads(path.read_text()) if path.exists() else {}
        except (ValueError, OSError):
            previous = {}
        if not isinstance(previous, dict):
            previous = {}
        previous.update(details, checked_at=time.time())
        path.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n")
        return True
    except (OSError, ValueError, TypeError):
        return False


def best_effort(callback, *args, **kwargs):
    """Optional local caches/diagnostics cannot invalidate a completed card."""
    try:
        return callback(*args, **kwargs)
    except Exception:
        return None


def current_job_locked(job_id, request):
    job = STATE.get("job")
    if not job or job.get("id") != job_id:
        return False
    revision = request.get("contact_revision")
    return revision is None or (
        revision == STATE["contact_revision"]
        and (not request.get("target") or request["target"] == STATE.get("active_target"))
    )


def public_error(exc):
    # Validation errors are authored user guidance. Unexpected errors may carry
    # private paths, database contents, or model inputs in their exception text.
    if isinstance(exc, ValueError):
        return str(exc)[:240]
    return "本地服务暂时不可用，请重试（" + type(exc).__name__ + "）"


def phase(text):
    with LOCK:
        if isinstance(text, dict):
            STATE["phase"] = text["text"]
            if STATE["job"]:
                for key in ("partial_scores", "proposal"):
                    if key in text:
                        STATE["job"][key] = text[key]
        else:
            STATE["phase"] = text


def initialize():
    global ENGINE
    try:
        contact_catalog()
        ENGINE = Engine(phase)
        with LOCK:
            STATE.update(ready=True, phase="本地模型已就绪")
    except Exception as exc:
        with LOCK:
            STATE.update(error=f"模型加载失败：{type(exc).__name__}: {exc}", phase="加载失败")


def run_job(job_id, request):
    try:
        with LOCK:
            if not current_job_locked(job_id, request):
                return
            if STATE.get("worker_job_id") not in (None, job_id):
                return
            STATE.update(worker_job_id=job_id, worker_busy=True)

        def lookup(queries):
            rows = []
            seen = set(request.get("history_ids", []))
            watermark = tuple(request["history_watermark"])
            for query in queries[:3]:
                for row in history_store().search(request["target"], query, 20, watermark):
                    if row["id"] in seen or (row["timestamp"], row["sort_seq"], row["id"]) > watermark:
                        continue
                    seen.add(row["id"])
                    rows.append({k: row[k] for k in ("id", "timestamp", "role", "text")})
                    rows[-1]["text"] = rows[-1]["text"][:800]
                    if len(rows) >= 8:
                        return rows
            return rows

        def job_phase(value):
            with LOCK:
                if not current_job_locked(job_id, request):
                    return
                STATE["phase"] = value["text"] if isinstance(value, dict) else value
                if isinstance(value, dict):
                    for key in ("partial_scores", "proposal"):
                        if key in value:
                            STATE["job"][key] = value[key]

        result = ENGINE.analyze(request, job_phase, lookup if request.get("source") == "history" else None)
        result["context_signature"] = request.get("context_signature")
        result["selected_message"] = request.get("selected_message")
        result["target"] = request.get("target")
        result["contact_revision"] = request.get("contact_revision")
        with LOCK:
            if current_job_locked(job_id, request):
                STATE["job"].update(status="done", result=result)
                RESULT_CACHE[STATE["job"]["fingerprint"]] = copy.deepcopy(STATE["job"])
                RESULT_CACHE.move_to_end(STATE["job"]["fingerprint"])
                while len(RESULT_CACHE) > 16:
                    RESULT_CACHE.popitem(last=False)
                STATE["phase"] = "分析完成"
                # Keep the identity check and optional persistence together so
                # a contact switch cannot race a late write from the old job.
                best_effort(record_decision, DATA_DIR / "decision-events.jsonl", request, result, "done", CARDS_VERSION)
                if request.get("source") == "history":
                    best_effort(
                        lambda: history_store().save_summary(
                            request["target"], request["history_ids"], result["analysis"]
                        )
                    )
                if (
                    private_debug_enabled()
                    and result.get("workflow_version") == CARDS_VERSION
                    and result.get("source") == "history"
                ):
                    private = DATA_DIR / "latest-card.json"

                    def save_latest():
                        private.write_text(json.dumps(STATE["job"], ensure_ascii=False))
                        private.chmod(0o600)

                    best_effort(save_latest)
                best_effort(
                    record_check,
                    analysis_status="passed",
                    source=result["source"],
                    messages_used=result["messages_used"],
                    elapsed_seconds=result["elapsed_seconds"],
                    reply_count=len(result["analysis"]["replies"]),
                    speaker_needs_review=result["speaker_needs_review"],
                    complete_structured_output=True,
                )
    except Exception as exc:
        with LOCK:
            if current_job_locked(job_id, request):
                STATE["job"].update(status="error", error=public_error(exc))
                STATE["phase"] = "分析未完成"
                best_effort(
                    record_decision,
                    DATA_DIR / "decision-events.jsonl",
                    request,
                    STATE["job"].get("partial_scores", {}),
                    "error",
                    CARDS_VERSION,
                )
    finally:
        with LOCK:
            if STATE.get("worker_job_id") == job_id:
                STATE.update(worker_job_id=None, worker_busy=False)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *args):
        pass  # No chat text, tokens, or HTTP headers in logs.

    def respond(self, status, value, content_type="application/json; charset=utf-8"):
        data = json.dumps(value, ensure_ascii=False).encode() if isinstance(value, (dict, list)) else value.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
        )
        try:
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass  # A closed UI tab must not emit a private request traceback.

    def authorized(self):
        host = self.headers.get("Host", "").split(":")[0]
        return host in ("127.0.0.1", "localhost") and secrets.compare_digest(
            self.headers.get("X-ChatLens-Token", "").encode("utf-8"), TOKEN.encode("utf-8")
        )

    def do_GET(self):
        try:
            return self.handle_get()
        except Exception as exc:
            return self.respond(500, {"error": public_error(exc)})

    def handle_get(self):
        if self.path in ("/cards.js", "/cards.css") and self.headers.get("Host", "").split(":")[0] in (
            "127.0.0.1",
            "localhost",
        ):
            name = self.path[1:]
            return self.respond(
                200,
                (ROOT / "web" / name).read_text(),
                "text/javascript; charset=utf-8" if name.endswith(".js") else "text/css; charset=utf-8",
            )
        if not self.authorized():
            return self.respond(403, {"error": "本地会话验证失败"})
        if self.path in ("/", "/plugin"):
            html = (ROOT / "web/cards.html").read_text().replace("__SESSION_TOKEN__", TOKEN)
            html = html.replace(
                "window.CARD_CONTACTS=[];",
                "window.CARD_CONTACTS="
                + json.dumps(SUPPORTED_TARGETS, ensure_ascii=False).replace("<", "\\u003c")
                + ";",
            )
            return self.respond(200, html, "text/html; charset=utf-8")
        if self.path == "/api/status":
            with LOCK:
                return self.respond(200, {**STATE, **contact_fields_locked()})
        if self.path == "/api/history/status":
            return self.respond(200, {**runtime_status(), **history_store().stats()})
        if self.path == "/api/contacts":
            catalog = contact_catalog()
            with LOCK:
                active = contact_fields_locked()
            return self.respond(
                200,
                {
                    "contacts": catalog.list(),
                    **active,
                    "catalog_error": catalog.error,
                    "total_contacts": len(catalog.records),
                    "available_contacts": sum(r["available"] for r in catalog.records.values()),
                },
            )
        return self.respond(404, {"error": "Not found"})

    def do_POST(self):
        if not self.authorized():
            return self.respond(403, {"error": "本地会话验证失败"})
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.respond(403, {"error": "Origin rejected"})
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if (
                self.headers.get("Transfer-Encoding")
                or len(lengths) != 1
                or not lengths[0].isascii()
                or not lengths[0].isdigit()
            ):
                return self.respond(400, {"error": "请求长度格式错误"})
            length = int(lengths[0])
            if not 0 < length <= 100000:
                return self.respond(413, {"error": "输入过大或为空"})
            raw = self.rfile.read(length)
            if len(raw) != length:
                return self.respond(400, {"error": "请求内容未完整传输"})

            def reject_constant(_):
                raise ValueError("请求含有非法数值")

            request = json.loads(raw, parse_constant=reject_constant)
            if not isinstance(request, dict):
                raise ValueError("请求格式错误")
            if self.path == "/api/click-observer/status":
                safe = {
                    k: bool(request.get(k)) for k in ("running", "mouse_monitor", "accessibility", "screen_capture")
                }
                for key, allowed in (
                    ("ax_capability", ("available", "ax_collapsed", "unavailable")),
                    ("locator_source", ("ax", "ocr", "none")),
                ):
                    value = request.get(key)
                    if value in allowed:
                        safe[key] = value
                for key in ("header_ocr_count", "ax_elapsed_ms"):
                    value = request.get(key)
                    if isinstance(value, (int, float)) and 0 <= value <= 100000:
                        safe[key] = value
                counts = request.get("ax_counts")
                if isinstance(counts, dict):
                    safe["ax_counts"] = {
                        k: v
                        for k, v in counts.items()
                        if k in ("nodes", "bubbles", "virtual_cells", "titles", "message_lists")
                        and isinstance(v, int)
                        and 0 <= v <= 100000
                    }
                safe["checked_at"] = time.time()
                with LOCK:
                    STATE["click_observer"] = safe
                return self.respond(200, {"ok": True})
            if self.path == "/api/contact/observe":
                title = request.get("title", "")
                texts = request.get("visible_texts", [])
                if (
                    not isinstance(title, str)
                    or len(title) > 200
                    or not isinstance(texts, list)
                    or len(texts) > 30
                    or any(not isinstance(t, str) or len(t) > 800 for t in texts)
                ):
                    raise ValueError("会话定位格式错误")
                catalog = contact_catalog()
                contact, method = catalog.resolve(title, texts, history_store())
                with LOCK:
                    state = set_contact_locked(
                        contact["id"] if contact else None,
                        display_name=contact["name"] if contact else title,
                        status="recognized" if contact else method,
                    )
                return self.respond(200, {**state, "resolution_method": method})
            if self.path == "/api/contact/select":
                target = resolve_target(request.get("target"))
                with LOCK:
                    contact = set_contact_locked(target, clear=True)
                return self.respond(200, contact)
            if self.path in ("/api/contact/enable", "/api/contact/defer"):
                target = resolve_target(request.get("target"))
                with LOCK:
                    if stale_contact(request, target) or target != STATE.get("active_target"):
                        return self.respond(409, {"error": "联系人已切换，已忽略旧确认"})
                    revision = STATE["contact_revision"]
                    prior_selection_revision = (STATE.get("selection") or {}).get("revision")
                    if self.path.endswith("/defer"):
                        STATE["pending_locator"] = None
                        return self.respond(200, {**contact_fields_locked(), "deferred": True})
                    contact_catalog_ref = CATALOG
                    contact_catalog_ref.enable(target)
                    STATE["enabling_revision"] = revision
                store = history_store()
                try:
                    sync = sync_history(store, target)
                except Exception:
                    with LOCK:
                        if STATE.get("enabling_revision") == revision:
                            STATE["enabling_revision"] = None
                    raise
                with LOCK:
                    if revision != STATE["contact_revision"] or target != STATE.get("active_target"):
                        return self.respond(409, {"error": "联系人已切换，已忽略旧分析位置"})
                    STATE["enabling_revision"] = None
                    existing = STATE.get("selection") or {}
                    if existing and existing.get("revision") != prior_selection_revision:
                        STATE["enabling_revision"] = None
                        return self.respond(
                            200,
                            {**contact_fields_locked(), "selection": existing, "sync_seconds": sync.get("seconds", 0)},
                        )
                    # Clicks arriving while the initial import runs replace the
                    # pending locator. Resolve the latest intent under the same
                    # lock as publication so an earlier point cannot overwrite it.
                    pending = STATE.get("pending_locator")
                    if pending:
                        resolved = resolve_click(store, target, pending)
                    else:
                        context = store.context(target)
                        latest = context["history_ids"][-1]
                        selected = store.context(target, through_id=latest)["selected_message"]
                        resolved = {
                            "status": "matched",
                            "message": selected,
                            "candidates": [],
                            "method": "newest-usable-database-message",
                        }
                    selection = make_selection(resolved, target, pending or {})
                    STATE.update(selection=selection, pending_locator=None, enabling_revision=None)
                    if selection["status"] in ("self_message", "unknown_role"):
                        STATE["job"] = None
                    return self.respond(
                        200, {**contact_fields_locked(), "selection": selection, "sync_seconds": sync.get("seconds", 0)}
                    )
            if self.path.startswith("/api/history/") or self.path == "/api/analyze/history":
                target = resolve_target(request.get("target", STATE.get("active_target")))
                request["target"] = target
                with LOCK:
                    if stale_contact(request, target):
                        return self.respond(409, {"error": "联系人已切换，已忽略旧位置请求"})
                if self.path == "/api/history/select-position":
                    for key in ("clicked_texts", "before_texts", "after_texts"):
                        values = request.get(key, [])
                        if (
                            not isinstance(values, list)
                            or len(values) > 8
                            or any(not isinstance(x, str) or len(x) > 800 for x in values)
                        ):
                            raise ValueError("点击定位文本格式错误")
                    with LOCK:
                        if stale_contact(request, target):
                            return self.respond(409, {"error": "联系人已切换，已忽略旧位置请求"})
                        if (
                            target == STATE.get("active_target")
                            and STATE.get("enabling_revision") == STATE["contact_revision"]
                        ):
                            STATE["pending_locator"] = dict(request)
                            return self.respond(
                                200, {**contact_fields_locked(), "status": "pending", "selection": None}
                            )
                if not CATALOG.enabled(target):
                    if self.path == "/api/history/select-position":
                        # Capture only a bounded locator. No history import, ASR,
                        # summaries or model calls occur until explicit enable.
                        for key in ("clicked_texts", "before_texts", "after_texts"):
                            values = request.get(key, [])
                            if (
                                not isinstance(values, list)
                                or len(values) > 8
                                or any(not isinstance(x, str) or len(x) > 800 for x in values)
                            ):
                                raise ValueError("点击定位文本格式错误")
                        with LOCK:
                            if stale_contact(request, target):
                                return self.respond(409, {"error": "联系人已切换，已忽略旧位置请求"})
                            if target != STATE.get("active_target"):
                                set_contact_locked(target)
                            STATE["pending_locator"] = dict(request)
                            STATE["selection"] = None
                            return self.respond(
                                200, {**contact_fields_locked(), "status": "requires_consent", "selection": None}
                            )
                    with LOCK:
                        return self.respond(403, {"error": "请先确认是否分析这个联系人", **contact_fields_locked()})
                store = history_store()
                if self.path == "/api/history/clear-selection":
                    with LOCK:
                        STATE["selection"] = None
                    return self.respond(200, {"ok": True})
                if self.path in ("/api/history/select", "/api/history/select-position"):
                    if self.path.endswith("select-position"):
                        try:
                            sync_history(store, target)
                        except ValueError as exc:
                            if str(exc) != "历史同步正在进行":
                                raise
                        resolved = resolve_click(store, target, request)
                    else:
                        context = store.context(target, through_id=request.get("message_id", ""))
                        resolved = {
                            "status": "matched",
                            "message": context["selected_message"],
                            "candidates": [],
                            "method": "database-message-id",
                        }
                    with LOCK:
                        if stale_contact(request, target):
                            return self.respond(409, {"error": "联系人已切换，已忽略旧位置请求"})
                        previous = STATE.get("selection") or {}
                        if (
                            self.path.endswith("select-position")
                            and resolved["status"] != "matched"
                            and previous.get("status") == "matched"
                            and previous.get("target") == target
                        ):
                            return self.respond(
                                200, {**make_selection(resolved, target, request), "preserved_selection": True}
                            )
                        set_contact_locked(target)
                        selection = make_selection(resolved, target, request)
                        STATE["selection"] = selection
                        if selection["status"] in ("self_message", "unknown_role"):
                            STATE["job"] = None
                    return self.respond(200, selection)
                if self.path == "/api/history/sync":
                    return self.respond(200, sync_history(store, target))
                if self.path == "/api/history/recent":
                    return self.respond(
                        200,
                        {
                            **store.recent(target, request.get("limit", 100), request.get("before")),
                            "stats": store.stats(target),
                        },
                    )
                if self.path == "/api/history/search":
                    through = (
                        store.context(target, through_id=request["through_id"])["history_watermark"]
                        if request.get("through_id")
                        else None
                    )
                    return self.respond(
                        200, {"messages": store.search(target, request.get("query", ""), through=through)}
                    )
                if self.path in ("/api/history/context", "/api/analyze/history"):
                    source_limit = request.get("source_history_limit", DEFAULT_ANALYSIS_HISTORY_LIMIT)
                    context = store.context(
                        target, request.get("query", ""), request.get("through_id"), source_history_limit=source_limit
                    )
                    context["context_signature"] = hashlib.sha256(
                        json.dumps(
                            [
                                history_source_signature(),
                                store.generation(target),
                                context["source_history_limit"],
                                context["history_ids"],
                                context["messages"],
                                request.get("through_id"),
                            ],
                            ensure_ascii=False,
                        ).encode()
                    ).hexdigest()
                    if self.path == "/api/history/context":
                        with LOCK:
                            if stale_contact(request, target):
                                return self.respond(409, {"error": "联系人已切换，已忽略旧上下文请求"})
                        return self.respond(200, context)
                    if (
                        request.get("workflow") == "llm_then_jev"
                        and (context.get("selected_message") or {}).get("role") != "other"
                    ):
                        return self.respond(
                            400,
                            {
                                "error": "请选择对方消息",
                                "status": "self_message"
                                if (context.get("selected_message") or {}).get("role") == "self"
                                else "unknown_role",
                            },
                        )
                    with LOCK:
                        if stale_contact(request, target):
                            return self.respond(409, {"error": "联系人已切换，已忽略旧分析请求"})
                        if target != STATE.get("active_target"):
                            set_contact_locked(target)
                        request["contact_revision"] = STATE["contact_revision"]
                    request.update(context, source="history")
                    if request.get("through_id"):
                        request["past_summaries"] = []
            if self.path in ("/api/analyze", "/api/analyze/history"):
                if self.path == "/api/analyze":
                    for key in (
                        "history_ids",
                        "history_evidence",
                        "past_summaries",
                        "history_watermark",
                        "context_signature",
                        "selected_message",
                        "through_id",
                    ):
                        request.pop(key, None)
                    if request.get("source") == "history":
                        request["source"] = "manual"
                    if request.get("target"):
                        request["target"] = resolve_target(request["target"])
                        if not CATALOG.enabled(request["target"]):
                            return self.respond(403, {"error": "请先确认是否分析这个联系人"})
                request["messages"] = validate_messages(request.get("messages"))
                fingerprint = hashlib.sha256(
                    (
                        CARDS_VERSION
                        + json.dumps(
                            {
                                key: request.get(key)
                                for key in (
                                    "workflow",
                                    "source",
                                    "target",
                                    "messages",
                                    "history_ids",
                                    "through_id",
                                    "query",
                                    "goal",
                                    "relationship",
                                    "context_signature",
                                    "source_history_limit",
                                )
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ).encode()
                ).hexdigest()
                with LOCK:
                    if request.get("target") and stale_contact(request, request["target"]):
                        return self.respond(409, {"error": "联系人已切换，已忽略旧分析请求"})
                    if request.get("target"):
                        if request["target"] != STATE.get("active_target"):
                            return self.respond(409, {"error": "联系人已切换，请重新选择分析位置"})
                        request["contact_revision"] = STATE["contact_revision"]
                    if not STATE["ready"]:
                        return self.respond(503, {"error": STATE["error"] or STATE["phase"]})
                    if (
                        STATE["job"]
                        and request.get("workflow") == "llm_then_jev"
                        and STATE["job"].get("fingerprint") == fingerprint
                        and STATE["job"].get("contact_revision") == request.get("contact_revision")
                    ):
                        if STATE["job"]["status"] in ("running", "done"):
                            return self.respond(
                                202,
                                {
                                    "id": STATE["job"]["id"],
                                    "cached": STATE["job"]["status"] == "done",
                                    "fingerprint": fingerprint,
                                },
                            )
                    if STATE.get("worker_job_id") or (STATE["job"] and STATE["job"]["status"] == "running"):
                        return self.respond(409, {"error": "当前分析仍在运行"})
                    if request.get("workflow") == "llm_then_jev" and fingerprint in RESULT_CACHE:
                        STATE["job"] = copy.deepcopy(RESULT_CACHE[fingerprint])
                        STATE["job"]["contact_revision"] = request.get("contact_revision")
                        STATE["job"]["target"] = request.get("target")
                        STATE["job"]["result"]["contact_revision"] = request.get("contact_revision")
                        STATE["job"]["result"]["target"] = request.get("target")
                        STATE["phase"] = "分析完成"
                        RESULT_CACHE.move_to_end(fingerprint)
                        return self.respond(202, {"id": STATE["job"]["id"], "cached": True})
                    job_id = secrets.token_hex(12)
                    STATE["job"] = {
                        "id": job_id,
                        "status": "running",
                        "started": time.time(),
                        "fingerprint": fingerprint,
                        "target": request.get("target"),
                        "contact_revision": request.get("contact_revision"),
                    }
                    # Reserve before submitting, independently of the visible
                    # card. Switching contacts may clear job but not this slot.
                    STATE.update(worker_job_id=job_id, worker_busy=True)
                try:
                    POOL.submit(run_job, job_id, request)
                except Exception:
                    with LOCK:
                        if STATE.get("worker_job_id") == job_id:
                            STATE.update(worker_job_id=None, worker_busy=False)
                        if current_job_locked(job_id, request):
                            STATE["job"].update(status="error", error="分析任务未能启动，请重试")
                    raise
                return self.respond(202, {"id": job_id})
            if self.path == "/api/clear":
                with LOCK:
                    if STATE.get("worker_job_id") or (STATE["job"] and STATE["job"]["status"] == "running"):
                        return self.respond(409, {"error": "请等待当前分析完成后清空"})
                    STATE["job"] = None
                    RESULT_CACHE.clear()
                    if ENGINE:
                        ENGINE.decider._state_kv.clear()
                return self.respond(200, {"ok": True})
            if self.path == "/api/observation":
                code = request.get("code")
                codes = ["matched", "permission", "target_mismatch", "no_window", "capture_error", "busy"]
                if code not in codes:
                    raise ValueError("Invalid capture status")
                count = request.get("line_count", 0)
                if not isinstance(count, int) or not 0 <= count <= 1000:
                    raise ValueError("Invalid line count")
                with LOCK:
                    record_check(
                        capture_status=code,
                        ocr_line_count=count,
                        capture_scope="visible-wechat-window",
                        chat_content_saved=False,
                    )
                return self.respond(200, {"ok": True})
            return self.respond(404, {"error": "Not found"})
        except (json.JSONDecodeError, UnicodeError, RecursionError):
            return self.respond(400, {"error": "请求JSON格式错误"})
        except (ValueError, TypeError) as exc:
            return self.respond(400, {"error": public_error(exc)})
        except (TimeoutError, socket.timeout):
            return self.respond(408, {"error": "请求传输超时"})
        except Exception as exc:
            return self.respond(500, {"error": public_error(exc)})


def main():
    os.umask(0o077)
    prepare_private_dirs()
    install_local_network_guard()
    server = ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("CHATLENS_PORT", "0"))), Handler)
    private = DATA_DIR
    private.mkdir(mode=0o700, exist_ok=True)
    session_file = private / "bridge-session.json"
    session_file.write_text(json.dumps({"port": server.server_port, "token": TOKEN}))
    session_file.chmod(0o600)
    latest = private / "latest-card.json"
    if private_debug_enabled() and latest.exists():
        try:
            previous = json.loads(latest.read_text())
            if (
                isinstance(previous, dict)
                and previous.get("status") == "done"
                and isinstance(previous.get("result"), dict)
                and previous["result"].get("workflow_version") == CARDS_VERSION
            ):
                STATE["job"] = previous
        except (ValueError, OSError):
            pass
    # Only the native parent reads this stdout pipe. Never write this token to disk.
    print(json.dumps({"port": server.server_port, "token": TOKEN}), flush=True)
    POOL.submit(initialize)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        POOL.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
