"""Use audited local crypto hooks; decrypt stable copies and import one exact contact."""

import contextlib

try:
    import fcntl
except ImportError:  # The optional native database adapter is macOS-only.
    fcntl = None
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from .config import DATA_DIR, WXEXPORT_PATH

PRIVATE = DATA_DIR
VENDOR = WXEXPORT_PATH
LOCK = threading.Lock()


def file_fingerprint(paths, root):
    return {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths if p.is_file()}


def reusable_snapshot(cache, fingerprint, source_label, source, decrypted, files):
    """Validate a legacy snapshot once; stat-identical validated copies stay reusable."""
    identity = hashlib.sha256(str(source).encode()).hexdigest()
    if (
        cache.get("fingerprint") != fingerprint
        or cache.get("source") != source_label
        or cache.get("source_identity", identity) != identity
    ):
        return None
    copies = [decrypted / p.relative_to(source) for p in files]
    if not all(p.is_file() for p in copies):
        return None
    current = file_fingerprint(copies, decrypted)
    if "snapshot_fingerprint" in cache and cache["snapshot_fingerprint"] != current:
        return None
    if "snapshot_fingerprint" not in cache:
        try:
            for p in copies:
                with sqlite3.connect(p.as_uri() + "?mode=ro", uri=True) as db:
                    if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        return None
        except sqlite3.Error:
            return None
    return current


def synced_targets(cache):
    targets = cache.get("synced_targets", [])
    if not isinstance(targets, list):
        targets = []
    return set(s for s in targets + [cache.get("target")] if isinstance(s, str) and s)


def save_cache(path, fingerprint, source_label, source, snapshot_fingerprint, targets):
    state = {
        "fingerprint": fingerprint,
        "source": source_label,
        "source_identity": hashlib.sha256(str(source).encode()).hexdigest(),
        "snapshot_fingerprint": snapshot_fingerprint,
        "synced_targets": sorted(targets),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False))
    temporary.chmod(0o600)
    os.replace(temporary, path)


def resolve_contact(contacts, target, msgdbs):
    """A same-name contact without any local history cannot supply this conversation."""

    def norm(s):
        return "".join(str(s or "").split())

    exact_ids = [r for r in contacts if r[0] == target]
    if len(exact_ids) == 1:
        return exact_ids[0], {"exact_contact_matches": 1, "contact_match_strategy": "canonical-contact-id"}
    candidates = [r for r in contacts if any(norm(x) == norm(target) for x in r)]
    count = len(candidates)
    if count == 1:
        return candidates[0], {"exact_contact_matches": 1, "contact_match_strategy": "unique-exact-name"}
    usable = []
    for candidate in candidates:
        table = "Msg_" + hashlib.md5(candidate[0].encode()).hexdigest()
        for path in msgdbs:
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
                if (
                    db.execute("SELECT 1 FROM sqlite_master WHERE type=? AND name=?", ("table", table)).fetchone()
                    and db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
                ):
                    usable.append(candidate)
                    break
    if len(usable) == 1:
        return usable[0], {
            "exact_contact_matches": count,
            "history_contact_matches": 1,
            "contact_match_strategy": "unique-exact-name-with-local-history",
        }
    raise ValueError(f"联系人精确匹配数量为 {count}，其中有历史的为 {len(usable)}，需要唯一联系人名称或备注")


def runtime_status():
    path = PRIVATE / "runtime-events.jsonl"
    events = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    keys = PRIVATE / "runtime-candidates.txt"
    return {
        "hook_ready": any(e.get("event") == "armed" and e.get("hooks", 0) >= 3 for e in events),
        "candidate_count": len(set(keys.read_text().splitlines())) if keys.exists() else 0,
    }


def stable_copy(source, destination):
    paths = [source, Path(str(source) + "-wal")]

    def signature():
        return [(p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in paths]

    for _ in range(3):
        before = signature()
        for p in paths:
            out = destination if p == source else Path(str(destination) + "-wal")
            if p.exists():
                shutil.copyfile(p, out)
            else:
                out.unlink(missing_ok=True)
        if before == signature():
            return
    raise ValueError("微信正在写入数据库，请稍后再读取")


def select_source(configured, debug, keys, verify):
    """Prefer the already logged-in official data, validating before copying."""
    for label, source in (("official-wechat-database", configured), ("isolated-wechat-database", debug)):
        files = [source / "contact/contact.db", *sorted((source / "message").glob("message_[0-9]*.db"))]
        if not source.is_dir() or len(files) < 2 or not all(p.is_file() for p in files):
            continue
        valid = True
        for path in files:
            with path.open("rb") as stream:
                page = stream.read(4096)
            if not any(verify(key, page) for key in keys):
                valid = False
                break
        if valid:
            return source, label
    raise ValueError("现有密钥无法校验消息库；保留已导入历史，不启动登录或权限操作")


def sync_history(store, target):
    # JSON imports are a complete offline source. Consent must not require an
    # unrelated native key/adapter when no native source was configured.
    if not (PRIVATE / "active-source.json").exists():
        try:
            stats = store.stats(target)
        except ValueError:
            stats = {}
        if stats.get("messages", 0):
            return {"added": 0, **stats, "unchanged": True, "source": "imported-local-history", "seconds": 0}
        raise ValueError("尚未导入此联系人的历史，请先使用 chatlens import-json 或配置本地数据适配器")
    if not LOCK.acquire(blocking=False):
        raise ValueError("历史同步正在进行")
    process_lock = None
    try:
        PRIVATE.mkdir(exist_ok=True, mode=0o700)
        process_lock = (PRIVATE / "history-sync.lock").open("a")
        (PRIVATE / "history-sync.lock").chmod(0o600)
        if fcntl is None:
            raise ValueError("微信原生数据库适配器目前仅支持 macOS；可使用 JSON 导入")
        try:
            fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("历史同步正在进行") from None
        started = time.perf_counter()
        keyfile = PRIVATE / "runtime-candidates.txt"
        keys = (
            list(dict.fromkeys(bytes.fromhex(s) for s in keyfile.read_text().splitlines() if len(s) == 64))
            if keyfile.exists()
            else []
        )
        if not keys:
            raise ValueError("未配置本地数据适配器：请参阅 docs/setup.md，或先导入 JSON 历史")
        config = json.loads((PRIVATE / "active-source.json").read_text())
        account = Path(config["db_dir"]).parent.name
        debug = (
            Path.home()
            / "Library/Containers/io.github.timfang4162.wechat4-macos-frida-export.sip/Data/Documents/xwechat_files"
        )
        if VENDOR is not None:
            sys.path.insert(0, str(VENDOR / "src"))
        from wxexport.keys.map import page1_hmac_ok
        from wxexport.decrypt import decrypt_database, merge_wal
        from wxexport.msgparse import parse_message

        source, source_label = select_source(
            Path(config["db_dir"]), debug / account / "db_storage", keys, page1_hmac_ok
        )
        decrypted = PRIVATE / "decrypted"
        decrypted.mkdir(exist_ok=True)
        files = [source / "contact/contact.db", *sorted((source / "message").glob("message_[0-9]*.db"))]
        fingerprint = file_fingerprint([p for f in files for p in (f, Path(str(f) + "-wal"))], source)
        cachefile = PRIVATE / "history-source-state.json"
        try:
            cache = json.loads(cachefile.read_text()) if cachefile.exists() else {}
        except (ValueError, OSError):
            cache = {}
        if not isinstance(cache, dict):
            cache = {}
        snapshot = reusable_snapshot(cache, fingerprint, source_label, source, decrypted, files)
        targets = synced_targets(cache) if snapshot is not None else set()
        if snapshot is not None and target in targets:
            try:
                stats = store.stats(target)
                save_cache(cachefile, fingerprint, source_label, source, snapshot, targets)
                return {
                    "added": 0,
                    **stats,
                    "unchanged": True,
                    "source": source_label,
                    "snapshot_reused": True,
                    "decrypted_databases": 0,
                    "seconds": round(time.perf_counter() - started, 3),
                }
            except ValueError:
                pass
        matches = 0
        missing = 0
        if snapshot is None:
            with tempfile.TemporaryDirectory(prefix="history-", dir=PRIVATE) as directory:
                stage = Path(directory)
                for original in files:
                    if not original.exists():
                        continue
                    relative = original.relative_to(source)
                    encrypted = stage / "encrypted" / relative
                    encrypted.parent.mkdir(parents=True, exist_ok=True)
                    stable_copy(original, encrypted)
                    with encrypted.open("rb") as f:
                        page = f.read(4096)
                    key = next((k for k in keys if page1_hmac_ok(k, page)), None)
                    if key is None:
                        missing += 1
                        continue
                    output = stage / "plain" / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with contextlib.redirect_stdout(io.StringIO()):
                        if not decrypt_database(str(encrypted), str(output), key):
                            raise ValueError("数据库密钥校验失败")
                        merge_wal(str(encrypted), str(output), key)
                    with sqlite3.connect(output.as_uri() + "?mode=ro", uri=True) as db:
                        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                            raise ValueError("历史快照完整性校验失败，请重试")
                    matches += 1
                if missing:
                    raise ValueError(f"还有 {missing} 个联系人或消息库未取得密钥，请在独立副本打开目标聊天后重试")
                for original in files:
                    relative = original.relative_to(source)
                    output = stage / "plain" / relative
                    dst = decrypted / relative
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(output, dst)
                    dst.chmod(0o600)
        contactdb = decrypted / "contact/contact.db"
        if not contactdb.exists():
            raise ValueError("联系人数据库尚未解密")
        snapshot_metadata = file_fingerprint([decrypted / p.relative_to(source) for p in files], decrypted)
        save_cache(cachefile, fingerprint, source_label, source, snapshot_metadata, targets)
        with sqlite3.connect(contactdb.as_uri() + "?mode=ro", uri=True) as db:
            contacts = db.execute("SELECT username,nick_name,remark,alias FROM contact").fetchall()
        msgdbs = [decrypted / p.relative_to(source) for p in files if p.parent.name == "message"]
        contact, match_metadata = resolve_contact(contacts, target, msgdbs)
        talker = contact[0]
        table = "Msg_" + hashlib.md5(talker.encode()).hexdigest()
        identities = set()
        for path in msgdbs:
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
                identities.update(r[0] for r in db.execute("SELECT user_name FROM Name2Id"))
        own = [s for s in identities if s and "@" not in s and (account == s or account.startswith(s + "_"))]
        self_id = own[0] if len(own) == 1 else None
        messages = []
        for path in msgdbs:
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
                names = dict(db.execute("SELECT rowid,user_name FROM Name2Id"))
                if not db.execute("SELECT 1 FROM sqlite_master WHERE type=? AND name=?", ("table", table)).fetchone():
                    continue
                rows = db.execute(
                    f"SELECT local_id,server_id,local_type,sort_seq,create_time,real_sender_id,"
                    f"message_content,WCDB_CT_message_content FROM {table}"
                )
                for local, server, kind, seq, stamp, sender, content, compression in rows:
                    who = names.get(sender)
                    role = "self" if self_id and who == self_id else ("other" if who == talker else "unknown")
                    parsed = parse_message(kind, content, compression, False)
                    messages.append(
                        {
                            "local_id": local,
                            "server_id": str(server or 0),
                            "kind": kind,
                            "sort_seq": seq or 0,
                            "timestamp": stamp or 0,
                            "role": role,
                            "text": parsed["content"],
                            "source": path.name,
                        }
                    )
        if not messages:
            raise ValueError("本机没有目标会话的历史消息，可能需要先从手机迁移")
        display = contact[2] or contact[1] or contact[3] or target
        result = store.ingest(talker, display, messages, self_verified=self_id is not None)
        result.update(
            decrypted_databases=matches,
            seconds=round(time.perf_counter() - started, 3),
            snapshot_reused=snapshot is not None,
            source=source_label,
            scope="local-history-and-current-database-updates",
            **match_metadata,
        )
        targets.add(target)
        snapshot = file_fingerprint([decrypted / p.relative_to(source) for p in files], decrypted)
        save_cache(cachefile, fingerprint, source_label, source, snapshot, targets)
        return result
    finally:
        if process_lock is not None:
            process_lock.close()
        LOCK.release()


def contact_snapshot():
    """Refresh contact metadata only; never import histories or invoke a model."""
    if not LOCK.acquire(blocking=False):
        raise ValueError("历史同步正在进行")
    process_lock = None
    try:
        PRIVATE.mkdir(exist_ok=True, mode=0o700)
        process_lock = (PRIVATE / "history-sync.lock").open("a")
        (PRIVATE / "history-sync.lock").chmod(0o600)
        if fcntl is None:
            raise ValueError("微信原生数据库适配器目前仅支持 macOS；可使用 JSON 导入")
        try:
            fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("历史同步正在进行") from None
        config = json.loads((PRIVATE / "active-source.json").read_text())
        configured = Path(config["db_dir"])
        account = configured.parent.name
        keyfile = PRIVATE / "runtime-candidates.txt"
        keys = list(dict.fromkeys(bytes.fromhex(s) for s in keyfile.read_text().splitlines() if len(s) == 64))
        if VENDOR is not None:
            sys.path.insert(0, str(VENDOR / "src"))
        from wxexport.keys.map import page1_hmac_ok
        from wxexport.decrypt import decrypt_database, merge_wal

        debug = (
            Path.home()
            / "Library/Containers/io.github.timfang4162.wechat4-macos-frida-export.sip/Data/Documents/xwechat_files"
        )
        source, label = select_source(configured, debug / account / "db_storage", keys, page1_hmac_ok)
        original = source / "contact/contact.db"
        fingerprint = file_fingerprint([original, Path(str(original) + "-wal")], source)
        account_key = hashlib.sha256(str(configured).encode()).hexdigest()
        cachefile = PRIVATE / "contact-source-state.json"
        try:
            cache = json.loads(cachefile.read_text())
        except (ValueError, OSError):
            cache = {}
        output = PRIVATE / "decrypted/contact/contact.db"
        output_fingerprint = [output.stat().st_size, output.stat().st_mtime_ns] if output.exists() else None
        # History sync may replace the same contact snapshot atomically. A
        # changed output mtime alone must not make the two snapshot caches
        # repeatedly invalidate each other and redo all crypto every 15 seconds.
        if (
            cache.get("fingerprint") == fingerprint
            and cache.get("account_key") == account_key
            and output.exists()
            and cache.get("snapshot_fingerprint") != output_fingerprint
        ):
            try:
                with sqlite3.connect(output.as_uri() + "?mode=ro", uri=True) as db:
                    valid = db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
                    if valid:
                        db.execute("SELECT username FROM contact LIMIT 1").fetchone()
            except sqlite3.Error:
                valid = False
            if valid:
                cache["snapshot_fingerprint"] = output_fingerprint
                temporary = cachefile.with_suffix(".tmp")
                temporary.write_text(json.dumps(cache))
                temporary.chmod(0o600)
                os.replace(temporary, cachefile)
        if (
            cache.get("fingerprint") != fingerprint
            or cache.get("account_key") != account_key
            or not output.exists()
            or cache.get("snapshot_fingerprint") != output_fingerprint
        ):
            with tempfile.TemporaryDirectory(prefix="contacts-", dir=PRIVATE) as directory:
                encrypted = Path(directory) / "contact.db"
                plain = Path(directory) / "plain.db"
                stable_copy(original, encrypted)
                with encrypted.open("rb") as stream:
                    page = stream.read(4096)
                key = next((k for k in keys if page1_hmac_ok(k, page)), None)
                if key is None:
                    raise ValueError("联系人数据库尚未取得有效密钥")
                with contextlib.redirect_stdout(io.StringIO()):
                    if not decrypt_database(str(encrypted), str(plain), key):
                        raise ValueError("联系人数据库校验失败")
                    merge_wal(str(encrypted), str(plain), key)
                with sqlite3.connect(plain.as_uri() + "?mode=ro", uri=True) as db:
                    if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise ValueError("联系人快照校验失败")
                output.parent.mkdir(parents=True, exist_ok=True)
                os.replace(plain, output)
                output.chmod(0o600)
            temporary = cachefile.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "account_key": account_key,
                        "fingerprint": fingerprint,
                        "source": label,
                        "snapshot_fingerprint": [output.stat().st_size, output.stat().st_mtime_ns],
                    }
                )
            )
            temporary.chmod(0o600)
            os.replace(temporary, cachefile)
        with sqlite3.connect(output.as_uri() + "?mode=ro", uri=True) as db:
            contacts = db.execute("SELECT username,nick_name,remark,alias FROM contact").fetchall()
        tables = set()
        for path in sorted((PRIVATE / "decrypted/message").glob("message_[0-9]*.db")):
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
                tables.update(
                    r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Msg_%'")
                )
        return {
            "contacts": contacts,
            "history_tables": tables,
            "account_key": account_key,
            "source": label,
            "source_signature": hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest(),
        }
    finally:
        if process_lock is not None:
            process_lock.close()
        LOCK.release()


def history_source_signature():
    """Private source generation hash used only to invalidate local result cache."""
    try:
        state = json.loads((PRIVATE / "history-source-state.json").read_text())
        value = [state.get("source_identity"), state.get("source"), state.get("fingerprint")]
    except (ValueError, OSError):
        value = []
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
