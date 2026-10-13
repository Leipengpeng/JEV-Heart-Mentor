"""Import the documented JSON interchange format; never reads WeChat itself."""

import json
from pathlib import Path

from .config import DATA_DIR, prepare_private_dirs
from .history_store import HistoryStore


def load_export(path):
    path = Path(path)
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("JSON 文件超过 20 MB，请分批导入")
    data = json.loads(path.read_text(encoding="utf-8"))
    contacts = data.get("contacts") if isinstance(data, dict) else None
    if not isinstance(contacts, list) or not contacts or len(contacts) > 1000:
        raise ValueError("需要包含 1–1000 个 contacts 的 JSON 对象")
    prepared = []
    seen = set()
    for contact in contacts:
        if not isinstance(contact, dict):
            raise ValueError("联系人必须是对象")
        identity, name, rows = contact.get("id"), contact.get("name"), contact.get("messages")
        if any(not isinstance(x, str) or not x.strip() or len(x) > 200 for x in (identity, name)):
            raise ValueError("联系人 id 和 name 必须是非空字符串")
        if identity in seen:
            raise ValueError("同一文件的联系人 id 不能重复")
        seen.add(identity)
        if not isinstance(rows, list) or len(rows) > 100000:
            raise ValueError("messages 必须是数组，单人最多 100000 条")
        messages = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("role") not in ("self", "other", "unknown"):
                raise ValueError("消息 role 需要是 self、other 或 unknown")
            text, timestamp = row.get("text"), row.get("timestamp")
            if not isinstance(text, str) or len(text) > 100000 or type(timestamp) is not int or timestamp < 0:
                raise ValueError("消息需要 text 字符串和非负整数 timestamp")
            local_id, sort_seq, kind = (
                row.get("local_id", index + 1),
                row.get("sort_seq", index + 1),
                row.get("kind", 1),
            )
            if any(type(x) is not int or x < 0 for x in (local_id, sort_seq, kind)):
                raise ValueError("local_id、sort_seq、kind 需要是非负整数")
            server_id = row.get("server_id", "0")
            if not isinstance(server_id, (str, int)) or isinstance(server_id, bool):
                raise ValueError("server_id 需要是字符串或整数")
            # A stable author-controlled source lets multiple exports coexist.
            source = row.get("source", "json-import")
            if not isinstance(source, str) or not source or len(source) > 200:
                raise ValueError("source 需要是非空短字符串")
            messages.append(
                dict(
                    local_id=local_id,
                    server_id=str(server_id),
                    source=source,
                    timestamp=timestamp,
                    sort_seq=sort_seq,
                    kind=kind,
                    role=row["role"],
                    text=text,
                )
            )
        prepared.append((identity, name, messages))
    return prepared


def import_json(path, store=None):
    # Validate the entire file before creating or changing a database.
    contacts = load_export(path)
    if store is None:
        prepare_private_dirs()
        store = HistoryStore(DATA_DIR / "memory.sqlite3")
    count = 0
    for identity, name, rows in contacts:
        count += store.ingest(identity, name, rows)["added"]
    return {"contacts": len(contacts), "messages_added": count, "analysis_enabled": False}
