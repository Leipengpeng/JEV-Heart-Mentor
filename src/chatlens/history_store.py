"""Private, local message memory with stable IDs and per-contact retrieval."""

import hashlib
import json
import re
import sqlite3
import time
import unicodedata

DEFAULT_ANALYSIS_HISTORY_LIMIT = 300
MAX_SOURCE_HISTORY_LIMIT = 1000


def validate_source_history_limit(value):
    if type(value) is not int or not 1 <= value <= MAX_SOURCE_HISTORY_LIMIT:
        raise ValueError("历史分析需要 1–1000 条来源消息")
    return value


def normalize_display_text(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or "")))


from pathlib import Path


class HistoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS contacts (
                    talker TEXT PRIMARY KEY, display TEXT NOT NULL,
                    imported_at REAL NOT NULL, self_verified INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, talker TEXT NOT NULL, timestamp INTEGER NOT NULL,
                    sort_seq INTEGER NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL,
                    kind INTEGER NOT NULL, source TEXT NOT NULL, server_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_time ON messages(talker,timestamp,sort_seq,id);
                CREATE VIRTUAL TABLE IF NOT EXISTS message_search USING fts5(
                    id UNINDEXED, talker UNINDEXED, text, tokenize='trigram'
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    id TEXT PRIMARY KEY, talker TEXT NOT NULL, created_at REAL NOT NULL,
                    summary TEXT NOT NULL, evidence TEXT NOT NULL,
                    uncertainties TEXT NOT NULL, message_ids TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS history_generations (talker TEXT PRIMARY KEY, revision INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS voice_transcripts (
                    message_id TEXT PRIMARY KEY REFERENCES messages(id),
                    transcript TEXT NOT NULL, model TEXT NOT NULL, audio_sha256 TEXT NOT NULL,
                    duration_seconds REAL NOT NULL, metrics TEXT NOT NULL, created_at REAL NOT NULL
                );
            """)
            # Display names are mutable and can collide; talker is the identity.
            unique_display = any(
                r[2] and any(c[2] == "display" for c in db.execute("PRAGMA index_info(" + r[1] + ")"))
                for r in db.execute("PRAGMA index_list(contacts)")
            )
            if unique_display:
                db.executescript("""ALTER TABLE contacts RENAME TO contacts_old;
                    CREATE TABLE contacts (talker TEXT PRIMARY KEY, display TEXT NOT NULL,
                        imported_at REAL NOT NULL, self_verified INTEGER NOT NULL);
                    INSERT INTO contacts SELECT * FROM contacts_old;
                    DROP TABLE contacts_old;""")
            db.executescript("""CREATE TABLE IF NOT EXISTS message_display (
                id TEXT PRIMARY KEY, talker TEXT NOT NULL, normalized TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS message_display_exact ON message_display(talker,normalized);
                CREATE VIRTUAL TABLE IF NOT EXISTS display_search USING fts5(
                    id UNINDEXED, talker UNINDEXED, normalized, tokenize='trigram');""")
            # One migration pass; subsequent starts only process missing rows.
            for r in db.execute(
                "SELECT id,talker,text FROM messages WHERE id NOT IN (SELECT id FROM message_display)"
            ).fetchall():
                text = normalize_display_text(r["text"])
                db.execute("INSERT INTO message_display VALUES(?,?,?)", (r["id"], r["talker"], text))
                db.execute("INSERT INTO display_search VALUES(?,?,?)", (r["id"], r["talker"], text))
        self.path.chmod(0o600)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def ingest(self, talker, display, messages, self_verified=False):
        if not talker or not display or not isinstance(messages, list):
            raise ValueError("历史消息格式错误")
        added = 0
        changed = False
        with self.connect() as db:
            for item in messages:
                server = str(item.get("server_id") or "0")
                identity = "server:" + server if server != "0" else f"local:{item['source']}:{item['local_id']}"
                key = hashlib.sha256((talker + ":" + identity).encode()).hexdigest()
                role = item.get("role", "unknown")
                if role not in ("self", "other", "unknown"):
                    raise ValueError("历史消息发言方错误")
                text = str(item.get("text", ""))
                transcript = db.execute(
                    "SELECT transcript FROM voice_transcripts WHERE message_id=?", (key,)
                ).fetchone()
                if int(item.get("kind", 1)) == 34 and transcript:
                    text = "[语音自动转写，待核对] " + transcript["transcript"]
                existing = db.execute(
                    "SELECT timestamp,sort_seq,role,text,kind FROM messages WHERE id=?", (key,)
                ).fetchone()
                changed = (
                    changed
                    or existing is None
                    or tuple(existing)
                    != (int(item["timestamp"]), int(item.get("sort_seq", 0)), role, text, int(item.get("kind", 1)))
                )
                db.execute(
                    """INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET role=excluded.role,text=excluded.text,
                    kind=excluded.kind,timestamp=excluded.timestamp,sort_seq=excluded.sort_seq""",
                    (
                        key,
                        talker,
                        int(item["timestamp"]),
                        int(item.get("sort_seq", 0)),
                        role,
                        text,
                        int(item.get("kind", 1)),
                        item["source"],
                        server,
                    ),
                )
                if existing is None or existing["text"] != text:
                    if existing is not None:
                        db.execute("DELETE FROM message_search WHERE id=?", (key,))
                    db.execute("INSERT INTO message_search VALUES(?,?,?)", (key, talker, text))
                    norm = normalize_display_text(text)
                    db.execute("INSERT OR REPLACE INTO message_display VALUES(?,?,?)", (key, talker, norm))
                    db.execute("DELETE FROM display_search WHERE id=?", (key,))
                    db.execute("INSERT INTO display_search VALUES(?,?,?)", (key, talker, norm))
                added += existing is None
            if changed:
                db.execute(
                    "INSERT INTO history_generations VALUES(?,1) ON CONFLICT(talker) DO UPDATE SET revision=revision+1",
                    (talker,),
                )
            db.execute(
                """INSERT INTO contacts VALUES(?,?,?,?) ON CONFLICT(talker)
                DO UPDATE SET display=excluded.display,imported_at=excluded.imported_at,
                self_verified=excluded.self_verified""",
                (talker, display, time.time(), int(self_verified)),
            )
        return {"added": added, **self.stats(talker)}

    def contact(self, db, target):
        row = db.execute("SELECT * FROM contacts WHERE talker=?", (target,)).fetchone()
        if row is None:
            matches = db.execute("SELECT * FROM contacts WHERE display=?", (target,)).fetchall()
            if len(matches) > 1:
                raise ValueError("联系人名称重复，请使用唯一联系人标识")
            row = matches[0] if matches else None
        if row is None:
            raise ValueError("尚未导入这个联系人的历史，请先连接历史文件")
        return row

    def generation(self, target):
        with self.connect() as db:
            c = self.contact(db, target)
            row = db.execute("SELECT revision FROM history_generations WHERE talker=?", (c["talker"],)).fetchone()
            return row[0] if row else 0

    def stats(self, target=None):
        with self.connect() as db:
            if not target:
                return {
                    "contacts": db.execute("SELECT COUNT(*) FROM contacts").fetchone()[0],
                    "messages": db.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                }
            c = self.contact(db, target)
            r = db.execute(
                "SELECT COUNT(*),MIN(timestamp),MAX(timestamp) FROM messages WHERE talker=?", (c["talker"],)
            ).fetchone()
            return {
                "target": c["display"],
                "messages": r[0],
                "earliest": r[1],
                "latest": r[2],
                "imported_at": c["imported_at"],
                "self_verified": bool(c["self_verified"]),
            }

    def recent(self, target, limit=100, before=None):
        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("每页需要 1–100 条消息")
        with self.connect() as db:
            c = self.contact(db, target)
            args = [c["talker"]]
            extra = ""
            if before:
                if (
                    not isinstance(before, list)
                    or len(before) != 3
                    or any(not isinstance(x, int) for x in before[:2])
                    or not isinstance(before[2], str)
                    or not re.fullmatch("[0-9a-f]{64}", before[2])
                ):
                    raise ValueError("历史分页游标格式错误")
                extra = " AND (timestamp,sort_seq,id)<(?,?,?)"
                args += before
            rows = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM messages WHERE talker=?"
                    + extra
                    + " ORDER BY timestamp DESC,sort_seq DESC,id DESC LIMIT ?",
                    args + [limit + 1],
                )
            ]
            more = len(rows) > limit
            rows = rows[:limit]
            cursor = [rows[-1]["timestamp"], rows[-1]["sort_seq"], rows[-1]["id"]] if rows and more else None
            return {"messages": list(reversed(rows)), "before": cursor, "has_more": more}

    def search(self, target, query, limit=20, through=None):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200 or not 1 <= limit <= 100:
            raise ValueError("请输入 1–200 字的历史检索词")
        query = query.strip()
        with self.connect() as db:
            c = self.contact(db, target)
            boundary = " AND (m.timestamp,m.sort_seq,m.id)<=(?,?,?)" if through is not None else ""
            boundary_args = list(through) if through is not None else []
            if len(query) >= 3:
                phrase = '"' + query.replace('"', '""') + '"'
                rows = db.execute(
                    """SELECT m.* FROM message_search f JOIN messages m ON m.id=f.id
                    WHERE message_search MATCH ? AND m.talker=?"""
                    + boundary
                    + """
                    ORDER BY m.timestamp DESC,m.sort_seq DESC,m.id DESC LIMIT ?""",
                    [phrase, c["talker"], *boundary_args, limit],
                )
            else:
                escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                rows = db.execute(
                    """SELECT m.* FROM messages m WHERE m.talker=? AND m.text LIKE ? ESCAPE '\\'"""
                    + boundary
                    + """
                    ORDER BY m.timestamp DESC,m.sort_seq DESC,m.id DESC LIMIT ?""",
                    [c["talker"], "%" + escaped + "%", *boundary_args, limit],
                )
            return [dict(r) for r in rows]

    def context(self, target, query="", through_id=None, source_history_limit=100):
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("历史检索词格式错误")
        limit = validate_source_history_limit(source_history_limit)
        selected = None
        with self.connect() as db:
            c = self.contact(db, target)
            boundary = []
            where = ""
            if through_id is None:
                latest = db.execute(
                    "SELECT timestamp,sort_seq,id FROM messages WHERE talker=? AND role='other' "
                    "AND trim(text)<>'' AND ((kind & 4294967295) IN (1,49) OR "
                    "(kind=34 AND text LIKE '[语音自动转写，待核对] %')) "
                    "ORDER BY timestamp DESC,sort_seq DESC,id DESC LIMIT 1",
                    (c["talker"],),
                ).fetchone()
                if latest:
                    boundary = list(latest)
                    where = " AND (timestamp,sort_seq,id)<=(?,?,?)"
            if through_id is not None:
                if not isinstance(through_id, str) or not re.fullmatch("[0-9a-f]{64}", through_id):
                    raise ValueError("消息位置格式错误")
                row = db.execute("SELECT * FROM messages WHERE talker=? AND id=?", (c["talker"], through_id)).fetchone()
                if row is None:
                    raise ValueError("所选消息不属于当前联系人")
                selected = dict(row)
                boundary = [selected["timestamp"], selected["sort_seq"], selected["id"]]
                where = " AND (timestamp,sort_seq,id)<=(?,?,?)"
            # Filter media before LIMIT: a run of image/voice placeholders must
            # not displace the requested usable textual context messages.
            recent = list(
                reversed(
                    [
                        dict(r)
                        for r in db.execute(
                            "SELECT * FROM messages WHERE talker=?"
                            + where
                            + " AND trim(text)<>'' AND ((kind & 4294967295) IN (1,49) OR "
                            "(kind=34 AND text LIKE '[语音自动转写，待核对] %')) "
                            "ORDER BY timestamp DESC,sort_seq DESC,id DESC LIMIT ?",
                            [c["talker"], *boundary, limit],
                        )
                    ]
                )
            )
            recent = [m for m in recent if m["text"].strip()]
        if selected and selected["id"] not in {m["id"] for m in recent}:
            note = (
                "[所选语音尚未转写，不能判断其内容]"
                if selected["kind"] == 34
                else "[所选消息没有可用文字内容，仅依据此前对话判断]"
            )
            recent = (recent[-(limit - 1) :] if limit > 1 else []) + [{**selected, "text": note}]
        if not recent:
            raise ValueError("没有可供文字分析的历史消息")
        ids = {m["id"] for m in recent}
        watermark = (
            [selected["timestamp"], selected["sort_seq"], selected["id"]]
            if selected
            else [recent[-1]["timestamp"], recent[-1]["sort_seq"], recent[-1]["id"]]
        )
        evidence = (
            [m for m in self.search(target, query, 20, watermark) if m["id"] not in ids][:8] if query.strip() else []
        )
        with self.connect() as db:
            c = self.contact(db, target)
            summaries = [
                dict(r)
                for r in db.execute(
                    "SELECT summary,evidence,uncertainties,message_ids,created_at"
                    " FROM summaries WHERE talker=? ORDER BY created_at DESC LIMIT 100",
                    (c["talker"],),
                )
            ]
            safe_summaries = []
            for summary in summaries:
                source_ids = json.loads(summary["message_ids"])
                if not source_ids:
                    continue
                rows = db.execute(
                    "SELECT timestamp,sort_seq,id FROM messages WHERE talker=? AND id IN ("
                    + ",".join("?" for _ in source_ids)
                    + ")",
                    [c["talker"], *source_ids],
                ).fetchall()
                if len(rows) == len(set(source_ids)) and all(tuple(r) <= tuple(watermark) for r in rows):
                    safe_summaries.append(summary)
                if len(safe_summaries) == 3:
                    break
        return {
            "messages": [{"role": m["role"], "text": m["text"]} for m in recent],
            "history_ids": [m["id"] for m in recent],
            "context_message_limit": limit,
            "source_history_limit": limit,
            "context_messages": [
                {"index": i, "message_id": m["id"], "timestamp": m["timestamp"], "role": m["role"]}
                for i, m in enumerate(recent)
            ],
            "history_watermark": watermark,
            "history_evidence": [
                {"id": m["id"], "timestamp": m["timestamp"], "role": m["role"], "text": m["text"][:800]}
                for m in evidence
            ],
            "past_summaries": safe_summaries,
            "stats": self.stats(target),
            "selected_message": (
                {k: selected[k] for k in ("id", "timestamp", "sort_seq", "role", "text", "kind")} if selected else None
            ),
        }

    def save_transcript(self, message_id, transcript, model, audio_sha256, duration_seconds, metrics):
        """Attach uncertain ASR text to its original voice message and refresh search."""
        if not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 20000:
            raise ValueError("语音转写文本格式错误")
        if not re.fullmatch("[0-9a-f]{64}", audio_sha256) or duration_seconds <= 0:
            raise ValueError("语音来源格式错误")
        with self.connect() as db:
            row = db.execute("SELECT talker,kind FROM messages WHERE id=?", (message_id,)).fetchone()
            if row is None or row["kind"] != 34:
                raise ValueError("语音来源消息不存在")
            db.execute(
                "INSERT OR REPLACE INTO voice_transcripts VALUES(?,?,?,?,?,?,?)",
                (
                    message_id,
                    transcript.strip(),
                    model,
                    audio_sha256,
                    float(duration_seconds),
                    json.dumps(metrics, ensure_ascii=False),
                    time.time(),
                ),
            )
            text = "[语音自动转写，待核对] " + transcript.strip()
            db.execute("UPDATE messages SET text=? WHERE id=?", (text, message_id))
            db.execute(
                "INSERT INTO history_generations VALUES(?,1) ON CONFLICT(talker) DO UPDATE SET revision=revision+1",
                (row["talker"],),
            )
            db.execute("DELETE FROM message_search WHERE id=?", (message_id,))
            db.execute("INSERT INTO message_search VALUES(?,?,?)", (message_id, row["talker"], text))
            norm = normalize_display_text(text)
            db.execute("INSERT OR REPLACE INTO message_display VALUES(?,?,?)", (message_id, row["talker"], norm))
            db.execute("DELETE FROM display_search WHERE id=?", (message_id,))
            db.execute("INSERT INTO display_search VALUES(?,?,?)", (message_id, row["talker"], norm))

    def save_summary(self, target, ids, analysis):
        if not ids:
            return
        if not isinstance(ids, list) or len(ids) > MAX_SOURCE_HISTORY_LIMIT or any(not isinstance(i, str) for i in ids):
            raise ValueError("摘要来源格式错误")
        with self.connect() as db:
            c = self.contact(db, target)
            placeholders = ",".join("?" for _ in ids)
            count = db.execute(
                f"SELECT COUNT(*) FROM messages WHERE talker=? AND id IN ({placeholders})", [c["talker"], *ids]
            ).fetchone()[0]
            if count != len(set(ids)):
                raise ValueError("摘要来源不属于当前联系人")
            key = hashlib.sha256((c["talker"] + json.dumps(ids)).encode()).hexdigest()
            db.execute(
                "INSERT OR REPLACE INTO summaries VALUES(?,?,?,?,?,?,?)",
                (
                    key,
                    c["talker"],
                    time.time(),
                    str(analysis["summary"]),
                    str(analysis["evidence"]),
                    json.dumps(analysis.get("uncertainties", []), ensure_ascii=False),
                    json.dumps(ids),
                ),
            )
