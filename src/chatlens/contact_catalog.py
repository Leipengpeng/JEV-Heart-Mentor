"""Local contact metadata and account-scoped, explicit analysis approval."""

import hashlib
import json
import os
import sqlite3
import threading
import time
import unicodedata
from functools import lru_cache

try:
    from pypinyin import lazy_pinyin, Style
except ImportError:
    lazy_pinyin = None
    Style = None
from pathlib import Path

from .history_connector import contact_snapshot
from .click_selection import normalized, similarity

SYSTEM_CONTACTS = {
    "weixin",
    "filehelper",
    "fmessage",
    "medianote",
    "floatbottle",
    "newsapp",
    "qqmail",
    "tmessage",
    "qmessage",
    "weibo",
    "notification_messages",
}


@lru_cache(maxsize=32768)
def contact_sort_name(display):
    """Cached Latin/pinyin collation key; no contact identifier is transformed."""
    value = unicodedata.normalize("NFKC", str(display or ""))
    if lazy_pinyin is not None:
        value = "".join(lazy_pinyin(value, style=Style.NORMAL, errors=lambda text: list(text)))
    # An optional dependency must never prevent local history access. An empty
    # key tells the UI to fall back to its zh-pinyin collator when Han characters
    # remain because the dependency was not installed.
    if any("\u3400" <= c <= "\u9fff" for c in value):
        return ""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
    return "".join(c for c in ascii_value if c.isalnum())


def eligible(talker):
    return bool(
        talker and "@chatroom" not in talker and not talker.startswith(("gh_", "@")) and talker not in SYSTEM_CONTACTS
    )


class ContactCatalog:
    def __init__(self, private, legacy_names=(), loader=contact_snapshot):
        self.private = Path(private)
        self.legacy_names = tuple(legacy_names)
        self.loader = loader
        self.lock = threading.RLock()
        self.records = {}
        self.approved = set()
        self.account_key = None
        self.checked_at = 0
        self.revision = ""
        self.source_signature = ""
        self.source = ""
        self.error = None

    def refresh(self, store=None, force=False):
        with self.lock:
            if self.records and not force and time.time() - self.checked_at < 15:
                return
            self.checked_at = time.time()
            try:
                snapshot = self.loader()
                self._install(snapshot, store)
                self.error = None
            except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
                self.error = str(exc)
                if not self.records and store is not None:
                    # Existing imports remain usable without requesting login or
                    # permissions. No new friend is approved in this fallback.
                    with store.connect() as db:
                        rows = db.execute("SELECT talker,display FROM contacts").fetchall()
                    self._install(
                        {
                            "contacts": [(r["talker"], r["display"], "", "") for r in rows],
                            "history_tables": {"Msg_" + hashlib.md5(r["talker"].encode()).hexdigest() for r in rows},
                            "account_key": "cached-imports",
                            "source": "cached-local-history",
                            "source_signature": "cached",
                        },
                        store,
                    )

    def _install(self, snapshot, store=None):
        records = {}
        imported = {}
        if store is not None:
            with store.connect() as db:
                imported = {
                    r["talker"]: r["count"]
                    for r in db.execute("SELECT talker,COUNT(*) AS count FROM messages GROUP BY talker")
                }
        for row in snapshot["contacts"]:
            talker, nick, remark, alias = row[:4]
            if not eligible(talker):
                continue
            display = remark or nick or alias or talker
            aliases = list(dict.fromkeys(str(x) for x in (display, nick, remark, alias) if x))
            available = "Msg_" + hashlib.md5(talker.encode()).hexdigest() in snapshot.get(
                "history_tables", set()
            ) or bool(imported.get(talker))
            records[talker] = {
                "id": talker,
                "name": display,
                "display_name": display,
                "sort_name": contact_sort_name(display),
                "aliases": aliases,
                "available": available,
                "messages": imported.get(talker, 0),
                "supported": True,
            }
        account = snapshot["account_key"]
        if self.account_key != account:
            self.account_key = account
            try:
                saved = json.loads((self.private / "analysis-approvals.json").read_text())
                approved = saved.get("accounts", {}).get(account, [])
            except (ValueError, OSError):
                approved = []
            self.approved = set(approved)
        self.records = records
        # Prior authorized contacts migrate only when the metadata identity is
        # unique. Never approve both people who happen to share a display name.
        migrated = False
        for name in self.legacy_names:
            matches = self.alias_matches(name)
            if len(matches) > 1 and store is not None:
                with store.connect() as db:
                    imported_matches = db.execute("SELECT talker FROM contacts WHERE display=?", (name,)).fetchall()
                known = {r["talker"] for r in imported_matches}
                matches = [r for r in matches if r["id"] in known] if len(known) == 1 else []
            if len(matches) == 1 and matches[0]["id"] not in self.approved:
                self.approved.add(matches[0]["id"])
                migrated = True
        self.source_signature = snapshot.get("source_signature", "")
        self.source = snapshot.get("source", "")
        self.revision = hashlib.sha256(
            json.dumps(
                [(k, v["name"], v["sort_name"], v["available"]) for k, v in sorted(records.items())], ensure_ascii=False
            ).encode()
        ).hexdigest()[:16]
        if migrated:
            self._save()

    def _save(self):
        self.private.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.private / "analysis-approvals.json"
        try:
            saved = json.loads(path.read_text())
        except (ValueError, OSError):
            saved = {}
        accounts = saved.get("accounts", {})
        accounts[self.account_key] = sorted(self.approved)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"accounts": accounts}, ensure_ascii=False))
        temporary.chmod(0o600)
        os.replace(temporary, path)

    def alias_matches(self, title):
        needle = normalized(title)
        return [r for r in self.records.values() if needle and any(normalized(a) == needle for a in r["aliases"])]

    def resolve(self, value, visible_texts=None, store=None):
        with self.lock:
            if value in self.records:
                return self.records[value], "canonical-contact-id"
            matches = self.alias_matches(value)
            if len(matches) == 1:
                return matches[0], "unique-contact-title"
            if len(matches) > 1:
                # Disambiguate only from already imported history; no automatic
                # history import or model analysis before consent.
                hits = []
                if store and visible_texts:
                    for contact in matches:
                        try:
                            rows = store.recent(contact["id"], 100)["messages"]
                        except ValueError:
                            continue
                        ordered = []
                        seen = set()
                        for text in visible_texts:
                            key = normalized(text)
                            if len(key) >= 2 and key not in seen:
                                ordered.append(text)
                                seen.add(key)
                        score = 0
                        position = 0
                        for text in ordered:
                            found = next(
                                (i for i in range(position, len(rows)) if similarity(text, rows[i]["text"])), None
                            )
                            if found is not None:
                                score += 1
                                position = found + 1
                        if score >= 2:
                            hits.append((score, contact))
                    hits.sort(key=lambda item: item[0], reverse=True)
                    if len(hits) == 1 or (len(hits) > 1 and hits[0][0] > hits[1][0]):
                        return hits[0][1], "visible-sequence-contact-match"
                return None, "ambiguous-contact-title"
            return None, "unknown-contact-title"

    def enabled(self, target):
        with self.lock:
            return target in self.approved and target in self.records

    def enable(self, target):
        with self.lock:
            if target not in self.records:
                raise ValueError("未找到这个联系人的本地记录")
            self.approved.add(target)
            self._save()

    def list(self):
        with self.lock:
            return [
                {**r, "analysis_enabled": r["id"] in self.approved, "target": r["id"]}
                for r in sorted(self.records.values(), key=lambda r: (r["sort_name"] or r["name"], r["name"], r["id"]))
            ]
