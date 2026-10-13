"""Synthetic databases verify contact switching without redoing crypto work."""

import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from chatlens import history_connector as connector
from chatlens.history_store import HistoryStore


class ContactSnapshotCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.private = self.root / "private"
        self.private.mkdir()
        self.source = self.root / "synthetic-self" / "db_storage"
        (self.source / "contact").mkdir(parents=True)
        (self.source / "message").mkdir()
        with sqlite3.connect(self.source / "contact/contact.db") as db:
            db.execute("CREATE TABLE contact(username,nick_name,remark,alias)")
            db.executemany(
                "INSERT INTO contact VALUES(?,?,?,?)",
                [("synthetic-a", "好友甲", "", ""), ("synthetic-b", "好友乙", "", "")],
            )
        with sqlite3.connect(self.source / "message/message_0.db") as db:
            db.execute("CREATE TABLE Name2Id(user_name)")
            db.executemany("INSERT INTO Name2Id VALUES(?)", [("synthetic-self",), ("synthetic-a",), ("synthetic-b",)])
            for i, talker in enumerate(("synthetic-a", "synthetic-b"), 2):
                table = "Msg_" + hashlib.md5(talker.encode()).hexdigest()
                db.execute(
                    f"CREATE TABLE {table}(local_id,server_id,local_type,sort_seq,create_time,real_sender_id,message_content,WCDB_CT_message_content)"
                )
                db.execute(f"INSERT INTO {table} VALUES(1,?,1,1,100,?,?,0)", (i, i, "合成测试消息"))
        (self.private / "runtime-candidates.txt").write_text("00" * 32 + "\n")
        (self.private / "active-source.json").write_text(json.dumps({"db_dir": str(self.source)}))
        self.store = HistoryStore(self.private / "memory.sqlite3")
        self.decrypt_count = 0

        def decrypt(source, destination, key):
            self.decrypt_count += 1
            shutil.copyfile(source, destination)
            return True

        fake_map = types.ModuleType("wxexport.keys.map")
        fake_map.page1_hmac_ok = lambda key, page: True
        fake_crypto = types.ModuleType("wxexport.decrypt")
        fake_crypto.decrypt_database = decrypt
        fake_crypto.merge_wal = lambda *args: None
        fake_parse = types.ModuleType("wxexport.msgparse")
        fake_parse.parse_message = lambda kind, content, compression, flag: {"content": content}
        self.patches = [
            patch.object(connector, "PRIVATE", self.private),
            patch.dict(
                sys.modules,
                {"wxexport.keys.map": fake_map, "wxexport.decrypt": fake_crypto, "wxexport.msgparse": fake_parse},
            ),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temporary.cleanup()

    def test_switch_contact_reuses_validated_snapshot_and_preserves_cache_for_both(self):
        first = connector.sync_history(self.store, "好友甲")
        second = connector.sync_history(self.store, "好友乙")
        again = connector.sync_history(self.store, "好友甲")
        self.assertEqual(first["decrypted_databases"], 2)
        self.assertEqual(second["messages"], 1)
        self.assertTrue(second["snapshot_reused"])
        self.assertEqual(second["decrypted_databases"], 0)
        self.assertTrue(again["unchanged"])
        self.assertEqual(self.decrypt_count, 2)
        cache = json.loads((self.private / "history-source-state.json").read_text())
        self.assertEqual(set(cache["synced_targets"]), {"好友甲", "好友乙"})

    def test_changed_snapshot_cannot_be_trusted_from_old_cache(self):
        connector.sync_history(self.store, "好友甲")
        snapshot = self.private / "decrypted/message/message_0.db"
        snapshot.write_bytes(b"corrupted snapshot")
        recovered = connector.sync_history(self.store, "好友乙")
        self.assertFalse(recovered["snapshot_reused"])
        self.assertEqual(self.decrypt_count, 4)
        self.assertEqual(recovered["messages"], 1)

    def test_legacy_cache_validated_once_and_migrated(self):
        connector.sync_history(self.store, "好友甲")
        cachefile = self.private / "history-source-state.json"
        cache = json.loads(cachefile.read_text())
        cache = {key: cache[key] for key in ("source", "fingerprint")}
        cache["target"] = "好友甲"
        cachefile.write_text(json.dumps(cache))
        result = connector.sync_history(self.store, "好友乙")
        self.assertTrue(result["snapshot_reused"])
        self.assertEqual(self.decrypt_count, 2)
        updated = json.loads(cachefile.read_text())
        self.assertIn("snapshot_fingerprint", updated)
        self.assertEqual(set(updated["synced_targets"]), {"好友甲", "好友乙"})

    def test_changed_source_invalidates_each_contact_sync_generation(self):
        connector.sync_history(self.store, "好友甲")
        connector.sync_history(self.store, "好友乙")
        contact = self.source / "contact/contact.db"
        with sqlite3.connect(contact) as db:
            db.execute("CREATE TABLE source_changed(value)")
        result = connector.sync_history(self.store, "好友乙")
        self.assertFalse(result["snapshot_reused"])
        self.assertEqual(self.decrypt_count, 4)
        cache = json.loads((self.private / "history-source-state.json").read_text())
        self.assertEqual(cache["synced_targets"], ["好友乙"])
        first = connector.sync_history(self.store, "好友甲")
        self.assertTrue(first["snapshot_reused"])
        self.assertNotIn("unchanged", first)

    def test_legacy_corruption_is_not_accepted_as_a_valid_snapshot(self):
        connector.sync_history(self.store, "好友甲")
        cachefile = self.private / "history-source-state.json"
        cache = json.loads(cachefile.read_text())
        cache.pop("snapshot_fingerprint")
        cachefile.write_text(json.dumps(cache))
        (self.private / "decrypted/contact/contact.db").write_bytes(b"broken")
        result = connector.sync_history(self.store, "好友乙")
        self.assertFalse(result["snapshot_reused"])
        self.assertEqual(self.decrypt_count, 4)

    def test_duplicate_names_resolve_only_when_one_has_local_history(self):
        with sqlite3.connect(self.source / "contact/contact.db") as db:
            db.execute("INSERT INTO contact VALUES(?,?,?,?)", ("synthetic-no-history", "好友甲", "", ""))
        result = connector.sync_history(self.store, "好友甲")
        self.assertEqual(result["exact_contact_matches"], 2)
        self.assertEqual(result["history_contact_matches"], 1)
        self.assertEqual(result["contact_match_strategy"], "unique-exact-name-with-local-history")
        with self.store.connect() as db:
            contact = self.store.contact(db, "好友甲")
            self.assertEqual(contact["talker"], "synthetic-a")
            talkers = {row[0] for row in db.execute("SELECT talker FROM messages")}
            self.assertEqual(talkers, {"synthetic-a"})

    def test_duplicate_names_with_two_histories_fail_without_importing_either(self):
        with sqlite3.connect(self.source / "contact/contact.db") as db:
            db.execute("UPDATE contact SET nick_name=? WHERE username=?", ("好友甲", "synthetic-b"))
        with self.assertRaisesRegex(ValueError, "精确匹配数量为 2，其中有历史的为 2"):
            connector.sync_history(self.store, "好友甲")
        self.assertEqual(self.store.stats()["messages"], 0)

    def test_messages_stay_bound_to_each_contact_when_cache_is_reused(self):
        connector.sync_history(self.store, "好友甲")
        connector.sync_history(self.store, "好友乙")
        first = self.store.recent("好友甲")["messages"][0]
        second = self.store.recent("好友乙")["messages"][0]
        self.assertEqual(first["talker"], "synthetic-a")
        self.assertEqual(second["talker"], "synthetic-b")
        self.assertNotEqual(first["id"], second["id"])
        with self.assertRaisesRegex(ValueError, "所选消息不属于当前联系人"):
            self.store.context("好友甲", through_id=second["id"])


if __name__ == "__main__":
    unittest.main()
