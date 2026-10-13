import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
import types
from unittest.mock import patch

from chatlens.history_store import HistoryStore
from chatlens import voice_history as voice


def media(path, rows):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE Name2Id(user_name TEXT)")
        db.execute("CREATE TABLE VoiceInfo(chat_name_id INTEGER, svr_id INTEGER, voice_data BLOB)")
        for talker, server, blob in rows:
            existing = db.execute("SELECT rowid FROM Name2Id WHERE user_name=?", (talker,)).fetchone()
            key = existing[0] if existing else db.execute("INSERT INTO Name2Id VALUES(?)", (talker,)).lastrowid
            db.execute("INSERT INTO VoiceInfo VALUES(?,?,?)", (key, server, blob))
    return path


class VoiceHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_blob_requires_exact_contact_and_full_server_id(self):
        server = 2**60 + 123
        path = media(self.root / "media.db", [("a", server, b"#!SILK_V3-a"), ("b", server, b"#!SILK_V3-b")])
        self.assertEqual(voice.voice_blob([path], "a", str(server)), b"#!SILK_V3-a")
        self.assertIsNone(voice.voice_blob([path], "a", str(server - 1)))
        second = media(self.root / "media-2.db", [("a", server, b"#!SILK_V3-conflict")])
        self.assertIsNone(voice.voice_blob([path, second], "a", str(server)))

    def test_inventory_isolated_and_cached_transcripts_skip_extraction(self):
        store = HistoryStore(self.root / "memory.sqlite3")
        row = {
            "local_id": 1,
            "server_id": "123",
            "kind": 34,
            "timestamp": 1,
            "sort_seq": 1,
            "role": "other",
            "text": "[语音]",
            "source": "message_0.db",
        }
        store.ingest("a", "甲", [row])
        store.ingest("b", "乙", [row])
        identity = store.recent("乙")["messages"][0]["id"]
        store.save_transcript(identity, "测试语音转写", "offline-asr", "a" * 64, 1, {})
        _, inventory = voice.voice_inventory(store, "甲")
        self.assertEqual(len(inventory), 1)
        self.assertFalse(inventory[0]["has_transcript"])
        metadata = {}
        with patch.object(voice, "PRIVATE", self.root / "missing-config"):
            records, found, missing = voice.extract(store, "乙", metadata)
        self.assertEqual((len(records), len(found), missing), (1, 0, 0))
        self.assertEqual(metadata["skipped_cached"], 1)

    def test_media_keys_are_validated_before_decrypt(self):
        source = self.root / "source"
        source.mkdir()
        audio = self.root / "audio"
        audio.mkdir()
        media(source / "media_0.db", [("a", 123, b"#!SILK_V3-a")])
        decrypt_calls = []

        def decrypt(original, plain, key):
            decrypt_calls.append(key)
            shutil.copyfile(original, plain)
            return True

        stats = {}
        plain = voice.read_media_databases(
            source, "official", [b"bad"], audio, lambda key, page: key == b"good", decrypt, lambda *args: None, stats
        )
        self.assertEqual(plain, [])
        self.assertEqual(decrypt_calls, [])
        self.assertEqual(stats["missing_keys"], 1)
        plain = voice.read_media_databases(
            source,
            "official",
            [b"bad", b"good"],
            audio,
            lambda key, page: key == b"good",
            decrypt,
            lambda *args: None,
            stats,
        )
        self.assertEqual(len(plain), 1)
        self.assertEqual(decrypt_calls, [b"good"])
        self.assertEqual(stats["validated_databases"], 1)

    def test_reports_and_official_source_do_not_overwrite_other_contact(self):
        hong, _ = voice.report_paths("示例好友甲")
        xu, _ = voice.report_paths("示例好友乙")
        self.assertNotEqual(hong, xu)
        self.assertRegex(xu.name, r"^contact-[0-9a-f]{12}-voice-history.json$")
        sources = voice.media_sources({"db_dir": str(self.root / "official-account/db_storage")})
        self.assertEqual(sources[0][0], "official-wechat-database")
        self.assertEqual(sources[0][1], self.root / "official-account/db_storage")

    def test_official_blobs_win_and_debug_only_fills_missing_messages(self):
        official = media(self.root / "official.db", [("a", 1, b"#!SILK_V3-official")])
        debug = media(self.root / "debug.db", [("a", 1, b"#!SILK_V3-stale"), ("a", 2, b"#!SILK_V3-fallback")])
        (self.root / "active-source.json").write_text(json.dumps({"db_dir": str(self.root / "configured")}))
        (self.root / "runtime-candidates.txt").write_text("00" * 32)
        records = [
            {"id": "a" * 64, "server_id": "1", "has_transcript": False},
            {"id": "b" * 64, "server_id": "2", "has_transcript": False},
        ]
        calls = []

        def read(source, label, keys, audio, verify, decrypt, merge, stats):
            calls.append(label)
            stats.update(missing_keys=0, voices_found=0)
            return [official if label == "official" else debug]

        metadata = {}
        adapter_map = types.ModuleType("wxexport.keys.map")
        adapter_map.page1_hmac_ok = lambda *args: True
        adapter_crypto = types.ModuleType("wxexport.decrypt")
        adapter_crypto.decrypt_database = lambda *args: True
        adapter_crypto.merge_wal = lambda *args: None
        with (
            patch.dict(sys.modules, {"wxexport.keys.map": adapter_map, "wxexport.decrypt": adapter_crypto}),
            patch.object(voice, "PRIVATE", self.root),
            patch.object(voice, "voice_inventory", return_value=({"talker": "a"}, records)),
            patch.object(voice, "media_sources", return_value=[("official", self.root), ("debug", self.root)]),
            patch.object(voice, "read_media_databases", side_effect=read),
        ):
            _, found, _ = voice.extract(None, "测试联系人", metadata)
        self.assertEqual(calls, ["official", "debug"])
        self.assertEqual([p.read_bytes() for _, p in found], [b"#!SILK_V3-official", b"#!SILK_V3-fallback"])
        self.assertEqual(metadata["media_sources"]["official"]["voices_found"], 1)
        self.assertEqual(metadata["media_sources"]["debug"]["voices_found"], 1)


if __name__ == "__main__":
    unittest.main()
