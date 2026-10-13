import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatlens.history_store import HistoryStore
from chatlens.history_connector import stable_copy


def message(i, text=None, stamp=None, source="message_0.db"):
    return {
        "local_id": i,
        "server_id": str(1000 + i),
        "source": source,
        "timestamp": stamp or i,
        "sort_seq": i,
        "role": "other",
        "kind": 1,
        "text": text or f"测试记录 {i}",
    }


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = HistoryStore(Path(self.temp.name) / "memory.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_replica_deduplication_and_updated_search(self):
        self.store.ingest("a", "甲", [message(1, "项目交付安排")])
        self.store.ingest("a", "甲", [message(1, "项目延期说明", source="message_1.db")])
        self.assertEqual(self.store.stats("甲")["messages"], 1)
        self.assertEqual(self.store.search("甲", "交付安排"), [])
        self.assertEqual(len(self.store.search("甲", "延期说明")), 1)

    def test_same_timestamp_paging_has_no_gaps_or_duplicates(self):
        self.store.ingest("a", "甲", [message(i, stamp=100) for i in range(1, 15)])
        seen = []
        cursor = None
        while True:
            page = self.store.recent("甲", 3, cursor)
            seen += [m["id"] for m in page["messages"]]
            if not page["has_more"]:
                break
            cursor = page["before"]
        self.assertEqual(len(seen), 14)
        self.assertEqual(len(set(seen)), 14)

    def test_contact_isolation_and_literal_short_search(self):
        self.store.ingest("a", "甲", [message(1, "文字含有百分号%")])
        self.store.ingest("b", "乙", [message(1, "只属于另一个联系人")])
        self.assertEqual(self.store.search("甲", "联系人"), [])
        self.assertEqual(len(self.store.search("甲", "%")), 1)
        self.assertEqual(self.store.search("甲", "_"), [])

    def test_old_evidence_is_separate_from_recent_context(self):
        rows = [message(i) for i in range(1, 201)]
        rows[0]["text"] = "项目交付安排"
        self.store.ingest("a", "甲", rows, self_verified=True)
        value = self.store.context("甲", "交付安排")
        self.assertEqual(len(value["messages"]), 100)
        self.assertEqual(len(value["history_evidence"]), 1)
        self.assertNotIn(value["history_evidence"][0]["id"], value["history_ids"])

    def test_summary_provenance_cannot_cross_contacts(self):
        self.store.ingest("a", "甲", [message(1)])
        self.store.ingest("b", "乙", [message(2)])
        ids = self.store.context("乙")["history_ids"]
        analysis = {"summary": "摘要", "evidence": "依据"}
        with self.assertRaises(ValueError):
            self.store.save_summary("甲", ids, analysis)
        self.store.save_summary("乙", ids, analysis)
        self.assertEqual(len(self.store.context("乙")["past_summaries"]), 1)

    def test_invalid_cursor_and_query_rejected(self):
        self.store.ingest("a", "甲", [message(1)])
        with self.assertRaises(ValueError):
            self.store.recent("甲", before=[1, 2, "invalid"])
        with self.assertRaises(ValueError):
            self.store.context("甲", query=[])

    def test_voice_transcript_survives_sync_and_is_searchable_context(self):
        voice = message(1, "[语音]")
        voice["kind"] = 34
        self.store.ingest("a", "甲", [voice, message(2, "文字")])
        identity = self.store.recent("甲")["messages"][0]["id"]
        self.store.save_transcript(
            identity, "下周讨论项目方案", "offline-asr", "a" * 64, 3.0, {"human_verified": False}
        )
        self.store.ingest("a", "甲", [voice])
        self.assertEqual(len(self.store.search("甲", "项目方案")), 1)
        context = self.store.context("甲")
        self.assertEqual(len(context["messages"]), 2)
        self.assertIn("待核对", context["messages"][0]["text"])
        self.assertIn(identity, context["history_ids"])

    def test_transcript_cannot_be_attached_to_text_or_missing_message(self):
        self.store.ingest("a", "甲", [message(1)])
        identity = self.store.recent("甲")["messages"][0]["id"]
        for key in [identity, "b" * 64]:
            with self.assertRaises(ValueError):
                self.store.save_transcript(key, "文字", "asr", "a" * 64, 1, {})

    def test_snapshot_rejects_concurrent_writes(self):
        source = Path(self.temp.name) / "source.db"
        dest = Path(self.temp.name) / "snapshot.db"
        source.write_bytes(b"original")
        import shutil

        copy = shutil.copyfile

        def changing(a, b):
            copy(a, b)
            source.write_bytes(source.read_bytes() + b"x")

        with patch("chatlens.history_connector.shutil.copyfile", side_effect=changing):
            with self.assertRaises(ValueError):
                stable_copy(source, dest)

    def test_context_retains_100_usable_records_across_unsupported_media(self):
        rows = []
        for i in range(1, 101):
            rows.append(message(i * 2 - 1, "早期重要约定" if i == 1 else "可用文字" + str(i)))
            image = message(i * 2, "[图片]")
            image["kind"] = 3
            rows.append(image)
        self.store.ingest("a", "甲", rows)
        context = self.store.context("a")
        self.assertEqual(len(context["messages"]), 100)
        self.assertEqual(context["messages"][0]["text"], "早期重要约定")
        self.assertEqual(len(context["context_messages"]), 100)
        self.assertEqual(context["context_messages"][0]["index"], 0)
        self.assertEqual(context["context_messages"][-1]["timestamp"], 199)
        self.assertEqual(context["messages"][-1]["text"], "可用文字100")

    def test_default_context_ends_at_latest_other_not_later_self_reply(self):
        own = message(3, "我方后续回复")
        own["role"] = "self"
        self.store.ingest("a", "甲", [message(1, "前文"), message(2, "对方当前句"), own])
        context = self.store.context("a")
        self.assertEqual(context["messages"][-1]["text"], "对方当前句")
        self.assertEqual(len(context["messages"]), 2)

    def test_100_context_cutoff_excludes_future_even_with_same_timestamp(self):
        self.store.ingest("a", "甲", [message(i, stamp=100) for i in range(1, 151)])
        with self.store.connect() as db:
            anchor = db.execute("SELECT id FROM messages WHERE sort_seq=110").fetchone()[0]
        context = self.store.context("a", through_id=anchor)
        self.assertEqual(len(context["messages"]), 100)
        self.assertEqual(context["messages"][0]["text"], "测试记录 11")
        self.assertEqual(context["messages"][-1]["text"], "测试记录 110")
        self.assertEqual(context["history_watermark"][1], 110)

    def test_extended_context_preserves_300_usable_messages_and_anchor_metadata(self):
        rows = []
        for i in range(1, 351):
            rows.append(message(i * 2 - 1, "合成文字" + str(i), stamp=100))
            media = message(i * 2, "[图片]", stamp=100)
            media["kind"] = 3
            rows.append(media)
        self.store.ingest("a", "甲", rows)
        with self.store.connect() as db:
            anchor = db.execute("SELECT id FROM messages WHERE sort_seq=?", (619,)).fetchone()[0]
        context = self.store.context("a", through_id=anchor, source_history_limit=300)
        self.assertEqual(len(context["messages"]), 300)
        self.assertEqual(context["messages"][0]["text"], "合成文字11")
        self.assertEqual(context["messages"][-1]["text"], "合成文字310")
        self.assertEqual(context["selected_message"]["id"], anchor)
        self.assertEqual(context["history_watermark"], [100, 619, anchor])
        self.assertEqual(context["context_message_limit"], 300)
        self.assertEqual(context["source_history_limit"], 300)
        for index, (raw, metadata, identity) in enumerate(
            zip(context["messages"], context["context_messages"], context["history_ids"])
        ):
            self.assertEqual(metadata["index"], index)
            self.assertEqual(metadata["message_id"], identity)
            self.assertEqual(metadata["role"], raw["role"])
            self.assertEqual(metadata["timestamp"], 100)
        self.assertEqual(len(self.store.context("a")["messages"]), 100)
        self.assertEqual(len(self.store.recent("a")["messages"]), 100)

    def test_source_history_limit_validates_bounds_and_one_nontext_anchor(self):
        voice = message(2, "[语音]")
        voice["kind"] = 34
        self.store.ingest("a", "甲", [message(1), voice])
        identity = self.store.recent("a")["messages"][-1]["id"]
        one = self.store.context("a", through_id=identity, source_history_limit=1)
        self.assertEqual(len(one["messages"]), 1)
        self.assertEqual(one["history_ids"], [identity])
        self.assertEqual(one["context_messages"][0]["index"], 0)
        self.assertEqual(self.store.context("a", source_history_limit=1000)["source_history_limit"], 1000)
        for value in (0, 1001, True, False, 300.0, "300", None):
            with self.assertRaises(ValueError):
                self.store.context("a", source_history_limit=value)

    def test_extended_summary_keeps_all_sources_and_contact_ownership(self):
        self.store.ingest("a", "甲", [message(i) for i in range(1, 301)])
        self.store.ingest("b", "乙", [message(301)])
        context = self.store.context("a", source_history_limit=300)
        self.store.save_summary("a", context["history_ids"], {"summary": "合成摘要", "evidence": "合成依据"})
        summaries = self.store.context("a", source_history_limit=300)["past_summaries"]
        self.assertEqual(json.loads(summaries[0]["message_ids"]), context["history_ids"])
        with self.assertRaises(ValueError):
            self.store.save_summary("b", context["history_ids"], {"summary": "合成摘要", "evidence": "合成依据"})
        with self.assertRaises(ValueError):
            self.store.save_summary("a", context["history_ids"] * 4, {"summary": "合成摘要", "evidence": "合成依据"})


if __name__ == "__main__":
    unittest.main()
