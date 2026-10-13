import tempfile
import unittest
import sys
from pathlib import Path

from chatlens.history_store import HistoryStore
from chatlens.click_selection import resolve


def message(i, text, stamp=100):
    return {
        "local_id": i,
        "server_id": str(i),
        "source": "test.db",
        "timestamp": stamp,
        "sort_seq": i,
        "role": "other",
        "kind": 1,
        "text": text,
    }


class ClickSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = HistoryStore(Path(self.temp.name) / "memory.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_selected_boundary_excludes_later_same_timestamp_messages_and_summaries(self):
        self.store.ingest("a", "甲", [message(1, "此前事实"), message(2, "点击这里"), message(3, "未来答案")])
        rows = self.store.recent("甲")["messages"]
        analysis = {"summary": "未来答案", "evidence": "未来资料"}
        self.store.save_summary("甲", [rows[2]["id"]], analysis)
        self.store.save_summary("甲", [rows[0]["id"]], {"summary": "此前摘要", "evidence": "此前资料"})
        c = self.store.context("甲", through_id=rows[1]["id"])
        self.assertEqual([m["text"] for m in c["messages"]], ["此前事实", "点击这里"])
        self.assertEqual(c["past_summaries"][0]["summary"], "此前摘要")
        self.assertEqual(len(c["past_summaries"]), 1)
        self.assertEqual(c["history_watermark"][1], 2)

    def test_search_applies_boundary_before_limit(self):
        self.store.ingest(
            "a",
            "甲",
            [message(1, "项目交付安排")]
            + [message(i, "此前记录") for i in range(2, 130)]
            + [message(130, "点击位置")]
            + [message(i, "未来项目交付安排") for i in range(131, 170)],
        )
        row = next(m for m in self.store.recent("甲")["messages"] if m["sort_seq"] == 130)
        c = self.store.context("甲", "交付安排", through_id=row["id"])
        self.assertEqual(len(c["history_evidence"]), 1)
        self.assertEqual(c["history_evidence"][0]["text"], "项目交付安排")

    def test_contact_ownership_and_non_text_selection(self):
        row = message(2, "[语音]")
        row["kind"] = 34
        self.store.ingest("a", "甲", [message(1, "此前文字"), row])
        self.store.ingest("b", "乙", [message(3, "乙的内容")])
        voice = self.store.recent("甲")["messages"][1]
        c = self.store.context("甲", through_id=voice["id"])
        self.assertEqual(c["history_ids"][-1], voice["id"])
        self.assertIn("尚未转写", c["messages"][-1]["text"])
        with self.assertRaises(ValueError):
            self.store.context("乙", through_id=voice["id"])

    def test_duplicate_short_messages_are_not_silently_resolved_to_latest(self):
        self.store.ingest("a", "甲", [message(1, "好吧"), message(2, "不同的讨论"), message(3, "好吧")])
        selection = resolve(self.store, "甲", {"clicked_texts": ["好吧"]})
        self.assertEqual(selection["status"], "ambiguous")
        self.assertIsNone(selection["message"])
        selection = resolve(self.store, "甲", {"clicked_texts": ["好吧"], "before_texts": ["不同的讨论"]})
        self.assertEqual(selection["status"], "matched")
        self.assertEqual(selection["message"]["sort_seq"], 3)

    def test_long_fragment_resolves_full_database_text_and_no_match_is_explicit(self):
        self.store.ingest("a", "甲", [message(1, "这是一段完整的长消息，包含换行\n周末确定一下地点。")])
        result = resolve(self.store, "甲", {"clicked_texts": ["周末确定一下地点。"]})
        self.assertEqual(result["status"], "matched")
        self.assertIn("包含换行", result["message"]["text"])
        result = resolve(self.store, "甲", {"clicked_texts": ["完全无关的内容"]})
        self.assertEqual(result["status"], "no_match")
        self.assertIsNone(result["message"])

    def test_ax_single_character_matches_exactly_and_duplicate_remains_ambiguous(self):
        self.store.ingest("a", "甲", [message(1, "嗯"), message(2, "嗯好我知道了")])
        result = resolve(self.store, "甲", {"clicked_texts": ["嗯"], "source": "ax"})
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["message"]["sort_seq"], 1)
        result = resolve(self.store, "甲", {"clicked_texts": ["嗯"], "source": "ocr"})
        self.assertEqual(result["status"], "no_match")
        self.store.ingest("a", "甲", [message(3, "嗯")])
        result = resolve(self.store, "甲", {"clicked_texts": ["嗯"], "source": "ax"})
        self.assertEqual(result["status"], "ambiguous")

    def test_indexed_selection_only_loads_candidates_in_large_history(self):
        self.store.ingest(
            "a", "甲", [message(i, "合成常规记录" + str(i)) for i in range(1, 3000)] + [message(3000, "唯一的目标短语")]
        )
        result = resolve(self.store, "甲", {"clicked_texts": ["唯一的目标短语"], "source": "ax"})
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["message"]["sort_seq"], 3000)

    def test_duplicate_display_names_require_canonical_contact_identity(self):
        self.store.ingest("a", "同名", [message(1, "甲的消息")])
        self.store.ingest("b", "同名", [message(1, "乙的消息")])
        with self.assertRaisesRegex(ValueError, "名称重复"):
            self.store.recent("同名")
        result = resolve(self.store, "b", {"clicked_texts": ["乙的消息"], "source": "ax"})
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["message"]["text"], "乙的消息")

    def test_older_message_update_invalidates_target_generation(self):
        self.store.ingest("a", "甲", [message(1, "旧的检索事实")])
        old = self.store.generation("a")
        self.store.ingest("a", "甲", [message(1, "新的检索事实")])
        self.assertGreater(self.store.generation("a"), old)
        unchanged = self.store.generation("a")
        self.store.ingest("a", "甲", [message(1, "新的检索事实")])
        self.assertEqual(self.store.generation("a"), unchanged)


if __name__ == "__main__":
    unittest.main()
