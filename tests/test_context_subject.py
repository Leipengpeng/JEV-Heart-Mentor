"""100-message grounding and opponent-only focus using entirely synthetic text."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from chatlens import card_engine as cards
from chatlens.analysis_engine import analysis_focus, validate_messages, MAX_MESSAGE_CHARS
from test_cards import proposal, emotion, render_details, search_selection, render_once_result
from test_card_emotions import Decider, Tokenizer


class ContextSubjectTests(unittest.TestCase):
    def test_selected_other_message_is_subject_and_later_self_or_other_are_excluded(self):
        raw = [
            {"role": "self", "text": "我方背景"},
            {"role": "other", "text": "所选对方句"},
            {"role": "self", "text": "未来自己回复"},
            {"role": "other", "text": "未来对方句"},
        ]
        messages, subject = analysis_focus(
            validate_messages(raw),
            {
                "history_ids": ["a", "b", "c", "d"],
                "selected_message": {"id": "b", "role": "other", "text": "所选对方句", "timestamp": 100},
            },
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual(subject["speaker"], "other")
        self.assertEqual(subject["message_id"], "b")
        self.assertEqual(subject["text"], "所选对方句")
        self.assertEqual(subject["timestamp"], 100)
        self.assertNotIn("未来", json.dumps(messages, ensure_ascii=False))

    def test_duplicate_text_selected_id_keeps_correct_boundary(self):
        raw = [
            {"role": "other", "text": "好的"},
            {"role": "self", "text": "此前背景"},
            {"role": "other", "text": "好的"},
        ]
        messages, subject = analysis_focus(
            validate_messages(raw),
            {"history_ids": ["a", "b", "c"], "selected_message": {"id": "a", "role": "other", "text": "好的"}},
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(subject["message_id"], "a")

    def test_manual_latest_other_subject_does_not_analyze_last_self_reply(self):
        raw = [{"role": "other", "text": "对方当前句"}, {"role": "self", "text": "我方后续回复"}]
        messages, subject = analysis_focus(validate_messages(raw), {})
        self.assertEqual(len(messages), 1)
        self.assertEqual(subject["text"], "对方当前句")
        with self.assertRaises(ValueError):
            analysis_focus(validate_messages([raw[1]]), {})
        with self.assertRaises(ValueError):
            analysis_focus(validate_messages(raw), {"selected_message": {"role": "self", "text": "我方后续回复"}})

    def test_jev_search_covers_all_100_messages_and_times_before_selected_boundary(self):
        raw = [
            {
                "role": "other" if i % 2 == 0 or i == 99 else "self",
                "text": ("早期明确约定" if i == 0 else "合成背景" + str(i)) + ("文字" * 60),
            }
            for i in range(100)
        ]
        seen = []

        def capture(decider, rows, subject, grounding):
            seen.append((rows, subject, grounding))
            return search_selection(decider, rows, subject, grounding)

        p = proposal()
        p["options"][0]["kind"] = "reply"
        p["options"][0]["queries"] = []
        decision = {"ranked": p["options"], "needs_review": False, "seconds": 0, "p_null": 0}
        metadata = [{"index": i, "timestamp": 1700000000 + i, "message_id": str(i)} for i in range(100)]
        raw.append({"role": "other", "text": "未来禁止读取"})
        metadata.append({"index": 100, "timestamp": 1700000100, "message_id": "100"})
        with (
            patch.object(cards, "search_context", side_effect=capture),
            patch.object(cards, "rank_raw_context", return_value=decision),
            patch.object(cards, "rank_emotions", return_value=emotion()),
            patch.object(cards, "render_once", side_effect=render_once_result),
        ):
            result = cards.analyze_cards(
                SimpleNamespace(decider=object()),
                {
                    "messages": raw,
                    "context_messages": metadata,
                    "history_ids": [str(i) for i in range(101)],
                    "selected_message": {"id": "99", "role": "other", "text": raw[99]["text"]},
                },
                lambda x: None,
            )
        rows, subject, grounding = seen[0]
        self.assertEqual(len(rows), 100)
        self.assertEqual(rows[0]["source_index"], 0)
        self.assertEqual(rows[99]["source_index"], 99)
        self.assertEqual(rows[0]["timestamp"], 1700000000)
        self.assertIn("早期明确约定", rows[0]["text"])
        self.assertGreater(sum(len(row["text"]) for row in rows), 5000)
        self.assertNotIn("未来禁止读取", json.dumps(rows, ensure_ascii=False))
        self.assertEqual(grounding["reference_window_indices"], list(range(88, 100)))
        self.assertEqual(result["messages_used"], 100)
        self.assertFalse(result["context_limits"]["aggregate_tail_truncated"])
        self.assertEqual(result["analysis_subject"]["speaker"], "other")
        self.assertEqual(result["context_limits"]["message_texts_truncated"], 0)
        self.assertIn("不描述或评判我方行为", cards.PROMPT)
        self.assertIn("后来明确更正优先", cards.PROMPT)

    def test_per_message_cap_is_explicit_without_dropping_message_count(self):
        raw = [{"role": "other", "text": "长" * 700} for _ in range(100)]
        messages = validate_messages(raw)
        self.assertEqual(len(messages), 100)
        self.assertEqual(len(messages[0]["text"]), MAX_MESSAGE_CHARS)
        from chatlens.analysis_engine import context_limits

        limits = context_limits(raw, messages, "合成对话")
        self.assertEqual(limits["message_texts_truncated"], 100)
        self.assertEqual(limits["message_char_limit"], 512)

    def test_single_pass_packing_keeps_anchor_recent_window_and_raw_provenance_under_budget(self):
        from chatlens.single_pass_context import prepare_single_payload
        from chatlens.context_budget import prompt_tokens

        rows = [
            {
                "source_index": i,
                "role": "self" if i % 2 else "other",
                "text": ("久远原文" * 90) if i < 88 else f"近邻原文{i}",
                "timestamp": 1700000000 + i,
            }
            for i in range(100)
        ]
        rows[-1] = {"source_index": 99, "role": "other", "text": "这样加上会怎样？", "timestamp": 1700000099}
        selected = {**rows[0], "retrieval_fragment": {"text": rows[0]["text"][:32]}}
        payload = {
            "_context_rows": rows,
            "analysis_subject": {"speaker": "other", "text": rows[-1]["text"]},
            "jev_selected_rows": [selected],
            "requires_reference_resolution": True,
            "focus_topics": [],
            "jev_search_metrics": {"all_sources_covered": True},
        }
        tokenizer = Tokenizer()
        bounded, scope = prepare_single_payload(
            tokenizer, cards.SINGLE_PROMPT, payload, proposal()["options"], emotion()
        )
        self.assertTrue(scope["all_sources_covered"])
        self.assertFalse(scope["all_original_messages_sent_to_llm"])
        self.assertTrue(scope["near_context_complete"])
        self.assertEqual(scope["recent_raw_count"], 12)
        self.assertEqual(bounded["current_other_message"], rows[-1]["text"])
        self.assertIn("88.", bounded["conversation"])
        self.assertIn("99.", bounded["conversation"])
        self.assertEqual(bounded["matched_original_messages"][0]["index"], 0)
        self.assertEqual(bounded["matched_original_messages"][0]["text"], rows[0]["text"][:32])
        self.assertTrue(bounded["matched_original_messages"][0]["is_partial_quote"])
        self.assertEqual(scope["llm_compression_calls"], 0)
        self.assertLessEqual(
            prompt_tokens(tokenizer, cards.SINGLE_PROMPT, bounded) + scope["thinking_reserve_tokens"] + 32, 3000
        )

    def test_single_pass_packing_rejects_selected_evidence_outside_source_boundary(self):
        from chatlens.single_pass_context import prepare_single_payload

        rows = [{"source_index": 0, "role": "other", "text": "当前所选句。"}]
        payload = {
            "_context_rows": rows,
            "analysis_subject": {"speaker": "other", "text": rows[0]["text"]},
            "jev_selected_rows": [{"source_index": 1, "role": "other", "text": "未来不能引用。"}],
            "jev_search_metrics": {"all_sources_covered": True},
        }
        with self.assertRaises(ValueError):
            prepare_single_payload(Tokenizer(), cards.SINGLE_PROMPT, payload, proposal()["options"], emotion())

    def test_jev_retains_grounded_early_quotes_and_summary_inside_full_budget(self):
        messages = [
            {"role": "self" if i % 2 else "other", "text": "合成很长背景" + str(i) + "文字" * 200} for i in range(100)
        ]
        messages[-1] = {"role": "other", "text": "当前确认句"}
        p = proposal()
        p["context_notes"] = {"summary": "早期已经约定周五安排，需核对是否已更正。", "evidence_indices": [0, 2]}
        decider = Decider()
        decider.max_state = 1024
        with patch("torch.mps.synchronize"):
            decision = cards.rank(
                SimpleNamespace(decider=decider),
                messages,
                p,
                [],
                "自然回应",
                {"speaker": "other", "text": "当前确认句", "message_id": "subject-id"},
            )
        state = decider.calls[0][0]
        self.assertEqual(next(iter(state)), "selected_other_message")
        self.assertIn("uncertain_context_notes", state)
        self.assertEqual([q["index"] for q in state["uncertain_context_notes"]["source_quotes"]], [0, 2])
        self.assertLessEqual(decision["state_tokens"], 960)
        self.assertEqual(decision["input_scope"]["summary_source_count"], 100)
        self.assertGreaterEqual(decision["input_scope"]["raw_messages_count"], 3)
        self.assertLess(decision["input_scope"]["raw_messages_count"], 100)

    def test_emotion_older_evidence_is_original_text_without_generated_summary(self):
        raw = [
            {"role": "other", "text": "earlier-raw-marker"},
            {"role": "self", "text": "context"},
            {"role": "other", "text": "synthetic-current-subject"},
        ]
        decider = Decider()
        decider.max_state = 1024
        with patch("torch.mps.synchronize"):
            result = cards.rank_emotions(SimpleNamespace(decider=decider), raw, {"source": "history"}, [0])
        state = decider.calls[0][0]
        self.assertEqual(state["older_raw_quotes"][0]["text"], "earlier-raw-marker")
        self.assertNotIn("uncertain_context_notes", state)
        self.assertEqual(result["input_scope"]["summary_source_count"], 0)
        self.assertEqual(result["input_scope"]["evidence_selection"], "llm_selected_raw_indices")

    def test_notes_citations_are_bounded_and_summary_without_source_is_rejected(self):
        for notes in (
            {"summary": "无来源推断", "evidence_indices": []},
            {"summary": "不合法来源", "evidence_indices": [1000]},
        ):
            p = proposal()
            p["context_notes"] = notes
            with self.assertRaises(ValueError):
                cards.parse_proposal(json.dumps(p, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
