"""Synthetic-only regressions for exact token limits and lossless history slicing."""

import copy
import json
from pathlib import Path
import sys
import unittest

from chatlens.context_budget import (
    ContextBudgetError,
    fit_context_payload,
    indexed_messages,
    prompt_tokens,
    split_earlier,
)


class TemplateTokenizer:
    """One token per codepoint makes boundary results independent of any model."""

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((copy.deepcopy(messages), kwargs))
        return "<user>" + messages[0]["content"] + "</user><assistant><think>"

    def encode(self, text):
        return list(text)


class WeightedTokenizer(TemplateTokenizer):
    def encode(self, text):
        # Chinese costs three here, so a character-based estimate is incorrect.
        return [token for char in text for token in ([char] * (3 if ord(char) > 127 else 1))]


def synthetic_rows(count, width=12, start=0):
    return [
        {
            "source_index": start + index,
            "role": "other" if index % 2 else "self",
            "text": f"合成消息{index}：" + "研" * width,
        }
        for index in range(count)
    ]


def actual_count(tokenizer, header, payload):
    content = header + json.dumps(payload, ensure_ascii=False)
    return len(tokenizer.encode("<user>" + content + "</user><assistant><think>"))


class ContextBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TemplateTokenizer()

    def test_full_header_json_and_thinking_template_are_counted(self):
        payload = {"text": "合成消息", "all_actions": list(range(32))}
        count = prompt_tokens(self.tokenizer, "PROMPT\n", payload)
        self.assertEqual(count, actual_count(self.tokenizer, "PROMPT\n", payload))
        messages, kwargs = self.tokenizer.template_calls[-1]
        self.assertEqual(messages, [{"role": "user", "content": "PROMPT\n" + json.dumps(payload, ensure_ascii=False)}])
        self.assertEqual(kwargs, {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True})

    def test_encode_only_tokenizer_fallback_is_for_offline_fakes(self):
        class Fake:
            def encode(self, text):
                return list(text)

        self.assertEqual(
            prompt_tokens(Fake(), "head", {"text": "合成"}),
            len("head" + json.dumps({"text": "合成"}, ensure_ascii=False)),
        )

    def test_weighted_token_count_cannot_be_replaced_by_character_count(self):
        tokenizer = WeightedTokenizer()
        rows = synthetic_rows(8, width=20)
        chunks = split_earlier(rows, tokenizer, "摘要：", 360)
        for chunk in chunks:
            self.assertLessEqual(actual_count(tokenizer, "摘要：", chunk["payload"]), 360)
        self.assertGreater(len(chunks), 1)

    def test_all_fitting_originals_remain_and_payload_is_not_mutated(self):
        rows = synthetic_rows(5)
        base = {"actions": list(range(32)), "anchor": "最后一句"}
        original = copy.deepcopy((rows, base))
        result = fit_context_payload(base, rows, [], self.tokenizer, "header", 2000)
        self.assertEqual((rows, base), original)
        self.assertEqual(result["recent_messages"], rows)
        self.assertEqual(result["earlier_messages"], [])
        self.assertFalse(result["needs_compression"])
        self.assertEqual(result["payload"]["actions"], list(range(32)))

    def test_exact_anchor_boundary_fits_and_one_less_raises(self):
        rows = synthetic_rows(1)
        body = {"conversation": rows, "conversation_summaries": []}
        budget = actual_count(self.tokenizer, "header", body)
        result = fit_context_payload({}, rows, [], self.tokenizer, "header", budget)
        self.assertEqual(result["prompt_tokens"], budget)
        with self.assertRaises(ContextBudgetError) as captured:
            fit_context_payload({}, rows, [], self.tokenizer, "header", budget - 1)
        self.assertEqual(captured.exception.reason, "mandatory_anchor_too_large")

    def test_tail_is_contiguous_anchor_never_dropped_and_earlier_are_exposed(self):
        rows = synthetic_rows(12)
        last_three = {"conversation": rows[-3:], "conversation_summaries": []}
        limit = actual_count(self.tokenizer, "", last_three)
        result = fit_context_payload({}, rows, [], self.tokenizer, "", limit)
        self.assertEqual(result["recent_messages"], rows[-3:])
        self.assertEqual(result["earlier_messages"] + result["recent_messages"], rows)
        self.assertEqual(result["anchor_source_index"], 11)
        self.assertTrue(result["needs_compression"])
        self.assertFalse(result["minimum_recent_satisfied"])

    def test_formatter_keeps_original_source_numbers_and_string_style(self):
        rows = synthetic_rows(4, start=96)

        def formatter(items):
            return "\n".join(f"{row['source_index']}. {row['role']}：{row['text']}" for row in items)

        result = fit_context_payload({}, rows, [], self.tokenizer, "", 1000, conversation_formatter=formatter)
        self.assertIsInstance(result["payload"]["conversation"], str)
        self.assertTrue(result["payload"]["conversation"].startswith("96."))
        self.assertIn("99. other：", result["payload"]["conversation"])
        self.assertEqual(result["prompt_tokens"], actual_count(self.tokenizer, "", result["payload"]))

    def test_summaries_and_full_catalog_survive_even_when_recent_count_shrinks(self):
        rows = synthetic_rows(10)
        memories = [{"range": [0, 5], "quotes": "0. self：合成约定；5. other：合成更正"}]
        base = {"actions": [{"id": i} for i in range(32)]}
        limit = actual_count(
            self.tokenizer, "", {**base, "conversation": rows[-2:], "conversation_summaries": memories}
        )
        result = fit_context_payload(base, rows, memories, self.tokenizer, "", limit)
        self.assertEqual(result["payload"]["conversation_summaries"], memories)
        self.assertEqual(len(result["payload"]["actions"]), 32)
        self.assertEqual(result["recent_messages"], rows[-2:])

    def test_oversize_summary_is_explicit_error_not_silent_drop(self):
        rows = synthetic_rows(1)
        limit = actual_count(self.tokenizer, "", {"conversation": rows, "conversation_summaries": []}) + 20
        with self.assertRaises(ContextBudgetError) as captured:
            fit_context_payload({}, rows, [{"range": [0, 0], "quotes": "合" * 500}], self.tokenizer, "", limit)
        self.assertEqual(captured.exception.reason, "summaries_too_large")
        self.assertNotIn("合", str(captured.exception))

    def test_huge_selected_sentence_is_not_silently_truncated(self):
        rows = [{"role": "other", "text": "超长原话" * 1000}]
        with self.assertRaises(ContextBudgetError) as captured:
            fit_context_payload({}, rows, [], self.tokenizer, "", 300)
        self.assertEqual(captured.exception.reason, "mandatory_anchor_too_large")
        self.assertEqual(rows[0]["text"], "超长原话" * 1000)

    def test_chunking_covers_every_original_exactly_once_in_order(self):
        rows = synthetic_rows(101, width=25)
        before = copy.deepcopy(rows)
        chunks = split_earlier(rows, self.tokenizer, "压缩", 360)
        flattened = [row for chunk in chunks for row in chunk["messages"]]
        self.assertEqual(flattened, rows)
        self.assertEqual(rows, before)
        self.assertTrue(all(chunk["prompt_tokens"] <= 360 for chunk in chunks))
        self.assertEqual([index for chunk in chunks for index in chunk["source_indices"]], list(range(101)))

    def test_long_single_message_parts_rejoin_losslessly_with_same_id(self):
        text = '合成🙂\n引号"及\\字符' * 120
        source = {"source_index": 42, "id": "synthetic-original-id", "role": "other", "text": text}
        chunks = split_earlier([source], self.tokenizer, "header", 330)
        parts = [row for chunk in chunks for row in chunk["messages"]]
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(row["text"] for row in parts), text)
        self.assertEqual([row["part"] for row in parts], list(range(1, len(parts) + 1)))
        self.assertTrue(all(row["source_index"] == 42 and row["id"] == "synthetic-original-id" for row in parts))
        self.assertEqual(parts[0]["char_start"], 0)
        self.assertEqual(parts[-1]["char_end"], len(text))
        for previous, following in zip(parts, parts[1:]):
            self.assertEqual(previous["char_end"], following["char_start"])
        self.assertTrue(all(actual_count(self.tokenizer, "header", chunk["payload"]) <= 330 for chunk in chunks))

    def test_empty_text_and_existing_timestamp_are_preserved(self):
        rows = [{"index": 8, "role": "other", "text": "", "timestamp": 123}]
        chunk = split_earlier(rows, self.tokenizer, "", 300)[0]
        self.assertEqual(chunk["messages"][0], {**rows[0], "source_index": 8})

    def test_subject_boundary_prevents_future_rows_in_plan_and_chunks(self):
        rows = synthetic_rows(8)
        result = fit_context_payload({}, rows, [], self.tokenizer, "", 2000, subject_index=4)
        self.assertEqual(result["anchor_source_index"], 4)
        self.assertTrue(all(row["source_index"] <= 4 for row in result["recent_messages"]))
        chunks = split_earlier(rows, self.tokenizer, "", 300, subject_index=4)
        self.assertTrue(all(index <= 4 for chunk in chunks for index in chunk["source_indices"]))

    def test_future_summary_or_missing_anchor_cannot_leak(self):
        rows = synthetic_rows(5)
        with self.assertRaises(ValueError):
            fit_context_payload({}, rows, [{"range": [0, 6], "quotes": "合成未来"}], self.tokenizer, "", 2000)
        with self.assertRaises(ValueError):
            fit_context_payload({}, rows, [], self.tokenizer, "", 2000, subject_index=9)

    def test_invalid_source_order_is_rejected_instead_of_relabelled(self):
        for rows in (
            [{"source_index": 4, "text": "合成"}, {"source_index": 4, "text": "重复"}],
            [{"source_index": 4, "text": "合成"}, {"source_index": 3, "text": "倒序"}],
        ):
            with self.assertRaises(ValueError):
                indexed_messages(rows)

    def test_no_model_or_network_needed_and_mandatory_header_failure_is_explicit(self):
        with self.assertRaises(ContextBudgetError) as captured:
            split_earlier(synthetic_rows(1), self.tokenizer, "header" * 100, 100)
        self.assertEqual(captured.exception.reason, "chunk_header_too_large")
        chunks = split_earlier([], self.tokenizer, "", 100)
        self.assertEqual(chunks, [])

    def test_one_character_plus_part_metadata_must_fit_or_fail(self):
        source = {"source_index": 3, "role": "other", "text": "合成" * 100, "immutable": "x" * 120}
        with self.assertRaises(ContextBudgetError) as captured:
            split_earlier([source], self.tokenizer, "", 150)
        self.assertEqual(captured.exception.reason, "message_metadata_too_large")

    def test_custom_chunk_base_fields_are_retained_and_exactly_measured(self):
        base = {"source_role_key": {"self": "我", "other": "对方"}, "purpose": "合成测试"}
        chunks = split_earlier(
            synthetic_rows(7), self.tokenizer, "摘要", 360, base_payload=base, payload_key="raw_lines"
        )
        for chunk in chunks:
            self.assertEqual(chunk["payload"]["purpose"], "合成测试")
            self.assertEqual(chunk["payload"]["raw_lines"], chunk["messages"])
            self.assertEqual(chunk["prompt_tokens"], actual_count(self.tokenizer, "摘要", chunk["payload"]))

    def test_invalid_budget_and_context_field_collision_rejected(self):
        for budget in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                split_earlier([], self.tokenizer, "", budget)
        with self.assertRaises(ValueError):
            fit_context_payload({"conversation": "do not overwrite"}, synthetic_rows(1), [], self.tokenizer, "", 1000)


if __name__ == "__main__":
    unittest.main()
