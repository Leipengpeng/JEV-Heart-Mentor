"""Synthetic, offline tests of source quotation compression; no model loading."""

import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from chatlens import history_compressor as compression
from chatlens.context_budget import prompt_tokens, split_earlier


class CharacterTokenizer:
    name_or_path = "synthetic-character-tokenizer"

    def encode(self, text, **kwargs):
        return list(text)

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "</user><assistant><think>"


def row(index, text=None, role=None):
    return {
        "source_index": index,
        "role": role or ("other" if index % 2 else "self"),
        "text": text if text is not None else "合成原文" + str(index),
        "timestamp": 1700000000 + index,
    }


class HistoryCompressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / "synthetic-cache"
        self.cache_patch = mock.patch.object(compression, "CACHE_ROOT", self.cache)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.tokenizer = CharacterTokenizer()
        self.engine = SimpleNamespace(llm=object(), tokenizer=self.tokenizer)
        self.calls = []

    def fake_generate(self, model, tokenizer, user_prompt, **kwargs):
        self.assertIs(model, self.engine.llm)
        self.assertIs(tokenizer, self.tokenizer)
        self.assertTrue(user_prompt.startswith(compression.COMPRESSION_PROMPT))
        body = json.loads(user_prompt[len(compression.COMPRESSION_PROMPT) :])
        self.calls.append(body)
        count = prompt_tokens(tokenizer, compression.COMPRESSION_PROMPT, body)
        self.assertLessEqual(count, compression.SLICE_INPUT_LIMIT)
        # Only extract original fragments; no synthesis or role invention.
        selected = body["messages"][-compression.MAX_QUOTES :]
        value = {
            "items": [
                {"source_index": item["source_index"], "quote": item["text"][:64]} for item in selected if item["text"]
            ]
        }
        return {
            "final_answer": json.dumps(value, ensure_ascii=False),
            "finish_reason": "stop",
            "elapsed_seconds": 0.001,
            "prompt_tokens": count + 16,
            "generation_tokens": 100,
            "thinking_tokens": 16,
            "thinking_budget_forced": False,
        }

    def payload(self, rows):
        return {
            "conversation": compression.format_rows(rows),
            "conversation_message_count": len(rows),
            "analysis_subject": {"speaker": "other", "index": rows[-1]["source_index"], "text": rows[-1]["text"]},
            "current_other_message": rows[-1]["text"],
            "_context_rows": rows,
        }

    def test_quotes_are_exact_from_the_cited_slice_and_roles_are_program_owned(self):
        rows = [row(10, "合成原文：你来提交。", "self"), row(11, "另一条原文", "unknown")]
        result = compression._validate_items(
            {"items": [{"source_index": 11, "quote": "另一条原文"}, {"source_index": 10, "quote": "你来提交。"}]}, rows
        )
        self.assertEqual([item["source_index"] for item in result], [10, 11])
        self.assertEqual([item["role"] for item in result], ["self", "unknown"])
        self.assertEqual(result[0]["timestamp"], rows[0]["timestamp"])
        invalid = [
            {"source_index": 9, "quote": "你来提交。"},
            {"source_index": 11, "quote": "你来提交。"},
            {"source_index": 10, "quote": "你负责提交。"},
            {"source_index": True, "quote": "合成"},
            {"source_index": 10, "quote": "你来提交。", "role": "other"},
        ]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                compression._validate_items({"items": [item]}, rows)

    def test_quote_length_count_and_empty_results_are_bounded(self):
        rows = [row(0, "长" * 100)]
        self.assertEqual(compression._validate_items({"items": []}, rows), [])
        for value in (
            {"items": [{"source_index": 0, "quote": ""}]},
            {"items": [{"source_index": 0, "quote": "长" * 65}]},
            {"items": [{"source_index": 0, "quote": "长"}] * 5},
            {"items": "不是数组"},
        ):
            with self.assertRaises(ValueError):
                compression._validate_items(value, rows)

    def test_cache_hit_revalidates_sources_and_skips_generation(self):
        rows = [row(0, "合成来源零", "self"), row(1, "合成来源一", "other")]
        chunk = split_earlier(
            rows,
            self.tokenizer,
            compression.COMPRESSION_PROMPT,
            compression.SLICE_INPUT_LIMIT,
            base_payload={"current_other_message": "当前合成句"},
        )[0]
        metrics = []
        with mock.patch.object(compression, "generate_reply", side_effect=self.fake_generate) as generation:
            first = compression.compress_chunk(self.engine, chunk, lambda _: None, metrics)
            second = compression.compress_chunk(self.engine, chunk, lambda _: None, metrics)
        self.assertEqual(first, second)
        self.assertEqual(generation.call_count, 1)
        self.assertFalse(metrics[0]["cached"])
        self.assertTrue(metrics[1]["cached"])
        saved = list(self.cache.glob("*.json"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].stat().st_mode & 0o777, 0o600)
        # A bad cached citation must be validated again and recomputed.
        saved[0].write_text(json.dumps({"value": {"items": [{"source_index": 999, "quote": "伪造"}]}}))
        with mock.patch.object(compression, "generate_reply", side_effect=self.fake_generate) as generation:
            repaired = compression.compress_chunk(self.engine, chunk, lambda _: None, metrics)
        self.assertEqual(repaired, first)
        self.assertEqual(generation.call_count, 1)
        self.assertFalse(metrics[-1]["cached"])

    def test_invalid_llm_quote_is_explicit_source_only_fallback(self):
        rows = [row(0, "第一条合成原文", "self"), row(1, "第二条合成原文", "unknown")]
        chunk = split_earlier(
            rows,
            self.tokenizer,
            compression.COMPRESSION_PROMPT,
            compression.SLICE_INPUT_LIMIT,
            base_payload={"current_other_message": "当前合成句"},
        )[0]
        metrics = []
        invalid = {
            "final_answer": json.dumps({"items": [{"source_index": 0, "quote": "完全编造"}]}),
            "finish_reason": "stop",
            "prompt_tokens": chunk["prompt_tokens"],
        }
        with mock.patch.object(compression, "generate_reply", return_value=invalid):
            result = compression.compress_chunk(self.engine, chunk, lambda _: None, metrics)
        self.assertEqual(result, [{**item, "text": item["text"][:64]} for item in rows])
        self.assertFalse(metrics[0]["valid"])
        self.assertTrue(metrics[0]["source_only_fallback"])
        self.assertEqual(list(self.cache.glob("*.json")), [])

    def test_more_than_100_sources_compress_even_if_raw_prompt_would_fit(self):
        rows = [row(i, "句") for i in range(101)]
        rows[-1]["role"] = "other"
        header = "分析当前对方一句。\n"
        payload = self.payload(rows)
        clean = {key: value for key, value in payload.items() if not key.startswith("_")}
        self.assertLessEqual(prompt_tokens(self.tokenizer, header, clean), compression.INITIAL_INPUT_LIMIT)
        metrics = []
        with mock.patch.object(compression, "generate_reply", side_effect=self.fake_generate) as generation:
            fitted, meta = compression.prepare_model_payload(self.engine, payload, header, lambda _: None, metrics)
        self.assertEqual(meta["mode"], "sliced_original_quotes")
        self.assertTrue(generation.called)
        self.assertGreater(meta["compressed_source_messages_count"], 0)
        self.assertEqual(meta["source_messages_count"], 101)
        self.assertEqual(meta["recent_raw_messages_count"] + meta["compressed_source_messages_count"], 101)
        self.assertEqual(meta["selected_source_index"], 100)
        self.assertIn("100.", fitted["conversation"])
        self.assertEqual(fitted["analysis_subject"], payload["analysis_subject"])
        self.assertLessEqual(prompt_tokens(self.tokenizer, header, fitted), compression.INITIAL_INPUT_LIMIT)

    def test_300_sources_are_all_read_and_final_prompt_keeps_selected_original(self):
        rows = [row(i, "第" + str(i) + "条合成原文。" + ("背景文字" * 8)) for i in range(300)]
        rows[-1] = row(299, "唯一的所选对方当前句。", "other")
        header = "分析当前对方一句。\n"
        payload = self.payload(rows)
        metrics = []
        with mock.patch.object(compression, "generate_reply", side_effect=self.fake_generate):
            fitted, meta = compression.prepare_model_payload(self.engine, payload, header, lambda _: None, metrics)
        self.assertEqual(meta["source_messages_count"], 300)
        self.assertTrue(meta["all_sources_covered"])
        self.assertEqual(meta["recent_raw_messages_count"] + meta["compressed_source_messages_count"], 300)
        self.assertGreater(meta["chunks"], 1)
        seen = {item["source_index"] for body in self.calls for item in body["messages"]}
        recent = {int(index) for index in re.findall(r"^(\d+)\.", fitted["conversation"], re.MULTILINE)}
        self.assertEqual(seen | recent, set(range(300)))
        self.assertIn("299.", fitted["conversation"])
        self.assertIn(rows[-1]["text"], fitted["conversation"])
        self.assertNotIn("_context_rows", fitted)
        self.assertEqual(meta["initial_input_tokens"], prompt_tokens(self.tokenizer, header, fitted))
        self.assertLessEqual(
            meta["initial_input_tokens"] + compression.THINKING_RESERVE + compression.TEMPLATE_RESERVE, 3000
        )

    def test_unfit_selected_message_fails_instead_of_truncating_or_dropping_it(self):
        rows = [row(0, "之前合成句"), row(1, "当前原文" * 1500, "other")]
        with mock.patch.object(compression, "generate_reply") as generation:
            with self.assertRaises(ValueError):
                compression.prepare_model_payload(self.engine, self.payload(rows), "分析。", lambda _: None, [])
            generation.assert_not_called()

    def test_short_quotes_above_eight_are_kept_without_merge_or_protection_loss(self):
        rows = [{**row(i, "合成约定" + str(i) + "；" + ("背景文字" * 25)), "timestamp": None} for i in range(45)]
        rows[-1]["role"] = "other"
        payload = self.payload(rows)
        payload["rule_grounded_context"] = {
            "latest_confirmed_facts": [
                {"index": 0, "actor": "self", "slot": "task", "value": "合成约定", "status": "active"}
            ]
        }
        header = "分析当前对方一句。\n"
        metrics = []

        def generate_short(model, tokenizer, user_prompt, **kwargs):
            result = self.fake_generate(model, tokenizer, user_prompt, **kwargs)
            value = json.loads(result["final_answer"])
            for item in value["items"]:
                item["quote"] = item["quote"][:1]
            result["final_answer"] = json.dumps(value, ensure_ascii=False)
            return result

        with mock.patch.object(compression, "generate_reply", side_effect=generate_short):
            fitted, meta = compression.prepare_model_payload(self.engine, payload, header, lambda _: None, metrics)
        # All calls must read original slices; an extra call on one-character
        # quotations would be unnecessary lossy merging.
        self.assertGreaterEqual(meta["chunks"], 3)
        self.assertTrue(all(len(item["text"]) > 64 for body in self.calls for item in body["messages"]))
        expected = {item["source_index"] for body in self.calls for item in body["messages"][-compression.MAX_QUOTES :]}
        self.assertGreater(len(expected), 8)
        expected.add(0)  # The protected original is added alongside all short quotes.
        summaries = fitted["conversation_summaries"]
        retained = {
            int(index) for memory in summaries for index in re.findall(r"^(\d+)\.", memory["quotes"], re.MULTILINE)
        }
        self.assertEqual(retained, expected)
        self.assertEqual(meta["merge_rounds"], 0)
        self.assertEqual(meta["summary_cited_messages_count"], len(expected))
        wrapper_cost = prompt_tokens(self.tokenizer, "", {"conversation_summaries": summaries}) - prompt_tokens(
            self.tokenizer, "", {"conversation_summaries": []}
        )
        self.assertLess(wrapper_cost, 700)
        self.assertTrue(meta["all_sources_covered"])
        self.assertIn(rows[-1]["text"], fitted["conversation"])
        self.assertLessEqual(prompt_tokens(self.tokenizer, header, fitted), compression.INITIAL_INPUT_LIMIT)

    def test_multiple_parts_of_one_source_keep_longest_quote_once(self):
        rows = [row(5, "短"), row(5, "同源更长的引文"), row(3, "更早的原文")]
        result = compression._deduplicate(rows)
        self.assertEqual([item["source_index"] for item in result], [3, 5])
        self.assertEqual(result[-1]["text"], "同源更长的引文")


if __name__ == "__main__":
    unittest.main()
