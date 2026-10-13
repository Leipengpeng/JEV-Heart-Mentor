"""Synthetic source-coverage and actual-choice-contract tests, without models."""

import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from chatlens.jev_context_search import QUERY, search_context


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)


class FakeDecider:
    def __init__(self, max_state=1024):
        self.tok = CharacterTokenizer()
        self.max_state = max_state
        self.head = object()
        self.model = SimpleNamespace(render="semif")
        self.cache_requests = []

    def render_choice_suffix(self, query, labels):
        return query + "\n" + "\n".join(str(i) + ". " + value for i, value in enumerate(labels))

    def state_token_count(self, state):
        return len(self.tok.encode(state)) + 12

    def _state_kv_for(self, state):
        self.cache_requests.append(state)
        return object(), self.state_token_count(state)


def sources(count, width=24):
    return [
        {
            "source_index": i,
            "role": "other" if i % 2 else "self",
            "text": "合成原文" + str(i) + "。" + "背景" * width,
            "stable_id": "synthetic-private-source-" + str(i),
        }
        for i in range(count)
    ]


class JEVContextSearchTests(unittest.TestCase):
    def setUp(self):
        self.decider = FakeDecider()
        self.calls = []

    def score(self, state, queries):
        self.calls.append((state, queries))
        result = []
        for query, labels in queries:
            self.assertEqual(query, QUERY)
            weights = [100 if "关键事实" in label else 1 + int(label.split("[")[0]) / 1000 for label in labels]
            denominator = sum(weights) + 0.5
            result.append([value / denominator for value in weights] + [0.5 / denominator])
        return result

    def search(self, rows, **kwargs):
        rows[-1]["role"] = "other"
        focus = kwargs.pop("focus", {"index": rows[-1]["source_index"], "text": rows[-1]["text"]})
        return search_context(self.decider, rows, focus, {}, score_batch=self.score, **kwargs)

    def test_300_sources_are_covered_with_full_anchor_and_no_private_ids_in_prompts(self):
        rows = sources(300)
        rows[47]["text"] = "关键事实：原先少算了两项成绩。"
        value = self.search(rows)
        metrics = value["metrics"]
        self.assertTrue(metrics["all_sources_covered"])
        self.assertEqual(metrics["covered_source_indices"], list(range(300)))
        self.assertIn(47, metrics["primary_selected_source_indices"])
        self.assertLessEqual(metrics["max_suffix_tokens"], 1024)
        self.assertLessEqual(metrics["state_tokens"], 1024)
        self.assertEqual(metrics["option_texts_truncated"], 0)
        self.assertTrue(all(len(queries) <= 4 for _, queries in self.calls))
        for state, queries in self.calls:
            self.assertEqual(json.loads(state)["selected_other_message"]["text"], rows[-1]["text"])
            self.assertNotIn("synthetic-private-source", state)
            self.assertTrue(all("synthetic-private-source" not in label for _, labels in queries for label in labels))
        self.assertNotIn("关键事实", json.dumps(metrics, ensure_ascii=False))
        self.assertNotIn("synthetic-private-source", json.dumps(metrics))
        self.assertNotIn("state", value)

    def test_recent_rows_that_do_not_fit_state_are_scored_as_candidates(self):
        rows = sources(30, width=60)
        value = self.search(rows)
        metrics = value["metrics"]
        self.assertLess(metrics["state_recent_messages_count"], 12)
        self.assertEqual(
            set(metrics["state_source_indices"]) | set(metrics["candidate_source_indices"]), set(range(30))
        )
        self.assertFalse(set(metrics["state_source_indices"]) & set(metrics["candidate_source_indices"]))
        self.assertEqual(metrics["state_source_indices"], list(range(metrics["state_source_indices"][0], 30)))
        self.assertTrue(metrics["all_sources_covered"])

    def test_small_sample_fits_near_context_and_needs_no_retrieval_forward(self):
        rows = sources(3, width=1)
        value = self.search(rows)
        self.assertEqual(value["selected_rows"], [])
        self.assertEqual(value["metrics"]["state_source_indices"], [0, 1, 2])
        self.assertEqual(value["metrics"]["suffix_forward_batches"], 0)
        self.assertEqual(self.calls, [])

    def test_focus_boundary_excludes_future_rows_and_future_keyword(self):
        rows = sources(40)
        rows[25]["role"] = "other"
        rows[35]["text"] = "关键事实：未来绝不能泄露。"
        value = self.search(rows, focus={"index": 25, "text": rows[25]["text"]})
        self.assertEqual(value["metrics"]["covered_source_indices"], list(range(26)))
        self.assertTrue(all(row["source_index"] <= 25 for row in value["selected_rows"]))
        self.assertTrue(
            all(
                "未来绝不能泄露" not in state
                and all("未来绝不能泄露" not in label for _, labels in queries for label in labels)
                for state, queries in self.calls
            )
        )

    def test_anchor_mismatch_duplicate_without_index_and_self_anchor_fail(self):
        rows = sources(8, width=1)
        rows[-1]["role"] = "other"
        for focus in (
            {"index": 7, "text": "不是原文"},
            {"index": 6, "text": rows[6]["text"]},
            {"index": 99, "text": rows[-1]["text"]},
        ):
            with self.assertRaises(ValueError):
                self.search(rows, focus=focus)
        rows[1]["text"] = rows[7]["text"]
        with self.assertRaises(ValueError):
            self.search(rows, focus={"text": rows[7]["text"]})
        self.assertFalse(self.calls)

    def test_long_source_is_split_exhaustively_and_matching_tail_is_preserved(self):
        rows = sources(30, width=2)
        rows[0]["text"] = "普通背景。" * 150 + "关键事实：此前漏算了成绩。" + "后续原文。" * 100
        value = self.search(rows)
        intervals = set()
        for _, queries in self.calls:
            for _, labels in queries:
                for label in labels:
                    match = re.match(r"0\[(\d+):(\d+)\]", label)
                    if match:
                        intervals.add(tuple(map(int, match.groups())))
        intervals = sorted(intervals)
        self.assertEqual(intervals[0][0], 0)
        self.assertEqual(intervals[-1][1], len(rows[0]["text"]))
        self.assertTrue(all(left[1] == right[0] for left, right in zip(intervals, intervals[1:])))
        chosen = next(row for row in value["selected_rows"] if row["source_index"] == 0)
        fragment = chosen["retrieval_fragment"]
        self.assertEqual(fragment["text"], rows[0]["text"][fragment["char_start"] : fragment["char_end"]])
        self.assertIn("关键事实", fragment["text"])
        self.assertEqual(chosen["text"], rows[0]["text"])

    def test_finalists_have_a_joint_comparison_not_cross_group_probability_sort(self):
        rows = sources(180, width=45)
        rows[20]["text"] = "关键事实：真正需要找回的比较条件。" + "补充。" * 20
        value = self.search(rows)
        self.assertTrue(value["metrics"]["global_comparison_performed"])
        self.assertGreater(value["metrics"]["hierarchy_rounds"], 0)
        final_queries = self.calls[-1][1]
        self.assertEqual(len(final_queries), 1)
        labels = final_queries[0][1]
        self.assertTrue(any("20[" in label for label in labels))
        self.assertEqual(value["metrics"]["primary_selected_source_indices"][0], 20)
        self.assertLessEqual(value["metrics"]["selected_rows_count"], 8)
        self.assertEqual(
            [row["relevance_order"] for row in value["selected_rows"]], list(range(len(value["selected_rows"])))
        )

    def test_native_adapter_uses_batch4_cache_and_disables_option_truncation(self):
        rows = sources(60)
        rows[-1]["role"] = "other"

        def native(head, model, state, queries, **kwargs):
            self.assertIs(head, self.decider.head)
            self.assertIs(model, self.decider.model)
            self.assertIsNone(kwargs["max_option_tokens"])
            self.assertEqual(kwargs["max_suffix"], 1024)
            self.assertEqual(kwargs["chunk"], 4)
            self.assertIsNotNone(kwargs["state_cache"])
            return self.score(state, queries)

        typical_module = ModuleType("typical_ai")
        native_module = ModuleType("typical_ai.native")
        native_module.native_kv_decide = native
        typical_module.native = native_module
        with mock.patch.dict(sys.modules, {"typical_ai": typical_module, "typical_ai.native": native_module}):
            value = search_context(self.decider, rows, {"index": 59, "text": rows[-1]["text"]}, {})
        self.assertEqual(len(self.decider.cache_requests), 1)
        self.assertTrue(value["metrics"]["all_sources_covered"])
        self.assertGreater(value["metrics"]["suffix_forward_batches"], 0)

    def test_null_only_queries_do_not_manufacture_matched_rows(self):
        rows = sources(40)
        rows[-1]["role"] = "other"

        def null(state, queries):
            return [[0.0] * len(labels) + [1.0] for _, labels in queries]

        value = search_context(self.decider, rows, {"index": 39, "text": rows[-1]["text"]}, {}, score_batch=null)
        self.assertEqual(value["selected_rows"], [])
        self.assertTrue(value["metrics"]["all_sources_covered"])
        self.assertGreater(value["metrics"]["null_only_questions"], 0)

    def test_invalid_native_probabilities_are_rejected(self):
        rows = sources(20)
        rows[-1]["role"] = "other"
        for kind in ("missing_null", "nan", "negative", "unnormalized"):

            def invalid(state, queries):
                result = []
                for _, labels in queries:
                    count = len(labels)
                    values = [1 / (count + 1)] * (count + 1)
                    if kind == "missing_null":
                        values.pop()
                    if kind == "nan":
                        values[0] = float("nan")
                    if kind == "negative":
                        values[0] = -0.2
                    if kind == "unnormalized":
                        values = [1.0] * (count + 1)
                    result.append(values)
                return result

            with self.subTest(kind=kind), self.assertRaises(ValueError):
                search_context(self.decider, rows, {"index": 19, "text": rows[-1]["text"]}, {}, score_batch=invalid)

    def test_oversize_anchor_fails_without_silent_truncation(self):
        rows = sources(2)
        rows[-1]["text"] = "当前所选原文" * 500
        with self.assertRaises(ValueError):
            self.search(rows)
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
