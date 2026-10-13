"""Synthetic tests: the emotion head receives bounded raw speaker evidence."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chatlens import card_engine as cards


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


class Decider:
    tok = Tokenizer()
    max_option_tokens = 24

    def __init__(self, scores=None, null=0.01):
        self.scores = scores
        self.null = null
        self.calls = []

    def decide(self, state, question, labels):
        self.calls.append((json.loads(state), question, list(labels)))
        scores = self.scores if self.scores is not None else [1.0] + [0.0] * (len(labels) - 1)
        return dict(zip(labels, scores)), {"p_null": self.null}


class EmotionProtocolTests(unittest.TestCase):
    def score(self, messages, scores=None, null=0.01):
        decider = Decider(scores, null)
        with patch("torch.mps.synchronize"):
            result = cards.rank_emotions(
                SimpleNamespace(decider=decider),
                messages,
                {"source": "history", "target": "Synthetic Friend", "through_id": "cutoff-id"},
            )
        return result, decider

    def test_fixed_choices_are_raw_and_bound_to_latest_other_message(self):
        messages = [
            {"role": "other", "text": "earlier-other"},
            {"role": "self", "text": "context-before-subject"},
            {"role": "other", "text": "actual-subject"},
            {"role": "self", "text": "reply-after-subject"},
        ]
        result, decider = self.score(messages)
        state, question, labels = decider.calls[0]
        self.assertEqual(labels, ["平静", "开心", "期待", "关心", "疑惑", "焦虑", "失望", "生气", "难过"])
        self.assertEqual(set(state), {"latest_other_message", "raw_conversation"})
        self.assertEqual(state["latest_other_message"], "actual-subject")
        self.assertIn("context-before-subject", state["raw_conversation"])
        self.assertNotIn("reply-after-subject", json.dumps(state))
        self.assertEqual(question["type"], "choice")
        self.assertEqual(result["scope"]["speaker"], "other")
        self.assertEqual(result["scope"]["through_id"], "cutoff-id")
        self.assertEqual(result["subject_index"], 2)
        self.assertEqual(result["messages_used"], 3)
        self.assertEqual(decider.max_option_tokens, 24)
        self.assertEqual(len(decider.calls), 1)

    def test_selected_emotion_order_ten_point_edges_and_unchanged_native_scores(self):
        scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        result, decider = self.score([{"role": "other", "text": "synthetic-subject"}], scores)
        self.assertEqual(result["selected_label"], "难过")
        self.assertEqual(result["ranked"][0], {"label": "难过", "score": 1.0, "score10": 10.0, "rank": 1})
        self.assertTrue(all(row["score10"] == 0 for row in result["ranked"][1:]))
        self.assertEqual(sum(row["score"] for row in result["ranked"]), 1.0)
        self.assertFalse(result["needs_review"])
        self.assertGreaterEqual(result["seconds"], 0.0)
        self.assertEqual(len(result["input_sha256"]), 64)

    def test_nan_inf_negative_over_one_and_unnormalized_scores_are_rejected(self):
        for score in (float("nan"), float("inf"), -0.1, 1.1, 0.1):
            with self.subTest(score=score):
                with self.assertRaises(RuntimeError):
                    self.score([{"role": "other", "text": "synthetic-subject"}], [score] + [0.0] * 8)
        for null in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(null=null):
                with self.assertRaises(RuntimeError):
                    self.score([{"role": "other", "text": "synthetic-subject"}], null=null)

    def test_all_fixed_choices_must_be_returned(self):
        with self.assertRaises(RuntimeError):
            self.score([{"role": "other", "text": "synthetic-subject"}], [1.0, 0.0])
        probabilities = dict.fromkeys(cards.EMOTION_LABELS, 0.0)
        probabilities["平静"] = 1.0
        probabilities["unexpected"] = 0.0
        with self.assertRaises(RuntimeError):
            cards.checked_scores(probabilities, {"p_null": 0.0}, list(cards.EMOTION_LABELS))

    def test_no_other_speaker_does_not_score_self_or_unknown(self):
        result, decider = self.score(
            [{"role": "self", "text": "my-own-message"}, {"role": "unknown", "text": "unverified-speaker"}]
        )
        self.assertFalse(result["available"])
        self.assertIsNone(result["selected_label"])
        self.assertEqual(result["ranked"], [])
        self.assertEqual(result["p_null"], 1.0)
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["seconds"], 0.0)
        self.assertEqual(decider.calls, [])

    def test_weak_or_rejected_scores_and_voice_preserve_review_metadata(self):
        messages = [{"role": "other", "text": "synthetic-subject"}]
        tied = [0.48, 0.47, 0.05] + [0.0] * 6
        self.assertTrue(self.score(messages, tied)[0]["needs_review"])
        self.assertTrue(self.score(messages, null=0.7)[0]["needs_review"])
        voice = [{"role": "other", "text": "[语音自动转写，待核对] synthetic-subject"}]
        self.assertTrue(self.score(voice)[0]["needs_review"])
        self.assertFalse(self.score(messages)[0]["needs_review"])

    def test_context_tail_is_bounded_but_exact_subject_is_retained(self):
        subject = "last-subject"
        messages = [{"role": "self", "text": "old-" * 600}, {"role": "other", "text": subject}]
        result, decider = self.score(messages)
        state = decider.calls[0][0]
        self.assertEqual(state["latest_other_message"], subject)
        self.assertLessEqual(len(state["raw_conversation"]), 900)
        self.assertTrue(result["context_truncated"])

    def test_pipeline_excludes_generated_and_retrieved_content_from_emotion_input(self):
        from test_cards import proposal, render_once_result, search_selection

        p = proposal()
        p["options"][0]["kind"] = "reply"
        p["options"][0]["queries"] = []
        decision = {"ranked": p["options"], "needs_review": False, "seconds": 0.1, "p_null": 0.01}
        request = {
            "messages": [{"role": "other", "text": "raw-subject"}],
            "source": "history",
            "through_id": "selected-id",
            "history_evidence": [{"text": "retrieved-only-marker", "role": "other"}],
            "past_summaries": [{"summary": "generated-only-marker"}],
        }
        decider = Decider()
        with (
            patch("torch.mps.synchronize"),
            patch.object(cards, "search_context", side_effect=search_selection),
            patch.object(cards, "rank_raw_context", return_value=decision),
            patch.object(cards, "render_once", side_effect=render_once_result),
        ):
            result = cards.analyze_cards(SimpleNamespace(decider=decider), request, lambda x: None)
        state = decider.calls[0][0]
        self.assertEqual(state["latest_other_message"], "raw-subject")
        self.assertNotIn("retrieved-only-marker", json.dumps(state))
        self.assertNotIn("generated-only-marker", json.dumps(state))
        self.assertNotIn(p["analysis"], json.dumps(state))
        self.assertEqual(result["timings"]["action_jev_seconds"], 0.1)
        self.assertEqual(result["timings"]["jev_seconds"], 0.1 + result["timings"]["emotion_seconds"])
        self.assertEqual(result["emotion"]["scope"]["through_id"], "selected-id")
        self.assertEqual(len(decider.calls), 1)


if __name__ == "__main__":
    unittest.main()
