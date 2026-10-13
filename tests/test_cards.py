import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from chatlens import card_engine as cards
from chatlens.history_connector import select_source


def proposal():
    return {
        "analysis": "需要核对过去约定。",
        "evidence": "对方提到上周。",
        "uncertainties": [],
        "options": [
            {
                "id": "a1",
                "action": "回想",
                "detail": "核对上周约定。",
                "kind": "recall",
                "queries": ["约定"],
                "reply": "",
            },
            {
                "id": "a2",
                "action": "追问",
                "detail": "确认具体事项。",
                "kind": "clarify",
                "queries": [],
                "reply": "你指哪件事？",
            },
            {"id": "a3", "action": "沉默", "detail": "等待补充信息。", "kind": "pause", "queries": [], "reply": ""},
        ],
    }


def emotion():
    return {
        "seconds": 0.2,
        "available": True,
        "fixed_labels": list(cards.EMOTION_LABELS),
        "ranked": [{"label": "平静", "score": 1, "score10": 10, "rank": 1}],
        "selected_label": "平静",
        "p_null": 0.01,
        "needs_review": False,
    }


def render_details(engine, payload, decision, phase, measurements):
    return [{key: row[key] for key in ("id", "action", "detail", "reply")} for row in decision["ranked"][:3]]


def search_selection(decider, rows, subject, grounding):
    return {
        "selected_rows": rows[:1] if len(rows) > 12 else [],
        "metrics": {
            "all_sources_covered": True,
            "source_messages_count": len(rows),
            "elapsed_seconds": 0,
            "candidate_slices": 1,
            "initial_chunks": 1 if len(rows) > 12 else 0,
        },
    }


def render_once_result(engine, payload, decision, emotion_result, phase, measurements):
    result = proposal()
    result["context_notes"] = {"summary": "", "evidence_indices": []}
    result["options"] = render_details(engine, payload, decision, phase, measurements)
    result["action_list_produced"] = False
    return result


class CardProtocolTests(unittest.TestCase):
    def test_invalid_actions_are_rejected(self):
        for change in ("duplicate", "send", "one_option"):
            p = proposal()
            if change == "duplicate":
                p["options"][1]["action"] = p["options"][0]["action"]
            if change == "send":
                p["options"][1]["kind"] = "send"
            if change == "one_option":
                p["options"] = p["options"][:1]
            if change == "missing_query":
                p["options"][0]["queries"] = []
            with self.assertRaises(ValueError):
                cards.parse_proposal(json.dumps(p, ensure_ascii=False))

    def test_jev_ignores_llm_shortlist_scores_full_bank_and_restores_token_limit(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text)

            def decode(self, tokens):
                return "".join(tokens)

        class Decider:
            max_option_tokens = 24
            tok = Tokenizer()

            def decide(self, state, question, labels):
                self.seen = labels
                scores = [0.2 / 30] * 31
                scores[next(i for i, r in enumerate(cards.KNOWLEDGE["actions"]) if r["id"] == "probe")] = 0.8
                return dict(zip(labels, scores)), {"p_null": 0.02}

        decider = Decider()
        engine = SimpleNamespace(decider=decider)
        with patch("torch.mps.synchronize"):
            decision = cards.rank(engine, [{"role": "other", "text": "上周的约定呢？"}], proposal(), [], "确认具体事项")
        self.assertEqual(decision["selected_id"], "probe")
        self.assertEqual(len(decider.seen), 31)
        self.assertEqual(len(decision["ranked"]), 31)
        self.assertFalse(decision["llm_preselection"])
        self.assertEqual(decision["selection_policy"], "jev_full_fixed_bank")
        self.assertEqual(decision["ranked"][0]["rank"], 1)
        self.assertEqual(decision["ranked"][0]["score"], 0.8)
        self.assertEqual(decision["ranked"][0]["score10"], 8.0)
        self.assertTrue(all(0 <= row["score10"] <= 10 for row in decision["ranked"]))
        self.assertEqual(decider.max_option_tokens, 24)
        self.assertIn(cards.STRATEGY_BANK[0][2], decider.seen[0])

    def test_jev_history_search_and_choices_precede_only_one_render(self):
        order = []

        def search(*args):
            order.append("SEARCH")
            return search_selection(*args)

        def generate(*args):
            order.append("LLM")
            return render_once_result(*args)

        def rank(*args):
            order.append("JEV")
            p = proposal()
            rows = p["options"]
            return {"ranked": rows, "needs_review": False, "seconds": 0.1, "p_null": 0.01}

        def lookup(queries):
            self.fail("The JEV-first single-pass path must not generate a second analysis via lookup")

        def score_emotion(*args):
            order.append("EMOTION")
            return emotion()

        with (
            patch.object(cards, "search_context", side_effect=search),
            patch.object(cards, "rank_raw_context", side_effect=rank),
            patch.object(cards, "rank_emotions", side_effect=score_emotion),
            patch.object(cards, "render_once", side_effect=generate),
            patch.object(cards, "propose", side_effect=AssertionError("Unexpected LLM preselection")),
        ):
            messages = [{"role": "self", "text": f"合成旧事{i}"} for i in range(13)] + [
                {"role": "other", "text": "上周的约定？"}
            ]
            r = cards.analyze_cards(SimpleNamespace(decider=object()), {"messages": messages}, lambda x: None, lookup)
        self.assertEqual(order, ["SEARCH", "JEV", "EMOTION", "LLM"])
        self.assertTrue(r["retrieval"]["executed"])
        self.assertAlmostEqual(r["timings"]["action_jev_seconds"], 0.1)
        self.assertAlmostEqual(r["timings"]["emotion_seconds"], 0.2)
        self.assertAlmostEqual(r["timings"]["jev_seconds"], 0.3)

    def test_uncertain_jev_choice_does_not_execute_a_tool(self):
        p = proposal()
        decision = {"ranked": p["options"], "needs_review": True, "seconds": 0.1, "p_null": 0.7}
        with (
            patch.object(cards, "search_context", side_effect=search_selection),
            patch.object(cards, "rank_raw_context", return_value=decision),
            patch.object(cards, "rank_emotions", return_value=emotion()),
            patch.object(cards, "render_once", side_effect=render_once_result),
        ):
            r = cards.analyze_cards(
                SimpleNamespace(decider=object()),
                {"messages": [{"role": "other", "text": "约定呢？"}]},
                lambda x: None,
                lambda q: self.fail("Unexpected lookup"),
            )
        self.assertTrue(r["decision"]["needs_review"])

    def test_missing_recall_keywords_do_not_trigger_llm_planned_lookup(self):
        p = proposal()
        p["options"][0]["queries"] = []
        p = cards.parse_proposal(json.dumps(p))
        decision = {"ranked": p["options"], "needs_review": False, "seconds": 0.1, "p_null": 0.01}
        with (
            patch.object(cards, "search_context", side_effect=search_selection),
            patch.object(cards, "rank_raw_context", return_value=decision),
            patch.object(cards, "rank_emotions", return_value=emotion()),
            patch.object(cards, "render_once", side_effect=render_once_result),
        ):
            r = cards.analyze_cards(
                SimpleNamespace(decider=object()),
                {"messages": [{"role": "other", "text": "那件事呢？"}]},
                lambda x: None,
                lambda q: self.fail("Unexpected lookup"),
            )
        self.assertEqual(r["decision"]["ranked"][0]["action"], "回想")


class SourceSelectionTests(unittest.TestCase):
    def test_official_database_is_preferred_with_existing_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ("official", "debug"):
                p = base / name
                (p / "contact").mkdir(parents=True)
                (p / "message").mkdir()
                (p / "contact/contact.db").write_bytes(b"valid")
                (p / "message/message_0.db").write_bytes(b"valid")
            verify = lambda key, page: page == b"valid"
            source, label = select_source(base / "official", base / "debug", [b"key"], verify)
            self.assertEqual(source, base / "official")
            self.assertEqual(label, "official-wechat-database")
            (base / "official/message/message_0.db").write_bytes(b"invalid")
            self.assertEqual(select_source(base / "official", base / "debug", [b"key"], verify)[0], base / "debug")
            (base / "debug/message/message_0.db").write_bytes(b"invalid")
            with self.assertRaises(ValueError):
                select_source(base / "official", base / "debug", [b"key"], verify)


if __name__ == "__main__":
    unittest.main()
