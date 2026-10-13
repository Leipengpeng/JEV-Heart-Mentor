import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from chatlens import card_engine as cards
from chatlens import galgame_knowledge as knowledge


class GalgameKnowledgeTests(unittest.TestCase):
    def test_runtime_corpus_is_authored_and_free_from_task_echoes(self):
        bank = knowledge.load_bank()
        self.assertEqual(len(bank["actions"]), 31)
        self.assertGreaterEqual(sum(len(r["examples"]) for r in bank["actions"]), 93)
        for action in bank["actions"]:
            self.assertLessEqual(len(action["jev_meaning"]), 15)
            for example in action["examples"]:
                self.assertEqual(example["kind"], "original")
                for field in ("analysis", "detail", "reply"):
                    self.assertIsNone(knowledge.instruction_echo(example[field]), (action["id"], field))

    def test_analysis_reference_has_no_candidate_actions_or_reply_drafts(self):
        rows = knowledge.analysis_examples(cards.KNOWLEDGE, "论文绪论终于改完了")
        self.assertTrue(rows)
        self.assertLessEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(set(row), {"kind", "analysis_patterns"})
            self.assertEqual(row["kind"], "original_sentence_pattern")
            self.assertNotIn("context", row)
            self.assertNotIn("other_message", row)

    def test_writing_references_preserve_jev_choices_and_ignore_other_actions(self):
        chosen = [
            {"id": "comfort", "action": "安慰"},
            {"id": "bold", "action": "直球"},
            {"id": "silence", "action": "沉默"},
        ]
        trigger = next(r for r in cards.KNOWLEDGE["actions"] if r["id"] == "comfort")["examples"][0]["trigger"]
        rows = knowledge.writing_examples(cards.KNOWLEDGE, chosen, trigger)
        self.assertEqual([r["action_id"] for r in rows], [r["id"] for r in chosen])
        self.assertTrue(all(len(r["detail_patterns"]) <= 2 for r in rows))
        self.assertTrue(rows[0]["detail_patterns"])
        self.assertTrue(all(r["kind"] == "original_sentence_pattern" for r in rows))
        self.assertTrue(all("examples" not in r for r in rows))

    def test_reported_echo_is_rejected_but_literal_evidence_is_preserved(self):
        bad = "接住对方进度，俏皮地推进下一步"
        with self.assertRaises(ValueError):
            knowledge.validate_display_text(bad)
        subject = "这种接住对方进度，俏皮地推进下一步的文案很生硬。"
        payload = {"analysis_subject": {"text": subject}, "conversation_message_count": 1}
        value = {
            "analysis": "对方正在嫌弃这句文案，读着像说明书。",
            "evidence": bad,
            "context_notes": {"summary": "", "evidence_indices": []},
            "uncertainties": [],
            "memory_queries": [],
        }
        self.assertEqual(cards.parse_analysis(json.dumps(value, ensure_ascii=False), payload)["evidence"], bad)
        with self.assertRaises(ValueError):
            cards.parse_analysis(
                json.dumps({**value, "analysis": "对方希望采用活泼风格推进互动"}, ensure_ascii=False), payload
            )

    def test_unrelated_examples_are_not_used_and_background_cannot_override_anchor(self):
        self.assertEqual(knowledge.analysis_examples(cards.KNOWLEDGE, "qzxyv"), [])
        chosen = [{"id": "praise", "action": "夸奖"}]
        self.assertEqual(knowledge.writing_examples(cards.KNOWLEDGE, chosen, "qzxyv")[0]["reply_patterns"], [])
        anchor = "面试没过"
        self.assertEqual(
            knowledge.analysis_examples(cards.KNOWLEDGE, anchor),
            knowledge.analysis_examples(cards.KNOWLEDGE, anchor, "蛋糕做得很好看。" * 100),
        )

    def test_literal_style_topic_is_allowed_but_internal_labels_and_half_sentences_fail(self):
        source = "这个幽默风格的海报挺好看"
        knowledge.validate_display_text("对方觉得这个幽默风格的海报好看，等你点个赞。", source)
        for text in (
            "对方（other）正在等你回应。",
            "对方把报告完成并",
            "俏皮地推进下一步",
            "接招用啃萝卜的比喻延续啃骨头的主题",
            "夸他【当前事项】做得漂亮",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                knowledge.validate_display_text(text)

    def test_analysis_enforces_source_quote_and_integer_enum_without_candidate_actions(self):
        payload = {
            "analysis_subject": {"text": "面试没过"},
            "conversation_message_count": 3,
            "conversation": "0. 对方：面试准备了很久\n1. 我：知道啦\n2. 对方：面试没过",
        }

        def structured(engine, body, phase, measurements, prompt, schema, parser, stage, max_tokens):
            self.assertEqual(schema["properties"]["evidence"], {"type": "string", "enum": ["面试没过"]})
            self.assertEqual(
                schema["properties"]["context_notes"]["properties"]["evidence_indices"]["items"]["enum"], [0, 1, 2]
            )
            self.assertNotIn("selected_actions", body)
            self.assertEqual(list(body)[-2:], ["current_other_message", "evidence_to_copy"])
            return {}

        with patch.object(cards, "_structured", side_effect=structured):
            cards.propose(None, payload, lambda x: None, [])

    def test_rare_subject_terms_retain_early_task_owner_cues(self):
        rows = [
            {"role": "other", "text": "我的论文绪论要重写。"},
            {"role": "self", "text": "你改完发给我，我帮你看看绪论。"},
        ]
        rows += [{"role": "self" if i % 2 else "other", "text": "整理桌面这件日常小事"} for i in range(97)]
        rows += [{"role": "other", "text": "论文绪论终于改完了"}]
        cues = cards.context_attention_candidates(rows, {"text": rows[-1]["text"]})
        self.assertEqual([r["index"] for r in cues], [0, 1])

    def test_invalid_hidden_notes_use_only_original_cues_without_invalidating_visible_analysis(self):
        payload = {
            "analysis_subject": {"text": "餐厅你来定吧"},
            "conversation_message_count": 3,
            "relevant_context_candidates": [
                {"index": 0, "role": "other", "text": "我不吃辣", "topic_relevant": True},
                {"index": 1, "role": "self", "text": "八点，我来选餐厅", "topic_relevant": True},
            ],
        }
        for notes in (
            {"summary": "没有相关前文才留空", "evidence_indices": [1]},
            {"summary": "无依据摘要", "evidence_indices": []},
        ):
            value = {
                "context_notes": notes,
                "analysis": "对方把选店球抛给你啦，就等你拍板。",
                "evidence": "餐厅你来定吧",
                "uncertainties": [],
                "memory_queries": [],
            }
            result = cards.parse_analysis(json.dumps(value, ensure_ascii=False), payload)
            self.assertEqual(result["context_notes_origin"], "program_original_cue_fallback")
            self.assertEqual(result["context_notes"]["evidence_indices"], [0, 1])
            self.assertEqual(result["context_notes"]["summary"], "对方：我不吃辣；我：八点，我来选餐厅")
            self.assertNotIn("无依据", result["context_notes"]["summary"])

    def test_renderer_retries_task_echo_and_keeps_winner_id(self):
        selected = cards.fixed_catalog_actions({})[:3]
        rows = [
            {
                "id": r["id"],
                "action": r["action"],
                "detail": "夸他把论文这关拿下了",
                "reply": "论文这关打通了，给你加鸡腿！",
            }
            for r in selected
        ]
        bad = [{**rows[0], "detail": "接住对方进度，俏皮地推进下一步"}] + rows[1:]
        outputs = iter([bad, rows])
        temperatures = []

        def generate(*args, **kwargs):
            temperatures.append(kwargs["temperature"])
            return {
                "elapsed_seconds": 0.1,
                "generation_tokens": 50,
                "thinking_tokens": 0,
                "thinking_budget_forced": False,
                "finish_reason": "stop",
                "final_answer": json.dumps({"options": next(outputs)}, ensure_ascii=False),
            }

        engine = SimpleNamespace(llm=None, tokenizer=None)
        measurements = []
        with patch.object(cards, "generate_reply", side_effect=generate):
            result = cards.write_selected_details(
                engine,
                {"analysis_subject": {"text": "论文终于改完了"}, "conversation": "对方：论文终于改完了"},
                {"ranked": selected},
                lambda value: None,
                measurements,
            )
        self.assertEqual([r["id"] for r in result], [r["id"] for r in selected])
        self.assertEqual(len(measurements), 2)
        self.assertEqual(temperatures, [0.6, 0.6])


if __name__ == "__main__":
    unittest.main()
