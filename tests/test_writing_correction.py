import copy
import json
from pathlib import Path
import sys
import unittest
from chatlens.writing_correction import build_feedback, apply_edits, WritingFieldError
from chatlens.context_budget import ContextBudgetError


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text)


class FeedbackTests(unittest.TestCase):
    def fixture(self):
        bounded = {
            "current_other_message": "合成话题进展。",
            "conversation": "完整的合成双方背景。",
            "matched_original_messages": [{"index": 0, "text": "合成前文。"}],
        }
        options = [
            {"id": str(i), "action": name, "detail": "问清合成事项进展。", "reply": "进展怎么样啦？"}
            for i, name in enumerate(["追问", "直说", "目移"])
        ]
        value = {
            "analysis": "对方在说明合成事项的进展。",
            "options": options,
            "evidence": "合成话题进展。",
            "context_notes": {"summary": "", "evidence_indices": [0]},
        }
        return bounded, value, {"ranked": copy.deepcopy(options)}

    def test_all_bad_fields_are_identified_and_untouched_fields_survive_exactly(self):
        bounded, value, decision = self.fixture()
        before = copy.deepcopy(bounded)
        value["options"][0]["detail"] = "用比喻延续对方的话题。"
        value["options"][1]["reply"] = "俏皮地推进下一步。"
        first = json.dumps(value, ensure_ascii=False)
        _, schema, packet, parsed, scope = build_feedback(
            Tokenizer(),
            "原始任务",
            bounded,
            first,
            WritingFieldError("options.0.detail", "展示文案复述了写作指令"),
            decision,
            {"raw_source_indices": [0, 1]},
        )
        self.assertEqual(packet["original_input"], before)
        self.assertEqual(packet["first_output"], first)
        self.assertEqual({e["path"] for e in packet["errors"]}, {"options.0.detail", "options.1.reply"})
        self.assertEqual(set(schema["properties"]["edits"]["properties"]), {"options.0.detail", "options.1.reply"})
        repaired = json.loads(
            apply_edits(
                json.dumps(
                    {
                        "edits": {
                            "options.0.detail": {"verb": "问", "content": "清合成事项进展。"},
                            "options.1.reply": "这件事还差哪一步呀？",
                        }
                    },
                    ensure_ascii=False,
                ),
                packet,
                parsed,
            )
        )
        self.assertEqual(repaired["analysis"], value["analysis"])
        self.assertEqual(repaired["options"][2], value["options"][2])
        self.assertEqual(repaired["options"][0]["reply"], value["options"][0]["reply"])
        self.assertEqual(bounded, before)
        self.assertTrue(scope["first_input_preserved"] and scope["first_output_preserved"])

    def test_correction_cannot_modify_action_or_insert_unrequested_field(self):
        bounded, value, decision = self.fixture()
        _, _, packet, parsed, _ = build_feedback(
            Tokenizer(),
            "任务",
            bounded,
            json.dumps(value, ensure_ascii=False),
            WritingFieldError("analysis", "句子尚未说完"),
            decision,
            {},
        )
        with self.assertRaises(ValueError):
            apply_edits(
                json.dumps({"edits": {"analysis": "对方在回应进展。", "options.0.action": "直球"}}), packet, parsed
            )

    def test_incomplete_json_feedback_keeps_original_text_and_error(self):
        bounded, value, decision = self.fixture()
        first = '{"analysis":"未结束'
        _, schema, packet, _, _ = build_feedback(
            Tokenizer(), "任务", bounded, first, ValueError("JSON输出未完成"), decision, {}
        )
        self.assertEqual(packet["mode"], "json_restore")
        self.assertEqual(packet["first_output"], first)
        self.assertEqual(packet["original_input"], bounded)
        self.assertIsNone(schema)

    def test_oversize_feedback_fails_without_dropping_original_information(self):
        bounded, value, decision = self.fixture()
        bounded["conversation"] = "完整原文" * 1000
        original = copy.deepcopy(bounded)
        with self.assertRaises(ContextBudgetError):
            build_feedback(
                Tokenizer(),
                "任务",
                bounded,
                json.dumps(value, ensure_ascii=False),
                WritingFieldError("analysis", "句子尚未说完"),
                decision,
                {},
            )
        self.assertEqual(bounded, original)


if __name__ == "__main__":
    unittest.main()
