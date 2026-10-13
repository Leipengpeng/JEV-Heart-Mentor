"""Interaction strategy choices remain human-readable and native-head bounded."""

import copy
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from chatlens import card_engine as cards
from test_cards import proposal, search_selection
from test_card_emotions import Decider, Tokenizer


def six_choices():
    p = proposal()
    p["analysis"] = "对方这句把球丢过来啦，等你接招呢。"
    p["context_notes"] = {"summary": "", "evidence_indices": []}
    p["options"] = [
        {
            "id": "a" + str(i + 1),
            "strategy_id": key,
            "action": title,
            "detail": "用" + title + "的方式接住当前这句话。",
            "kind": "pause" if title == "沉默" else "reply",
            "queries": [],
            "reply": "",
        }
        for i, (key, title, meaning) in enumerate(cards.STRATEGY_BANK[:5] + (cards.STRATEGY_BANK[28],))
    ]
    return p


def single_response_fixture(analysis):
    original = "这个成绩算上就第一了。"
    payload = {"current_other_message": original, "evidence_to_copy": original, "_visible_source_indices": [0, 1]}
    decision = {"ranked": cards.fixed_catalog_actions({})}
    value = {
        "analysis": analysis,
        "evidence": original,
        "context_notes": {"summary": "前文讨论了加试成绩是否计入总分。", "evidence_indices": [0]},
        "options": [
            {"id": row["id"], "action": row["action"], "detail": "围绕加试成绩回应这份期待。", "reply": ""}
            for row in decision["ranked"][:3]
        ],
    }
    return value, payload, decision


class InteractionChoiceTests(unittest.TestCase):
    def test_reference_bank_has_31_unique_two_character_interactions(self):
        self.assertEqual(len(cards.STRATEGY_BANK), 31)
        self.assertEqual(len({r[0] for r in cards.STRATEGY_BANK}), 31)
        self.assertEqual(len({r[1] for r in cards.STRATEGY_BANK}), 31)
        self.assertTrue(all(re.fullmatch(r"[\u3400-\u9fff]{2}", r[1]) for r in cards.STRATEGY_BANK))
        self.assertTrue({"撒娇", "调侃", "反撩", "直球", "沉默", "夸奖", "安慰"} <= {r[1] for r in cards.STRATEGY_BANK})
        self.assertNotIn("options", cards.ANALYSIS_SCHEMA["properties"])
        self.assertNotIn("strategy_id", cards.ANALYSIS_SCHEMA["properties"])

    def test_parser_accepts_six_and_custom_but_rejects_non_two_character_titles(self):
        p = six_choices()
        self.assertEqual(len(cards.parse_proposal(json.dumps(p))["options"]), 6)
        p["options"][0].update(strategy_id="custom", action="打趣")
        self.assertEqual(cards.parse_proposal(json.dumps(p))["options"][0]["strategy_id"], "custom")
        for title in ("认真听", "夸", "赞!", "A赞", "夸奖一下"):
            broken = copy.deepcopy(p)
            broken["options"][0]["action"] = title
            with self.subTest(title=title), self.assertRaises(ValueError):
                cards.parse_proposal(json.dumps(broken))
        p["options"][0]["strategy_id"] = "unsupported-new-kind"
        with self.assertRaises(ValueError):
            cards.parse_proposal(json.dumps(p))

    def test_jev_ignores_six_llm_choices_and_scores_complete_31_distribution(self):
        p = six_choices()
        scores = [0.64 / 30] * 31
        scores[1] = 0.36
        decider = Decider(scores=scores)
        decider.max_state = 1024
        with patch("torch.mps.synchronize"):
            decision = cards.rank(
                SimpleNamespace(decider=decider), [{"role": "other", "text": "今天好累呀"}], p, [], "自然互动"
            )
        self.assertEqual(len(decider.calls[0][2]), 31)
        self.assertEqual(len(decision["ranked"]), 31)
        self.assertEqual(decision["options_scored"], 31)
        self.assertEqual(decision["candidate_count"], 31)
        self.assertEqual(decision["action_bank_size"], 31)
        self.assertFalse(decision["llm_preselection"])
        self.assertEqual({r["id"] for r in decision["ranked"]}, {r[0] for r in cards.STRATEGY_BANK})
        self.assertEqual(decision["ranked"][0]["action"], "安慰")
        self.assertEqual(decision["ranked"][0]["score10"], 3.6)
        self.assertAlmostEqual(sum(r["score"] for r in decision["ranked"]), 1)
        self.assertLess(sum(r["score"] for r in decision["ranked"][:3]), 1)
        self.assertFalse(decision["needs_review"])
        self.assertLess(decision["review_score_floor"], 0.36)
        self.assertFalse(decision["state_truncated"])
        self.assertEqual(decider.max_option_tokens, 24)

    def test_prompt_uses_bold_other_focus_without_turning_inventions_into_memory(self):
        self.assertIn("大胆演绎", cards.PROMPT)
        self.assertIn("original_style_examples", cards.PROMPT)
        self.assertNotIn("俏皮、活泼、夸张", cards.PROMPT)
        self.assertIn("analysis以“对方”开头", cards.PROMPT)
        self.assertIn("不描述或评判我方行为", cards.PROMPT)
        self.assertIn("context_notes中的日期、偏好、已发生事件必须忠于原文", cards.PROMPT)
        self.assertIn("禁止提出、筛选、排序或选择行动", cards.PROMPT)
        self.assertIn("全部conversation", cards.PROMPT)
        self.assertEqual(cards.ANALYSIS_STYLE, "bold_roleplay")

    def test_real_tokenizer_all_31_full_meanings_and_native_state_fit(self):
        from chatlens.config import MODELS_DIR

        if not (MODELS_DIR / "qwen3-1.7b-base/tokenizer.json").exists():
            self.skipTest("Optional local tokenizer is not downloaded")
        from transformers import AutoTokenizer
        from typical_ai.native import _semif_head_tail

        tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "qwen3-1.7b-base"), local_files_only=True)
        engine = SimpleNamespace(
            decider=SimpleNamespace(tok=tokenizer, max_state=1024, model=SimpleNamespace(render="semif"))
        )
        labels = [title + "：" + meaning for _, title, meaning in cards.STRATEGY_BANK]
        question = {"type": "choice", "instructions": "结合对方当前这句话和前文，选择最自然的互动方式。"}
        limit, scope = cards.option_suffix_budget(engine, question, labels)
        self.assertLessEqual(scope["suffix_tokens"], 1024)
        self.assertLessEqual(limit, 128)
        self.assertEqual(scope["option_texts_truncated"], 0)
        state, scope = cards.bounded_jev_state(
            engine,
            {
                "selected_other_message": "当前对方句",
                "uncertain_context_notes": {
                    "summary": "早期约定需按后续更正核对",
                    "source_quotes": [{"index": 0, "role": "other", "text": "早期真实原文"}],
                },
            },
            [{"role": "other", "text": "背景" * 200} for _ in range(100)],
        )
        native_prefix = _semif_head_tail(tokenizer)[0]
        self.assertLessEqual(len(tokenizer.encode(native_prefix + state, add_special_tokens=False)), 1024)
        self.assertLessEqual(scope["state_tokens"], 960)

    def test_analysis_cannot_propose_actions_or_copy_placeholders(self):
        payload = {"analysis_subject": {"text": "今天好累呀"}, "conversation_message_count": 1}
        base = {
            "context_notes": {"summary": "", "evidence_indices": []},
            "analysis": "对方今天电量见底啦，想被你接住。",
            "evidence": "今天好累呀",
            "uncertainties": [],
            "memory_queries": [],
        }
        self.assertEqual(cards.parse_analysis(json.dumps(base, ensure_ascii=False), payload)["options"], [])
        for mutation in (
            {"options": []},
            {"strategy_id": "comfort"},
            {"analysis": "有原文支持的相关背景"},
            {"evidence": "示例中的一句话"},
            {"context_notes": {"summary": "旧约定", "evidence_indices": [12]}},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                cards.parse_analysis(json.dumps({**base, **mutation}, ensure_ascii=False), payload)

    def test_renderer_rejects_reordered_or_substituted_jev_winners(self):
        selected = cards.fixed_catalog_actions({})[:3]
        decision = {"ranked": selected}
        valid = [{"id": r["id"], "action": r["action"], "detail": "合成互动描述", "reply": ""} for r in selected]

        def render(rows):
            def fake(engine, payload, phase, measurements, prompt, schema, parser, stage, max_tokens):
                self.assertEqual(stage, "render_jev_top3")
                self.assertEqual([r["id"] for r in payload["selected_actions"]], [r["id"] for r in selected])
                return parser(json.dumps({"options": rows}, ensure_ascii=False))

            with patch.object(cards, "_structured", side_effect=fake):
                return cards.write_selected_details(None, {"conversation": "合成原文"}, decision, lambda x: None, [])

        self.assertEqual([r["id"] for r in render(valid)], ["praise", "comfort", "listen"])
        for rows in (
            list(reversed(valid)),
            [{**valid[0], "id": "bold"}] + valid[1:],
            valid[:2],
            [{**valid[0], "action": "直球"}] + valid[1:],
            [{**valid[0], "detail": selected[0]["strategy_meaning"]}] + valid[1:],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                render(rows)

    def test_literal_placeholder_phrase_in_source_is_not_rejected_as_fake_evidence(self):
        subject = "这是待确认信息，先记下来"
        payload = {"analysis_subject": {"text": subject}, "conversation_message_count": 1}
        value = {
            "context_notes": {"summary": "", "evidence_indices": []},
            "analysis": "对方把待办小纸条塞过来啦。",
            "evidence": "待确认信息",
            "uncertainties": [],
            "memory_queries": [],
        }
        self.assertEqual(cards.parse_analysis(json.dumps(value, ensure_ascii=False), payload)["evidence"], "待确认信息")

    def test_unrelated_constraint_does_not_force_fake_food_memory(self):
        messages = [
            {"role": "other", "text": "不要修改项目权限"},
            {"role": "self", "text": "明白"},
            {"role": "other", "text": "今晚餐厅你来定"},
        ]
        candidates = cards.context_attention_candidates(messages, {"text": messages[-1]["text"]})
        self.assertEqual(candidates, [])
        payload = {
            "analysis_subject": {"text": messages[-1]["text"]},
            "conversation_message_count": 3,
            "relevant_context_candidates": candidates,
        }
        value = {
            "context_notes": {"summary": "", "evidence_indices": []},
            "analysis": "对方把选店球抛给你啦。",
            "evidence": "今晚餐厅你来定",
            "uncertainties": [],
            "memory_queries": [],
        }
        cards.parse_analysis(json.dumps(value, ensure_ascii=False), payload)
        messages[0] = {"role": "other", "text": "我不吃辣"}
        payload["relevant_context_candidates"] = cards.context_attention_candidates(
            messages, {"text": messages[-1]["text"]}
        )
        result = cards.parse_analysis(json.dumps(value, ensure_ascii=False), payload)
        self.assertEqual(result["context_notes"]["evidence_indices"], [0])
        self.assertEqual(result["context_notes_origin"], "program_original_cue_fallback")
        self.assertIn("我不吃辣", result["context_notes"]["summary"])

    def test_one_llm_call_follows_jev_search_full_bank_and_emotions(self):
        messages = [{"role": "other" if i % 2 == 0 or i == 99 else "self", "text": f"合成背景{i}"} for i in range(100)]
        decider = Decider()
        decider.max_state = 1024
        engine = SimpleNamespace(decider=decider, llm=object(), tokenizer=Tokenizer())
        seen = []
        search_rows = []

        def search(*args):
            search_rows.extend(args[1])
            return search_selection(*args)

        def generated(llm, tokenizer, prompt, **kwargs):
            self.assertEqual([len(call[2]) for call in decider.calls], [31, 9])
            self.assertEqual(len(search_rows), 100)
            self.assertTrue(prompt.startswith(cards.SINGLE_PROMPT))
            payload = json.loads(prompt[len(cards.SINGLE_PROMPT) :])
            seen.append(payload)
            self.assertEqual(payload["conversation_message_count"], 100)
            self.assertIn("我：合成背景97", payload["conversation"])
            self.assertIn("对方：合成背景88", payload["conversation"])
            self.assertIn("99.", payload["conversation"])
            self.assertIn("analysis", kwargs["json_schema"]["properties"])
            self.assertIn("options", kwargs["json_schema"]["properties"])
            self.assertEqual(len(payload["selected_actions"]), 3)
            self.assertTrue({"id", "action"} <= set(payload["selected_actions"][0]))
            self.assertEqual(payload["reply_perspective"]["speaker"], "self")
            self.assertEqual(payload["current_other_message"], "合成背景99")
            self.assertLessEqual(len(tokenizer.encode(prompt)), 3000)
            value = {
                "context_notes": {"summary": "前文在讨论合成背景。", "evidence_indices": [98]},
                "analysis": "对方接着刚才的合成背景回应啦。",
                "evidence": "合成背景99",
                "options": [
                    {"id": r["id"], "action": r["action"], "detail": f"回应合成背景并{r['action']}", "reply": ""}
                    for r in payload["selected_actions"]
                ],
            }
            return {
                "elapsed_seconds": 0.1,
                "generation_tokens": 30,
                "thinking_tokens": 0,
                "thinking_budget_forced": False,
                "finish_reason": "stop",
                "final_answer": json.dumps(value, ensure_ascii=False),
                "schema_enforced": True,
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "initial_input_tokens": len(tokenizer.encode(prompt)),
                "final_input_tokens": len(tokenizer.encode(prompt)),
            }

        with (
            patch("torch.mps.synchronize"),
            patch.object(cards, "generate_reply", side_effect=generated) as generation,
            patch.object(cards, "search_context", side_effect=search),
            patch.object(cards, "propose", side_effect=AssertionError("Unexpected LLM preselection")),
        ):
            result = cards.analyze_cards(engine, {"messages": messages}, lambda x: None)
        decision = result["decision"]
        self.assertEqual(generation.call_count, 1)
        self.assertEqual(len(seen), 1)
        self.assertEqual([len(call[2]) for call in decider.calls], [31, 9])
        self.assertEqual(len(decision["ranked"]), 31)
        self.assertEqual(len(result["proposal"]["options"]), 3)
        self.assertEqual(decision["selected_id"], decision["ranked"][0]["id"])
        self.assertEqual([r["id"] for r in result["proposal"]["options"]], [r["id"] for r in decision["ranked"][:3]])
        self.assertTrue(decision["rendering_preserved_order"])
        self.assertFalse(result["llm_preselection"])
        self.assertEqual(result["selection_policy"], "jev_full_fixed_bank")
        self.assertAlmostEqual(result["timings"]["llm_seconds"], 0.1)

    def test_invalid_provenance_never_retries_and_incomplete_json_retries_at_most_once(self):
        messages = [{"role": "self", "text": "刚才讨论了合成背景。"}, {"role": "other", "text": "合成背景的下一句。"}]
        for mutation in ("reordered", "future_citation", "wrong_evidence", "unfinished"):
            with self.subTest(mutation=mutation):
                decider = Decider()
                decider.max_state = 1024
                engine = SimpleNamespace(decider=decider, llm=object(), tokenizer=Tokenizer())

                def generated(llm, tokenizer, prompt, **kwargs):
                    self.assertEqual([len(call[2]) for call in decider.calls], [31, 9])
                    if prompt.startswith(cards.SINGLE_PROMPT):
                        payload = json.loads(prompt[len(cards.SINGLE_PROMPT) :])
                    else:
                        from chatlens.writing_correction import CORRECTION_HEADER

                        payload = json.loads(prompt[len(CORRECTION_HEADER) :])["original_input"]
                    options = [
                        {"id": row["id"], "action": row["action"], "detail": "回应合成背景的进展。", "reply": ""}
                        for row in payload["selected_actions"]
                    ]
                    value = {
                        "analysis": "对方仍在讨论合成背景的进展。",
                        "evidence": messages[-1]["text"],
                        "context_notes": {"summary": "前文讨论了合成背景。", "evidence_indices": [0]},
                        "options": options,
                    }
                    if mutation == "reordered":
                        value["options"] = list(reversed(options))
                    if mutation == "future_citation":
                        value["context_notes"]["evidence_indices"] = [2]
                    if mutation == "wrong_evidence":
                        value["evidence"] = "来自未来的原文"
                    return {
                        "elapsed_seconds": 0.1,
                        "generation_tokens": 30,
                        "thinking_tokens": 0,
                        "thinking_budget_forced": False,
                        "finish_reason": "length" if mutation == "unfinished" else "stop",
                        "final_answer": json.dumps(value, ensure_ascii=False)[:20]
                        if mutation == "unfinished"
                        else json.dumps(value, ensure_ascii=False),
                        "schema_enforced": True,
                        "prompt_tokens": len(tokenizer.encode(prompt)),
                    }

                with (
                    patch("torch.mps.synchronize"),
                    patch.object(cards, "search_context", side_effect=search_selection),
                    patch.object(cards, "generate_reply", side_effect=generated) as generation,
                ):
                    with self.assertRaises(ValueError):
                        cards.analyze_cards(engine, {"messages": messages}, lambda x: None)
                self.assertEqual(generation.call_count, 2 if mutation == "unfinished" else 1)

    def test_error_feedback_preserves_full_first_input_and_only_edits_failed_field(self):
        messages = [{"role": "self", "text": "合成背景。"}, {"role": "other", "text": "合成背景的进展。"}]
        decider = Decider()
        decider.max_state = 1024
        engine = SimpleNamespace(decider=decider, llm=object(), tokenizer=Tokenizer())
        calls = []
        original_input = []
        first_output = []

        def generated(llm, tokenizer, prompt, **kwargs):
            calls.append(kwargs)
            self.assertEqual(kwargs["temperature"], 0.0)
            self.assertEqual(kwargs["thinking_temperature"], 0.0)
            self.assertEqual([len(call[2]) for call in decider.calls], [31, 9])
            if len(calls) > 1:
                from chatlens.writing_correction import CORRECTION_HEADER

                feedback = json.loads(prompt[len(CORRECTION_HEADER) :])
                self.assertEqual(feedback["original_input"], original_input[0])
                self.assertEqual(feedback["first_output"], first_output[0])
                self.assertEqual(feedback["mode"], "field_edits")
                self.assertEqual([e["path"] for e in feedback["errors"]], ["analysis"])
                self.assertEqual(feedback["errors"][0]["type"], "instruction_echo")
                return {
                    "elapsed_seconds": 0.1,
                    "generation_tokens": 20,
                    "thinking_tokens": 0,
                    "finish_reason": "stop",
                    "final_answer": json.dumps(
                        {"edits": {"analysis": "对方在回应合成背景的进展。"}}, ensure_ascii=False
                    ),
                    "prompt_tokens": 200,
                }
            data = json.loads(prompt[len(cards.SINGLE_PROMPT) :])
            original_input.append(data)
            value = {
                "analysis": "对方在回应合成背景的进展。",
                "evidence": messages[-1]["text"],
                "context_notes": {"summary": "", "evidence_indices": [0]},
                "options": [
                    {"id": r["id"], "action": r["action"], "detail": "问清合成背景的进展。", "reply": "进展怎么样啦？"}
                    for r in data["selected_actions"]
                ],
            }
            if len(calls) == 1:
                value["analysis"] = "对方俏皮地推进互动。"
            first_output.append(json.dumps(value, ensure_ascii=False))
            return {
                "elapsed_seconds": 0.1,
                "generation_tokens": 40,
                "thinking_tokens": 0,
                "finish_reason": "stop",
                "final_answer": json.dumps(value, ensure_ascii=False),
                "prompt_tokens": 200,
            }

        with (
            patch("torch.mps.synchronize"),
            patch.object(cards, "search_context", side_effect=search_selection),
            patch.object(cards, "generate_reply", side_effect=generated),
        ):
            result = cards.analyze_cards(engine, {"messages": messages}, lambda x: None)
        self.assertEqual(result["llm_call_count"], 2)
        self.assertEqual(result["proposal"]["rewrite_count"], 1)
        self.assertTrue(result["generations"][1]["automatic_retry"])
        self.assertEqual(result["proposal"]["correction_mode"], "field_edits")
        self.assertEqual(result["generations"][1]["stage"], "first_output_error_correction")
        self.assertTrue(result["generations"][1]["input_scope"]["first_input_preserved"])
        self.assertEqual(
            [r["id"] for r in result["proposal"]["options"]], [r["id"] for r in result["decision"]["ranked"][:3]]
        )
        self.assertEqual(result["decision"]["options_scored"], 31)

    def test_uncited_hidden_notes_are_discarded_without_fabricated_sources_or_generation(self):
        original = "这个成绩算上就第一了。"
        payload = {"current_other_message": original, "evidence_to_copy": original, "_visible_source_indices": [0, 1]}
        decision = {"ranked": cards.fixed_catalog_actions({})}
        before = copy.deepcopy(decision)
        value = {
            "analysis": "对方还在比较加试成绩能否计入总分，第一名就差这一步啦。",
            "evidence": original,
            "context_notes": {"summary": "此前提到了加试成绩能计入总分。", "evidence_indices": []},
            "options": [
                {"id": row["id"], "action": row["action"], "detail": "围绕加试成绩回应这份期待。", "reply": ""}
                for row in decision["ranked"][:3]
            ],
        }
        with patch.object(cards, "generate_reply", side_effect=AssertionError("No second model call")) as generation:
            parsed = cards.parse_single_response(json.dumps(value, ensure_ascii=False), payload, decision)
            generation.assert_not_called()
        self.assertEqual(parsed["analysis"], value["analysis"])
        self.assertEqual(parsed["evidence"], original)
        self.assertEqual(parsed["context_notes"], {"summary": "", "evidence_indices": []})
        self.assertEqual(parsed["context_notes_origin"], "missing_citations_discarded")
        self.assertEqual([row["id"] for row in parsed["options"]], [row["id"] for row in decision["ranked"][:3]])
        self.assertEqual(decision, before)
        # Missing sources can be discarded; explicitly false/future citations
        # remain invalid rather than being silently normalized into success.
        value["context_notes"]["evidence_indices"] = [2]
        with self.assertRaises(ValueError):
            cards.parse_single_response(json.dumps(value, ensure_ascii=False), payload, decision)

    def test_raw_action_state_remains_within_1024_when_decider_limit_is_larger(self):
        decider = Decider()
        decider.max_state = 4096
        messages = [{"role": "self" if i % 2 else "other", "text": f"合成原文{i}：" + "背景" * 80} for i in range(30)]
        messages[-1] = {"role": "other", "text": "当前所选原文保持完整。"}
        state, scope = cards.raw_decision_state(
            SimpleNamespace(decider=decider),
            messages,
            {"text": messages[-1]["text"]},
            {"focus_topics": [], "requires_reference_resolution": False},
            [],
            "根据原文自然回应",
        )
        self.assertEqual(scope["state_budget_tokens"], 960)
        self.assertLessEqual(scope["state_tokens"], 960)
        self.assertEqual(scope["state_tokens"], len(decider.tok.encode(state)))
        self.assertEqual(json.loads(state)["selected_other_message"], messages[-1]["text"])
        self.assertTrue(scope["recent_context_truncated"])
        self.assertFalse(scope["llm_analysis_supplied"])
        self.assertEqual(decider.max_state, 4096)

    def test_analysis_at_limit_preserves_completed_sentence_and_discards_cut_off_tail(self):
        first = "对方在比较加试成绩能否计入总分，如果补上就能冲到第一名啦。"
        unfinished = ("他还想继续说明哪些成绩尚未计入总分，" + "补充说明" * 30)[: 80 - len(first)]
        value, payload, decision = single_response_fixture(first + unfinished)
        self.assertEqual(len(value["analysis"]), 80)
        before = copy.deepcopy(decision)
        with patch.object(cards, "generate_reply", side_effect=AssertionError("No repair generation")) as generation:
            parsed = cards.parse_single_response(json.dumps(value, ensure_ascii=False), payload, decision)
            generation.assert_not_called()
        self.assertEqual(parsed["analysis"], first)
        self.assertEqual(parsed["analysis_tail_removed_chars"], len(unfinished))
        self.assertFalse(parsed["analysis_punctuation_added"])
        self.assertEqual(parsed["evidence"], payload["evidence_to_copy"])
        self.assertEqual([row["id"] for row in parsed["options"]], [row["id"] for row in decision["ranked"][:3]])
        self.assertEqual(decision, before)

    def test_analysis_at_limit_without_any_finished_sentence_fails_without_generation(self):
        unfinished = ("对方仍在比较加试成绩是否计入总分，" + "补充说明" * 30)[:80]
        value, payload, decision = single_response_fixture(unfinished)
        self.assertEqual(len(value["analysis"]), 80)
        with patch.object(cards, "generate_reply", side_effect=AssertionError("No repair generation")) as generation:
            with self.assertRaises(ValueError):
                cards.parse_single_response(json.dumps(value, ensure_ascii=False), payload, decision)
            generation.assert_not_called()

    def test_short_complete_analysis_adds_only_punctuation_but_unfinished_connective_fails(self):
        original = "对方想把加试成绩加进总分，看看能不能冲到第一名"
        value, payload, decision = single_response_fixture(original)
        with patch.object(cards, "generate_reply", side_effect=AssertionError("No repair generation")) as generation:
            parsed = cards.parse_single_response(json.dumps(value, ensure_ascii=False), payload, decision)
            self.assertEqual(parsed["analysis"], original + "。")
            self.assertTrue(parsed["analysis_punctuation_added"])
            self.assertEqual(parsed["analysis_tail_removed_chars"], 0)
            value["analysis"] = "对方想把加试成绩加进总分，所以"
            with self.assertRaises(ValueError):
                cards.parse_single_response(json.dumps(value, ensure_ascii=False), payload, decision)
            generation.assert_not_called()

    def test_copied_action_definitions_show_existing_drafts_without_new_generation_or_ranking(self):
        value, payload, decision = single_response_fixture("对方在比较加试成绩能否计入总分，冲第一名就差这一步啦。")
        drafts = ["那就差这份加试成绩啦。", "先别急，看看还能不能补进总分。", "你现在最在意哪项成绩呀？"]
        for index, (row, original) in enumerate(zip(value["options"], decision["ranked"][:3])):
            original.update(score=[0.6, 0.3, 0.1][index], score10=[6.0, 3.0, 1.0][index], rank=index + 1)
            row["detail"] = original["strategy_meaning"]
            row["reply"] = drafts[index]
        before = copy.deepcopy(decision)
        with patch.object(
            cards, "generate_reply", side_effect=AssertionError("No replacement generation")
        ) as generation:
            parsed = cards.parse_single_response(json.dumps(value, ensure_ascii=False), payload, decision)
            generation.assert_not_called()
        self.assertEqual([row["reply"] for row in parsed["options"]], drafts)
        self.assertEqual([row["detail"] for row in parsed["options"]], ["回复：「" + draft + "」" for draft in drafts])
        self.assertEqual([row["id"] for row in parsed["options"]], [row["id"] for row in decision["ranked"][:3]])
        self.assertEqual(
            parsed["writing_adjustments"],
            [
                {"id": row["id"], "reason": "definition_or_duplicate_replaced_by_current_draft"}
                for row in decision["ranked"][:3]
            ],
        )
        self.assertEqual(decision, before)

    def test_copied_nonpause_definition_without_current_draft_still_fails(self):
        value, payload, decision = single_response_fixture("对方在比较加试成绩能否计入总分，冲第一名就差这一步啦。")
        self.assertNotEqual(decision["ranked"][0]["kind"], "pause")
        value["options"][0]["detail"] = decision["ranked"][0]["strategy_meaning"]
        value["options"][0]["reply"] = ""
        with patch.object(
            cards, "generate_reply", side_effect=AssertionError("No replacement generation")
        ) as generation:
            with self.assertRaises(ValueError):
                cards.parse_single_response(json.dumps(value, ensure_ascii=False), payload, decision)
            generation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
