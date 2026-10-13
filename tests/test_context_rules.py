"""Behavioral regressions for original-message grounding, with no model calls.

These cases are authored synthetic conversations. The tests check fact/actor
binding, chronology and uncertainty rather than reproducing regex patterns.
The UI, contact ordering and JEV candidate bank are outside this module's scope.
"""

import copy
from pathlib import Path
import sys
import unittest

from chatlens.context_rules import extract_grounding
from synthetic_scenarios import cases as galgame_cases


def original_scenarios():
    return {name: copy.deepcopy(messages) for name, messages, _ in galgame_cases()}


def flatten_latest(result):
    return [fact for slots in result["latest"].values() for actors in slots.values() for fact in actors.values()]


def facts(result, *, slot=None, index=None, actor=None, status=None):
    return [
        row
        for row in result["constraints"]
        if (slot is None or row["slot"] == slot)
        and (index is None or row["index"] == index)
        and (actor is None or row["actor"] == actor)
        and (status is None or row["status"] == status)
    ]


class ContextRulesTests(unittest.TestCase):
    def grounding(self, messages, **kwargs):
        return extract_grounding(messages, **kwargs)

    def assert_original_sources(self, result, messages):
        for category in ("relevant_messages", "constraints"):
            for row in result[category]:
                with self.subTest(category=category, index=row["index"]):
                    self.assertIsInstance(row["index"], int)
                    self.assertGreaterEqual(row["index"], 0)
                    self.assertLess(row["index"], len(messages))
                    self.assertEqual(row["role"], messages[row["index"]]["role"])
                    self.assertEqual(row["text"], messages[row["index"]]["text"])

    def test_four_original_synthetic_scenes_keep_provenance_without_mutating_input(self):
        for name, messages in original_scenarios().items():
            before = copy.deepcopy(messages)
            with self.subTest(case=name):
                result = self.grounding(messages)
                self.assertEqual(messages, before)
                self.assertTrue(
                    {"version", "subject_index", "relevant_messages", "constraints", "latest", "ambiguities", "stats"}
                    <= set(result)
                )
                self.assertEqual(result["subject_index"], 99)
                self.assert_original_sources(result, messages)
                self.assertLessEqual(len(result["relevant_messages"]), 8)
                self.assertIn(99, [row["index"] for row in result["relevant_messages"]])
                for row in result["constraints"]:
                    self.assertIn(row["actor"], {"self", "other", "both", "unknown"})
                    self.assertIn(row["status"], {"active", "superseded", "negated", "unknown"})
                    self.assertIsInstance(row["value"], str)
                self.assertTrue(all(row["status"] == "active" for row in flatten_latest(result)))

    def test_omitted_score_object_keeps_two_sides_as_raw_nearby_context(self):
        messages = [
            {"role": "other", "text": "比赛总成绩还没计入加试成绩。"},
            {"role": "self", "text": "加试还有十六分，排在前面的只高出九分。"},
            {"role": "other", "text": "否则算上这一项，可以排第一。"},
        ]
        result = self.grounding(messages)
        self.assertTrue(result["requires_reference_resolution"])
        self.assertEqual(result["reference_window_indices"], [0, 1, 2])
        self.assertEqual(result["near_context"], [dict(index=i, **message) for i, message in enumerate(messages)])
        # A lexical reference rule supplies evidence, not an invented ranking,
        # a person whose grades these are, or a new unsupported topic.
        self.assertEqual(result["focus_topics"], [])
        self.assertFalse(any(row["slot"] == "ranking" for row in result["constraints"]))

    def test_nearby_paper_topic_wins_over_distant_keyword_overlap(self):
        messages = [{"role": "other", "text": "项目也要加上另一个表格。"}]
        messages += [{"role": "self", "text": "收到"} for _ in range(14)]
        messages += [
            {"role": "other", "text": "论文评比有个额外成果还没计入。"},
            {"role": "self", "text": "这项有三十分。"},
            {"role": "other", "text": "不然加上，应该第一。"},
        ]
        result = self.grounding(messages)
        self.assertTrue(result["requires_reference_resolution"])
        self.assertEqual(result["focus_topics"], ["paper"])
        self.assertEqual(result["reference_window_indices"], list(range(6, 18)))
        self.assertEqual(result["near_context"][-3]["index"], 15)
        self.assertNotIn(0, result["reference_window_indices"])

    def test_reference_without_nearby_topic_does_not_jump_to_remote_work_event(self):
        messages = [{"role": "other", "text": "项目报告也要加上。"}]
        messages += [{"role": "self", "text": "收到"} for _ in range(13)]
        messages += [{"role": "other", "text": "不然加上这一项就第一了。"}]
        result = self.grounding(messages)
        self.assertTrue(result["requires_reference_resolution"])
        self.assertEqual(result["focus_topics"], [])
        self.assertTrue(any(row["reason"] == "focus_topic_unknown" for row in result["ambiguities"]))
        self.assertEqual(result["reference_window_indices"], list(range(3, 15)))

    def test_explicit_current_topic_takes_priority_over_nearby_reference_topic(self):
        messages = [
            {"role": "self", "text": "论文还有一个章节。"},
            {"role": "other", "text": "项目报告加上新表格可以提交。"},
        ]
        result = self.grounding(messages)
        self.assertTrue(result["requires_reference_resolution"])
        self.assertEqual(result["focus_topics"], ["work"])

    def test_near_context_stops_at_selected_source_and_keeps_unknown_role(self):
        messages = [
            {"role": "unknown", "text": "刚才那项补充分。"},
            {"role": "self", "text": "还没加上。"},
            {"role": "other", "text": "否则算上应该第一。"},
            {"role": "other", "text": "未来的项目已经改了。"},
        ]
        result = self.grounding(messages, subject={"index": 2, "text": messages[2]["text"]})
        self.assertEqual(result["reference_window_indices"], [0, 1, 2])
        self.assertEqual(result["near_context"][0]["role"], "unknown")
        self.assertTrue(all(row["index"] <= 2 for row in result["near_context"]))
        self.assertEqual(result["near_context"][-1]["text"], messages[2]["text"])
        self.assertEqual(result["focus_topics"], [])

    def test_standalone_demonstratives_mark_resolution_without_guessing_an_object(self):
        for text in ("这。", "那？", "这个有点亏。", "这样就第一了。"):
            with self.subTest(text=text):
                result = self.grounding([{"role": "other", "text": text}])
                self.assertTrue(result["requires_reference_resolution"])
                self.assertEqual(result["reference_window_indices"], [0])
                self.assertEqual(result["focus_topics"], [])

    def test_repeated_desk_background_does_not_replace_early_paper_task(self):
        messages = original_scenarios()["concrete_progress"]
        result = self.grounding(messages)
        selected = {row["index"] for row in result["relevant_messages"]}
        self.assertIn(0, selected)
        self.assertIn(99, selected)
        self.assertNotIn(98, selected)
        self.assertTrue(facts(result, slot="completion", index=99, actor="other"))

    def test_interview_distress_does_not_select_repeated_desk_background(self):
        result = self.grounding(original_scenarios()["low_mood"])
        selected = {row["index"] for row in result["relevant_messages"]}
        self.assertIn(0, selected)
        self.assertIn(99, selected)
        self.assertNotIn(98, selected)

    def test_eight_oclock_correction_supersedes_seven_but_negative_seven_cannot_overwrite_it(self):
        result = self.grounding(original_scenarios()["later_time_correction"])
        current = result["latest"]["dining"]["time"]["both"]
        self.assertEqual(current["value"], "20:00")
        self.assertGreaterEqual(current["index"], 74)
        self.assertTrue(
            any(
                row["value"] == "19:00" and row["status"] == "superseded" for row in facts(result, slot="time", index=0)
            )
        )
        negative = facts(result, slot="time", index=74, status="negated")
        self.assertTrue(any(row["parsed"]["hour"] in {7, 19} for row in negative))
        self.assertFalse(any(row["parsed"]["hour"] in {8, 20} for row in negative))

    def test_no_spicy_preference_survives_long_history_and_time_changes(self):
        result = self.grounding(original_scenarios()["later_time_correction"])
        current = result["latest"]["dining"]["diet_spicy"]["other"]
        self.assertEqual(current["value"], "avoid")
        self.assertEqual(current["index"], 0)
        self.assertEqual(current["actor"], "other")

    def test_different_people_spicy_preferences_do_not_override_each_other(self):
        messages = [
            {"role": "other", "text": "这次聚餐我不吃辣，请选清淡的。"},
            {"role": "self", "text": "这次聚餐我爱吃辣，我自己的菜可以辣一点。"},
        ]
        result = self.grounding(messages)
        current = result["latest"]["dining"]["diet_spicy"]
        self.assertEqual(current["other"]["value"], "avoid")
        self.assertEqual(current["self"]["value"], "prefer")
        self.assertEqual(current["other"]["index"], 0)
        self.assertEqual(current["self"]["index"], 1)

    def test_other_completed_paper_and_self_receives_work_remain_distinct(self):
        messages = [
            {"role": "self", "text": "论文改完发给我，我来接收，我帮你看研究问题。"},
            {"role": "other", "text": "我的论文绪论已经改完了，我发给你。"},
        ]
        result = self.grounding(messages)
        completed = facts(result, slot="completion", index=1)
        self.assertTrue(completed)
        self.assertTrue(all(row["actor"] == "other" for row in completed))
        receiving = facts(result, slot="responsibility", index=0, actor="self")
        self.assertTrue(receiving)
        self.assertEqual(result["latest"]["paper"]["completion"]["other"]["value"], "completed")
        self.assertNotIn("self", result["latest"]["paper"]["completion"])

    def test_other_you_are_late_and_running_penalty_bind_to_self(self):
        messages = original_scenarios()["light_teasing"]
        result = self.grounding(messages)
        current_lateness = facts(result, slot="late", index=99)
        penalty = facts(result, slot="penalty", index=99)
        self.assertTrue(current_lateness)
        self.assertTrue(penalty)
        self.assertTrue(all(row["actor"] == "self" for row in current_lateness + penalty))
        self.assertTrue(all(row["value"] == "conditional_late" for row in current_lateness))
        self.assertTrue(any("罚跑两圈" in row["value"] for row in penalty))

    def test_self_i_choose_restaurant_and_other_you_choose_bind_to_self(self):
        messages = [{"role": "self", "text": "聚餐的餐厅我来选。"}, {"role": "other", "text": "聚餐的餐厅你来定吧。"}]
        result = self.grounding(messages)
        assignments = facts(result, slot="responsibility")
        self.assertTrue(assignments)
        self.assertTrue(all(row["actor"] == "self" for row in assignments))
        current = result["latest"]["dining"]["responsibility"]["self"]
        self.assertEqual(current["value"], "assigned")
        self.assertEqual(current["index"], 1)
        self.assertNotIn("other", result["latest"]["dining"]["responsibility"])

    def test_subject_index_and_matching_text_exclude_future_messages(self):
        messages = [
            {"role": "other", "text": "这次聚餐先约晚上七点。"},
            {"role": "other", "text": "这次聚餐更正为晚上八点。"},
            {"role": "other", "text": "这是未来发言，这次聚餐再改成晚上九点。"},
        ]
        result = self.grounding(messages, subject={"index": 1, "text": messages[1]["text"]})
        self.assertTrue(all(row["index"] <= 1 for row in result["relevant_messages"] + result["constraints"]))
        self.assertEqual(result["latest"]["dining"]["time"]["both"]["value"], "20:00")
        self.assertFalse(any("九点" in row["text"] for row in result["constraints"]))

    def test_existing_pipeline_boundary_then_rules_cannot_recover_future_facts(self):
        from chatlens.analysis_engine import analysis_focus

        messages = [
            {"role": "other", "text": "这次聚餐先约晚上七点。"},
            {"role": "other", "text": "这次聚餐更正为晚上八点。"},
            {"role": "other", "text": "这是未来发言，这次聚餐再改成晚上九点。"},
        ]
        prefix, subject = analysis_focus(messages, {"selected_message": {"role": "other", "text": messages[1]["text"]}})
        self.assertEqual(len(prefix), 2)
        result = self.grounding(prefix, subject=subject)
        self.assertEqual(result["latest"]["dining"]["time"]["both"]["value"], "20:00")
        self.assertTrue(all(row["index"] < len(prefix) for row in result["constraints"]))

    def test_unique_subject_text_can_cut_off_future_without_guessing_an_index(self):
        messages = [
            {"role": "other", "text": "这次聚餐约晚上八点。"},
            {"role": "self", "text": "未来的我还想改为晚上九点。"},
        ]
        result = self.grounding(messages, subject={"text": messages[0]["text"]})
        self.assertTrue(all(row["index"] == 0 for row in result["constraints"]))
        self.assertEqual(result["latest"]["dining"]["time"]["both"]["value"], "20:00")

    def test_duplicate_subject_text_is_marked_ambiguous_instead_of_cutting_at_first_match(self):
        messages = [
            {"role": "other", "text": "餐厅你来定吧。"},
            {"role": "self", "text": "这次聚餐改成晚上八点。"},
            {"role": "other", "text": "餐厅你来定吧。"},
        ]
        result = self.grounding(messages, subject={"text": "餐厅你来定吧。"})
        self.assertTrue(result["ambiguities"])
        self.assertIn(2, [row["index"] for row in result["relevant_messages"]])
        self.assertTrue(facts(result, slot="time", index=1))

    def test_reported_speech_does_not_assign_quoted_i_to_actual_sender(self):
        messages = [{"role": "other", "text": "他说：“我来选餐厅，你别管了。”"}]
        result = self.grounding(messages)
        assignments = facts(result, slot="responsibility", index=0)
        self.assertTrue(result["ambiguities"])
        self.assertTrue(assignments)
        self.assertTrue(all(row["actor"] == "unknown" and row["status"] == "unknown" for row in assignments))
        self.assertFalse(any(row["slot"] == "responsibility" for row in flatten_latest(result)))

    def test_unknown_speaker_is_never_promoted_to_self_other_or_both(self):
        messages = [{"role": "unknown", "text": "我来选餐厅，我不吃辣。"}]
        result = self.grounding(messages)
        self.assertTrue(result["constraints"])
        self.assertTrue(result["ambiguities"])
        self.assertTrue(all(row["actor"] == "unknown" for row in result["constraints"]))
        self.assertFalse(flatten_latest(result))

    def test_group_personal_preferences_and_tasks_do_not_bind_to_individuals(self):
        # Shared event schedules are separate from personal ownership. A group
        # does not tell us whose individual preference or work this is.
        for group in ("我们", "你们", "咱俩"):
            with self.subTest(group=group):
                messages = [{"role": "other", "text": f"{group}来选餐厅，{group}不吃辣。"}]
                result = self.grounding(messages)
                self.assertTrue(result["ambiguities"])
                self.assertTrue(result["constraints"])
                self.assertTrue(
                    all(row["actor"] == "unknown" and row["status"] == "unknown" for row in result["constraints"])
                )
                self.assertFalse(flatten_latest(result))

    def test_explicit_shared_group_event_time_and_date_can_be_both(self):
        messages = [{"role": "other", "text": "我们聚餐约定8月6日下午两点。"}]
        result = self.grounding(messages)
        schedule = [row for row in flatten_latest(result) if row["slot"] in {"time", "date"}]
        self.assertEqual({row["slot"] for row in schedule}, {"time", "date"})
        self.assertTrue(all(row["actor"] == "both" for row in schedule))
        self.assertTrue(any(row["slot"] == "time" and row["value"] == "14:00" for row in schedule))
        self.assertTrue(any(row["slot"] == "date" and row["value"] == "8月6日" for row in schedule))

    def test_multiple_undecided_dates_and_times_keep_sources_and_do_not_pick_one(self):
        messages = [{"role": "other", "text": "见面日期定8月6日还是8月7日，晚上八点或九点，还没决定。"}]
        result = self.grounding(messages)
        self.assert_original_sources(result, messages)
        dates = facts(result, slot="date", index=0)
        times = facts(result, slot="time", index=0)
        self.assertGreaterEqual(len({row["value"] for row in dates}), 2)
        self.assertGreaterEqual(len({row["value"] for row in times}), 2)
        self.assertTrue(result["ambiguities"])
        self.assertTrue(all(row["status"] == "unknown" for row in dates + times))
        self.assertFalse(any(row["slot"] in {"date", "time"} for row in flatten_latest(result)))

    def test_time_without_meridiem_does_not_invent_evening_or_morning(self):
        messages = [{"role": "other", "text": "碰头就8:30吧。"}]
        result = self.grounding(messages)
        row = result["latest"]["meeting"]["time"]["both"]
        self.assertEqual(row["parsed"]["hour"], 8)
        self.assertEqual(row["parsed"]["minute"], 30)
        self.assertEqual(row["parsed"]["period"], "unknown")
        self.assertNotEqual(row["value"], "20:30")

    def test_cue_limit_does_not_limit_fact_extraction_or_drop_selected_message(self):
        messages = original_scenarios()["later_time_correction"]
        result = self.grounding(messages, max_cues=2)
        self.assertLessEqual(len(result["relevant_messages"]), 2)
        self.assertIn(99, [row["index"] for row in result["relevant_messages"]])
        self.assertEqual(result["latest"]["dining"]["time"]["both"]["value"], "20:00")
        self.assertEqual(result["latest"]["dining"]["diet_spicy"]["other"]["value"], "avoid")

    def test_invalid_dates_keep_source_and_never_become_active_schedule(self):
        for invalid in ("2026-02-30", "2026-13-01", "2026-02-29", "8月40日"):
            with self.subTest(date=invalid):
                messages = [{"role": "other", "text": f"碰头日期定{invalid}。"}]
                result = self.grounding(messages)
                self.assert_original_sources(result, messages)
                dates = facts(result, slot="date")
                self.assertTrue(dates)
                self.assertTrue(all(row["status"] == "unknown" for row in dates))
                self.assertTrue(result["ambiguities"])
                self.assertFalse(any(row["slot"] == "date" for row in flatten_latest(result)))

    def test_leap_day_is_valid_only_when_explicit_year_supports_it(self):
        messages = [{"role": "other", "text": "碰头日期定2028-02-29。"}]
        result = self.grounding(messages)
        dates = [row for row in flatten_latest(result) if row["slot"] == "date"]
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0]["value"], "2028-02-29")
        self.assertEqual(dates[0]["status"], "active")

    def test_invalid_clock_times_keep_uncertainty_instead_of_becoming_schedule(self):
        for invalid in ("25:30", "23:80", "二十五点"):
            with self.subTest(time=invalid):
                messages = [{"role": "other", "text": f"碰头时间定{invalid}。"}]
                result = self.grounding(messages)
                self.assert_original_sources(result, messages)
                times = facts(result, slot="time")
                self.assertTrue(times)
                self.assertTrue(
                    all(row["status"] == "unknown" and row["parsed"]["certainty"] == "unknown" for row in times)
                )
                self.assertTrue(result["ambiguities"])
                self.assertFalse(any(row["slot"] == "time" for row in flatten_latest(result)))

    def test_index_text_conflict_is_unresolved_not_a_silent_wrong_cutoff(self):
        messages = [
            {"role": "other", "text": "这次聚餐先约晚上七点。"},
            {"role": "other", "text": "这次聚餐改成晚上八点。"},
            {"role": "other", "text": "这是未来发言，聚餐再改成晚上九点。"},
        ]
        result = self.grounding(messages, subject={"index": 0, "text": messages[1]["text"]})
        self.assertIsNone(result["subject_index"])
        self.assertFalse(result["stats"]["subject_matched_uniquely"])
        self.assertTrue(any(row.get("reason") == "subject_index_mismatch" for row in result["ambiguities"]))
        # Ambiguous input is retained for audit; it must not masquerade as a
        # verified selected-prefix boundary. Callers must resolve this first.
        self.assertEqual(result["stats"]["scoped_messages"], len(messages))
        self.assertEqual(result["stats"]["truncated_messages"], 0)

    def test_work_document_assignment_and_completion_keep_opposite_actors(self):
        messages = [
            {"role": "self", "text": "项目报告我来写，你负责检查。"},
            {"role": "other", "text": "报告检查我已经完成了，你来提交。"},
        ]
        result = self.grounding(messages)
        self.assertTrue(facts(result, slot="responsibility", index=0, actor="self"))
        self.assertTrue(facts(result, slot="responsibility", index=0, actor="other"))
        self.assertTrue(facts(result, slot="completion", index=1, actor="other", status="active"))
        self.assertTrue(facts(result, slot="responsibility", index=1, actor="self", status="active"))

    def test_unrelated_earlier_activity_does_not_displace_current_work_subject(self):
        messages = [
            {"role": "other", "text": "聚餐我不吃辣，周六晚上七点。"},
            {"role": "self", "text": "项目报告我来写，数据检查交给你。"},
            {"role": "other", "text": "项目报告的数据检查已经完成了，我把结果发给你。"},
        ]
        result = self.grounding(messages)
        self.assertIn("work", result["focus_topics"])
        selected = {row["index"] for row in result["relevant_messages"]}
        self.assertIn(1, selected)
        self.assertIn(2, selected)
        self.assertNotIn(0, selected)
        # The unrelated original fact remains auditable, while the attention
        # cues and relevant constraints follow the current work subject.
        self.assertTrue(facts(result, slot="diet_spicy", index=0))
        self.assertFalse(any(row["index"] == 0 for row in result["relevant_constraints"]))


if __name__ == "__main__":
    unittest.main()
