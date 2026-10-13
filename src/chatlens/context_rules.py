"""Explainable, offline grounding rules for already selected chat history.

This module does not locate WeChat messages, call a model, or choose actions.
Original text and original source indices are retained. ``rule_score`` only
orders attention cues; it is neither a confidence score nor a probability.
"""

from __future__ import annotations

import re
import unicodedata
import calendar
import datetime
from collections import defaultdict

VERSION = "context-rules-1.1"

REFERENCE_WINDOW_SIZE = 12
# Detect a need to resolve omitted objects; this does not resolve what was
# added, whose result it was, or whether a conditional ranking is already true.
_REFERENCE = re.compile(
    r"这个|那个|这样|那样|这些|那些|这事|那事|那就|那也|那还|"
    r"加上|算上|再加|计入|算进去|不然|否则|"
    r"第[一二三四五六七八九十\d]+|"
    r"(?:^|[，,。；;！？!?])\s*[这那](?:[，,。；;！？!?]|$)"
)

_TOPICS = {
    "paper": re.compile(r"论文|绪论|稿件|投稿|审稿|文献|摘要|研究问题|章节"),
    "badminton": re.compile(r"羽毛球|球馆|球场|球拍|拍子|打球"),
    "dining": re.compile(
        r"餐厅|饭店|聚餐|吃饭|吃什么|饭局|晚饭|午饭|早饭|辣椒|吃辣|忌口|过敏|吃素|咖啡|奶茶|汤面|订位"
    ),
    "interview": re.compile(r"面试|实习|应聘|招聘|录取"),
    "meeting": re.compile(r"开会|会议|碰头|见面|约会|集合"),
    "travel": re.compile(r"出发|车站|车票|航班|机场|赶车|取件|取快递"),
    "work": re.compile(r"报告|项目|需求|表格|文件|演示|汇报|提交|交稿"),
}
_TIME = re.compile(
    r"(?P<period>凌晨|清晨|早上|上午|中午|下午|傍晚|晚上|夜里)?\s*(?P<hour>[零〇一二两三四五六七八九十\d]{1,3})(?:(?:点|时)(?P<tail>半|[零〇一二两三四五六七八九十\d]{1,3}分?)?|:(?P<minute>\d{1,2}))(?!\d)"
)
_DATE = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|[零〇一二两三四五六七八九十\d]{1,3}月[零〇一二两三四五六七八九十\d]{1,3}[日号]|(?:下|本|这)?(?:周|星期)[一二三四五六日天]|今天|明天|后天"
)
_QUOTED = re.compile(r'[“”"「」『』]|(?:他说|她说|他们说|她们说|你说|我说|朋友说|听说|据说|转述|引用|写着)')
_GROUP = re.compile(r"我们|你们|他们|她们|咱们|大家|咱俩|我俩|你俩")
_PRONOUN = re.compile(r"我(?!们|俩)|你(?!们|俩)")
_CHANGE = re.compile(
    r"更正|改成|改到|改为|换成|换到|重新|取消|作废|以[^，,。；;！？!?\n]{1,60}为准|还是我|现在定|确定|就按|照[^，,。；;！？!?\n]{1,60}来"
)
_NEGATIVE_TIME = re.compile(r"赶不到|到不了|来不了|不行|不方便|不能|不要|别|不是|不约|取消|作废")
_DONE = re.compile(r"改完|写完|做完|完成|交完|弄完|整理完|提交了|交了|搞定")
_PENDING = re.compile(r"还没|没有|没(?:改完|写完|做完|完成|交完|弄完|整理完|提交|搞定)|尚未|未完成")
_ASSIGN = re.compile(
    r"来(?:选|定|找|订|安排|负责|做|改|写|看|检查|提交)|负责|帮你(?:看看|看|检查|改|选|找|收)|替你(?:看|检查|改|选|找)|(?:要|得|需要)(?:重写|改写|修改|检查|整理)"
)
_LATE = re.compile(r"迟到")
_PENALTY = re.compile(r"罚(?:你|我)?(?:跑|站|做)(?:[零〇一二两三四五六七八九十\d]+)?(?:圈|轮|次|分钟|米)?")
_UNCERTAIN = re.compile(
    r"还没决定|没决定|没定|待定|未定|不确定|不是不|并非不|不一定|好像|可能|也许|大概|是不是|是否|能不能|要不要|(?:吗|么)[？?]?$"
)
_STOP_BIGRAMS = {
    "今天",
    "明天",
    "刚才",
    "之前",
    "现在",
    "还是",
    "可以",
    "这个",
    "那个",
    "一下",
    "没有",
    "就是",
    "好了",
    "那就",
    "的话",
    "我们",
    "你们",
    "对方",
    "自己",
    "然后",
    "但是",
    "时候",
    "时间",
    "完成",
}


def normalize_text(text):
    """NFKC and whitespace normalization used for matching, never quotation."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def _number(value):
    value = value.replace("两", "二").replace("〇", "零")
    if value.isdigit():
        return int(value)
    digits = {c: i for i, c in enumerate("零一二三四五六七八九")}
    if value in digits:
        return digits[value]
    if value.count("十") == 1:
        left, right = value.split("十")
        if (not left or left in digits) and (not right or right in digits):
            return digits.get(left, 1) * 10 + digits.get(right, 0)
    return None


def _topics(text):
    return [name for name, regex in _TOPICS.items() if regex.search(text)]


def _grams(text):
    pieces = re.findall(r"[\u3400-\u9fff]+|[a-zA-Z]{2,}", text.lower())
    result = set()
    for piece in pieces:
        if re.fullmatch(r"[a-z]+", piece):
            result.add(piece)
        else:
            result.update(piece[i : i + 2] for i in range(len(piece) - 1))
    return result - _STOP_BIGRAMS


def _actor(clause, role, position=None):
    """Only explicit singular 我/你, with nearest preceding subject preferred."""
    if role not in {"self", "other"} or _QUOTED.search(clause) or _GROUP.search(clause):
        return "unknown"
    matches = list(_PRONOUN.finditer(clause))
    if not matches:
        return "unknown"
    if position is not None:
        preceding = [m for m in matches if m.start() <= position]
        if preceding:
            pronoun = preceding[-1].group()
        elif len({m.group() for m in matches}) == 1:
            pronoun = matches[0].group()
        else:
            return "unknown"
    elif len({m.group() for m in matches}) == 1:
        pronoun = matches[0].group()
    else:
        return "unknown"
    return role if pronoun == "我" else ("other" if role == "self" else "self")


def _time_value(match):
    hour = _number(match.group("hour"))
    tail = match.group("tail") or ""
    minute = 30 if tail == "半" else _number((match.group("minute") or tail.rstrip("分") or "0"))
    period = match.group("period") or "unknown"
    if hour is None or minute is None or not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return match.group().strip(), {"hour": hour, "minute": minute, "period": period, "certainty": "unknown"}
    canonical_hour = hour
    certainty = "known"
    if period in {"下午", "傍晚", "晚上", "夜里"}:
        if 1 <= hour < 12:
            canonical_hour += 12
        elif period in {"晚上", "夜里"} and hour == 12:
            canonical_hour = 0
    elif period == "中午":
        if hour == 12:
            canonical_hour = 12
        elif 1 <= hour <= 2:
            canonical_hour += 12
        else:
            certainty = "unknown"
    elif period in {"凌晨", "清晨", "早上", "上午"}:
        if hour == 12:
            certainty = "unknown"
    elif hour <= 12:
        # The lexical clock reading is known, the AM/PM interpretation is not.
        certainty = "period_unknown"
    parsed = {
        "hour": canonical_hour if certainty == "known" else hour,
        "minute": minute,
        "period": period,
        "certainty": certainty,
    }
    value = f"{canonical_hour:02d}:{minute:02d}" if certainty == "known" else match.group().strip()
    return value, parsed


def _date_value(surface):
    full = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", surface)
    partial = re.fullmatch(
        r"([零〇一二两三四五六七八九十\d]{1,3})月([零〇一二两三四五六七八九十\d]{1,3})[日号]", surface
    )
    if full:
        year, month, day = map(int, full.groups())
        try:
            value = datetime.date(year, month, day).isoformat()
        except ValueError:
            return surface, {"year": year, "month": month, "day": day, "certainty": "unknown", "valid": False}
        return value, {"year": year, "month": month, "day": day, "certainty": "known", "valid": True}
    if partial:
        month, day = (_number(v) for v in partial.groups())
        # The year is absent: February 29 is possible, February 30 never is.
        valid = (
            month is not None
            and day is not None
            and 1 <= month <= 12
            and 1 <= day <= calendar.monthrange(2000, month)[1]
        )
        return surface, {
            "year": None,
            "month": month,
            "day": day,
            "certainty": "year_unknown" if valid else "unknown",
            "valid": bool(valid),
        }
    return surface, {"certainty": "relative", "valid": True}


def _negative_time(clause, match, time_count=1):
    prefix, suffix = clause[: match.start()], clause[match.end() :]
    if re.search(r"(?:不是|不要|别|取消|作废|不约)\s*$", prefix):
        return True
    if re.match(r"\s*(?:我|你)?\s*(?:赶不到|到不了|来不了|不行|不方便|不能)", suffix):
        return True
    # A single time can use the whole short clause. With multiple times the
    # negation stays adjacent to its own candidate, not the whole sentence.
    return time_count == 1 and bool(_NEGATIVE_TIME.search(clause))


def extract_grounding(messages, subject=None, max_cues=8):
    """Extract original attention cues and auditable constraints.

    ``latest[topic][slot][actor]`` contains only active positive evidence. A
    repeated statement may update its source; conflicting values require an
    explicit correction/confirmation, otherwise both remain auditable and the
    newer conflict is unknown. Actor-specific preferences never overwrite one
    another. An unqualified clock time retains an unknown day period.
    """
    if not isinstance(messages, (list, tuple)):
        raise TypeError("messages must be a chronological list")
    if not isinstance(max_cues, int) or isinstance(max_cues, bool):
        raise TypeError("max_cues must be an integer")
    max_cues = min(100, max(0, max_cues))
    ambiguities = []
    rows = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("text"), str):
            ambiguities.append({"index": index, "reason": "invalid_message"})
            continue
        original = message["text"]
        role = message.get("role", "unknown")
        if role not in {"self", "other"}:
            role = "unknown"
        normalized = normalize_text(original)
        if normalized:
            rows.append({"index": index, "role": role, "text": original, "normalized": normalized})
    subject = subject if isinstance(subject, dict) else {}
    subject_text = normalize_text(subject.get("text", ""))
    anchor = None
    explicit_index = subject.get("index")
    if explicit_index is not None:
        found = (
            next((r for r in rows if r["index"] == explicit_index), None)
            if isinstance(explicit_index, int) and not isinstance(explicit_index, bool)
            else None
        )
        if found is not None and (not subject_text or found["normalized"] == subject_text):
            anchor = found["index"]
            subject_text = found["normalized"]
        else:
            ambiguities.append(
                {
                    "reason": "subject_index_mismatch",
                    "index": explicit_index if isinstance(explicit_index, int) else None,
                }
            )
    elif subject_text:
        found = [r for r in rows if r["normalized"] == subject_text]
        if len(found) == 1:
            anchor = found[0]["index"]
        else:
            ambiguities.append({"reason": "subject_not_unique", "matches": len(found)})
    elif rows:
        anchor = rows[-1]["index"]
        subject_text = rows[-1]["normalized"]
    scoped = [r for r in rows if anchor is None or r["index"] <= anchor]
    near_rows = scoped[-REFERENCE_WINDOW_SIZE:] if anchor is not None else []
    near_context = [{"index": r["index"], "role": r["role"], "text": r["text"]} for r in near_rows]
    requires_reference_resolution = bool(_REFERENCE.search(subject_text))
    subject_topics = _topics(subject_text)
    focus_topics = list(subject_topics)
    target_grams = _grams(subject_text)
    frequencies = defaultdict(int)
    for row in scoped:
        for gram in _grams(row["normalized"]):
            frequencies[gram] += 1
    rare_grams = {g for g in target_grams if frequencies[g] <= max(4, len(scoped) // 5)}
    if not focus_topics and requires_reference_resolution:
        # Ellipsis is normally attached to adjacent turns, even when the
        # omitted object's name shares no characters with the selected reply.
        # A distant lexical overlap must not substitute another event.
        for row in reversed(near_rows[:-1]):
            if _topics(row["normalized"]):
                focus_topics = _topics(row["normalized"])
                break
    if not focus_topics and not requires_reference_resolution:
        # A topic-less reference needs an actual shared keyword, not an arbitrary
        # old date or preference. If no such evidence exists, use recent raw cues.
        for row in reversed(scoped[:-1]):
            overlap = rare_grams & _grams(row["normalized"])
            if len(overlap) >= 2 or any(len(g) > 2 for g in overlap):
                focus_topics = _topics(row["normalized"])
                if focus_topics:
                    break
    if not focus_topics:
        ambiguities.append({"reason": "focus_topic_unknown", "index": anchor})
    constraints = []
    latest = {}
    last_topic = None
    last_topic_index = -100

    def ambiguity(row, reason, slot=None):
        item = {"index": row["index"], "role": row["role"], "text": row["text"], "reason": reason}
        if slot:
            item["slot"] = slot
        if item not in ambiguities:
            ambiguities.append(item)

    def add(row, topic, slot, value, actor, status="active", parsed=None, reason="explicit_rule", correction=False):
        constraint = {
            "index": row["index"],
            "role": row["role"],
            "text": row["text"],
            "topic": topic or "general",
            "slot": slot,
            "value": value,
            "actor": actor,
            "status": status,
            "reasons": [reason],
        }
        if parsed is not None:
            constraint["parsed"] = parsed
        if actor == "unknown" and status == "active":
            constraint["status"] = "unknown"
            ambiguity(row, "actor_unknown", slot)
        if constraint["status"] == "active" and constraint["topic"] != "general":
            bucket = latest.setdefault(constraint["topic"], {}).setdefault(slot, {})
            previous = bucket.get(actor)
            if previous and previous["value"] != value:
                if correction:
                    previous["status"] = "superseded"
                    previous["reasons"].append("explicit_later_correction")
                else:
                    constraint["status"] = "unknown"
                    ambiguity(row, "conflicting_values_without_correction", slot)
            if constraint["status"] == "active":
                bucket[actor] = constraint
        constraints.append(constraint)
        return constraint

    for row in scoped:
        text, role = row["normalized"], row["role"]
        row_topics = _topics(text)
        if row_topics:
            last_topic = row_topics[0]
            last_topic_index = row["index"]
        # Inherit only across a short, explicit same-topic exchange. The global
        # current focus is not enough to invent a topic for unrelated old text.
        topic = row_topics[0] if row_topics else None
        quoted = bool(_QUOTED.search(text))
        grouped = bool(_GROUP.search(text))
        if quoted:
            ambiguity(row, "quoted_or_reported_speech")
        if grouped:
            ambiguity(row, "group_actor_ambiguous")
        if role == "unknown":
            ambiguity(row, "speaker_role_unknown")
        clauses = [c.strip() for c in re.split(r"[，,。；;！？!?\n]+", text) if c.strip()]
        row_correction = bool(_CHANGE.search(text))
        late_actor = "unknown"
        for clause in clauses:
            local_topics = _topics(clause)
            local_topic = local_topics[0] if local_topics else topic
            # Borrow only from an adjacent explicit exchange, never from an
            # unrelated far-away date merely because the selected topic exists.
            if (
                local_topic is None
                and row["index"] - last_topic_index <= 2
                and re.search(r"时间|改成|改到|就按|约|见|碰头", clause)
            ):
                local_topic = last_topic
            ambiguous = quoted or grouped or role == "unknown"
            clause_uncertain = bool(_UNCERTAIN.search(clause))
            shared_event = bool(local_topic and re.search(r"聚餐|餐厅|饭局|约|见面|碰头|集合|时间|改成|改到", text))
            shared_actor_allowed = shared_event and not quoted and role in {"self", "other"}
            calendar_ambiguous = quoted or role == "unknown" or (grouped and not shared_actor_allowed)
            time_matches = list(_TIME.finditer(clause))
            unresolved_times = len(time_matches) > 1 and not (_NEGATIVE_TIME.search(clause) or _CHANGE.search(clause))
            if unresolved_times:
                ambiguity(row, "multiple_unresolved_times", "time")
            for match in time_matches:
                value, parsed = _time_value(match)
                if parsed["certainty"] == "unknown":
                    ambiguity(row, "invalid_or_unresolved_clock_time", "time")
                actor = _actor(clause, role, match.start())
                if actor == "unknown" and shared_actor_allowed:
                    actor = "both"
                status = (
                    "unknown"
                    if calendar_ambiguous or clause_uncertain or unresolved_times or parsed["certainty"] == "unknown"
                    else ("negated" if _negative_time(clause, match, len(time_matches)) else "active")
                )
                if not local_topic:
                    status = "unknown"
                add(
                    row,
                    local_topic,
                    "time",
                    value,
                    actor,
                    status,
                    parsed,
                    "explicit_clock_time",
                    row_correction or bool(re.search(r"就|确定|见|碰头|照|按", clause)),
                )
            date_matches = list(_DATE.finditer(clause))
            unresolved_dates = len(date_matches) > 1 and not (_NEGATIVE_TIME.search(clause) or _CHANGE.search(clause))
            if unresolved_dates:
                ambiguity(row, "multiple_unresolved_dates", "date")
            for match in date_matches:
                actor = _actor(clause, role, match.start())
                if actor == "unknown" and shared_actor_allowed:
                    actor = "both"
                value, parsed = _date_value(match.group())
                status = (
                    "unknown"
                    if calendar_ambiguous
                    or clause_uncertain
                    or unresolved_dates
                    or not local_topic
                    or not parsed["valid"]
                    else ("negated" if _NEGATIVE_TIME.search(clause) else "active")
                )
                if not parsed["valid"]:
                    ambiguity(row, "invalid_calendar_date", "date")
                add(
                    row,
                    local_topic,
                    "date",
                    value,
                    actor,
                    status,
                    parsed=parsed,
                    reason="explicit_date",
                    correction=row_correction,
                )
            # Personal preference and food preparation constraints are different
            # slots. Singular ownership must be explicit for personal preference.
            diet_patterns = (
                ("diet_spicy", "avoid", r"(?:不|不能|别|不要)(?:太)?吃辣|忌辣|不吃辣椒"),
                ("diet_spicy", "prefer", r"爱吃辣|喜欢吃辣|喜欢辣|能吃辣"),
                ("diet_spicy", "avoid", r"不放辣椒|不要辣椒|不加辣|清淡一点"),
                ("diet_vegetarian", "vegetarian", r"吃素|素食"),
                ("diet_drink", "avoid_coffee", r"不喝咖啡|不能喝咖啡"),
                ("diet_allergy", "allergy", r"过敏"),
            )
            for slot, value, regex in diet_patterns:
                match = re.search(regex, clause)
                if not match:
                    continue
                actor = _actor(clause, role, match.start())
                status = "unknown" if ambiguous or clause_uncertain else "active"
                if slot == "diet_allergy":
                    value = match.group()
                add(row, "dining", slot, value, actor, status, reason="explicit_diet_phrase", correction=row_correction)
            assign = _ASSIGN.search(clause)
            if assign and local_topic:
                actor = _actor(clause, role, assign.start())
                add(
                    row,
                    local_topic,
                    "responsibility",
                    "assigned",
                    actor,
                    "unknown" if ambiguous or clause_uncertain else "active",
                    reason="explicit_singular_assignment",
                    correction=row_correction,
                )
            completion = _DONE.search(clause)
            pending = _PENDING.search(clause)
            if (completion or pending) and local_topic:
                position = (pending or completion).start()
                actor = _actor(clause, role, position)
                # A following singular 我 in the same message can name the owner
                # (e.g. "绪论终于改完了，算被我啃下来了"). Mixed pronouns cannot.
                if actor == "unknown" and not ambiguous:
                    actor = _actor(text, role)
                conditional = bool(
                    re.search(
                        r"如果|要是|等.{1,120}(?:完|完成)|(?:改完|写完|做完|完成).{0,120}(?:可以|再|就|发给)", clause
                    )
                )
                confirmed = bool(
                    pending or re.search(r"已经|终于|[完了]了|搞定|完成了|提交了|交了|写完了|改完了|做完了", clause)
                )
                status = "unknown" if ambiguous or clause_uncertain or conditional or not confirmed else "active"
                if conditional:
                    ambiguity(row, "conditional_completion", "completion")
                add(
                    row,
                    local_topic,
                    "completion",
                    "pending" if pending else "completed",
                    actor,
                    status,
                    reason="explicit_completion_phrase",
                    correction=bool(confirmed),
                )
            late = _LATE.search(clause)
            if late and (local_topic == "badminton" or "badminton" in row_topics):
                late_actor = _actor(clause, role, late.start())
                conditional = bool(re.search(r"再迟到|如果|要是|别迟到|不要迟到", clause))
                add(
                    row,
                    "badminton",
                    "late",
                    "conditional_late" if conditional else "late",
                    late_actor,
                    "unknown" if ambiguous else "active",
                    reason="explicit_late_subject",
                )
            penalty = _PENALTY.search(clause)
            if penalty and (local_topic == "badminton" or "badminton" in row_topics):
                direct = re.search(r"罚(?P<pronoun>你|我)", clause)
                actor = _actor(direct.group(), role) if direct else late_actor
                if actor == "unknown" and not ambiguous:
                    actor = _actor(clause, role, penalty.start())
                add(
                    row,
                    "badminton",
                    "penalty",
                    penalty.group(),
                    actor,
                    "unknown" if ambiguous else "active",
                    parsed={"conditional": bool(late_actor != "unknown")},
                    reason="penalty_of_explicit_late_subject",
                )

    # Attention cue ranking. Repeated boilerplate must not fill the cue budget.
    scored = []
    constraint_indices = defaultdict(list)
    for constraint in constraints:
        constraint_indices[constraint["index"]].append(constraint)
    last_index = scoped[-1]["index"] if scoped else -1
    for row in scoped:
        row_topics = _topics(row["normalized"])
        overlap = sorted(rare_grams & _grams(row["normalized"]))
        topic_overlap = sorted(set(row_topics) & set(focus_topics))
        reasons = []
        score = 0
        if row["index"] == anchor:
            score += 100
            reasons.append("selected_message")
        if topic_overlap:
            score += 12
            reasons.append("focus_topic:" + ",".join(topic_overlap))
        if overlap:
            score += min(15, len(overlap) * 3)
            reasons.append("keyword_overlap:" + ",".join(overlap[:5]))
        relevant_facts = [c for c in constraint_indices[row["index"]] if c["topic"] in focus_topics]
        if relevant_facts:
            score += 8
            reasons.append("grounding_slots:" + ",".join(sorted({c["slot"] for c in relevant_facts})))
        if row["index"] >= last_index - 5:
            score += 1
            reasons.append("recent_raw_context")
        if score and (not focus_topics or topic_overlap or overlap or relevant_facts or row["index"] == anchor):
            scored.append(
                {
                    "index": row["index"],
                    "role": row["role"],
                    "text": row["text"],
                    "reasons": reasons,
                    "rule_score": score,
                    "_normalized": row["normalized"],
                }
            )
    ranked = sorted(scored, key=lambda r: (-r["rule_score"], -r["index"]))
    chosen, seen = [], set()
    for row in ranked:
        # Numeric counters in repeated background are not different facts.
        signature = re.sub(r"\d+", "#", row["_normalized"])
        if signature in seen and row["index"] != anchor:
            continue
        seen.add(signature)
        chosen.append(row)
        if len(chosen) >= max_cues:
            break
    if max_cues == 0:
        chosen = []
    relevant_messages = [
        {k: v for k, v in row.items() if k != "_normalized"} for row in sorted(chosen, key=lambda r: r["index"])
    ]
    cue_indices = {r["index"] for r in relevant_messages}
    relevant_constraints = [c for c in constraints if c["index"] in cue_indices and c["topic"] in focus_topics]
    return {
        "version": VERSION,
        "subject_index": anchor,
        "focus_topics": focus_topics,
        "relevant_messages": relevant_messages,
        "near_context": near_context,
        "reference_window_indices": [row["index"] for row in near_context],
        "requires_reference_resolution": requires_reference_resolution,
        "constraints": constraints,
        "relevant_constraints": relevant_constraints,
        "latest": latest,
        "ambiguities": ambiguities,
        "stats": {
            "input_messages": len(messages),
            "usable_messages": len(rows),
            "scoped_messages": len(scoped),
            "subject_index": anchor,
            "subject_matched_uniquely": anchor is not None,
            "truncated_messages": len(rows) - len(scoped),
            "max_cues": max_cues,
            "cues_returned": len(relevant_messages),
            "constraints_returned": len(constraints),
            "normalized_messages": sum(r["text"] != r["normalized"] for r in scoped),
            "rare_target_bigrams": len(rare_grams),
            "rule_score_is_probability": False,
        },
    }
