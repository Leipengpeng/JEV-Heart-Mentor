"""Local authored style examples; never treat fiction as conversation memory.

Reference retrieval changes writing examples, never the full JEV action bank.
No network calls or character-card instructions are executed at runtime.
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "knowledge/galgame"


def load_bank(path=None):
    value = json.loads((Path(path) if path else ROOT / "interaction_bank.json").read_text())
    rows = value["actions"]
    if not 1 <= len(rows) <= 64:
        raise ValueError("互动知识库必须包含1到64项完整动作")
    ids, names = set(), set()
    for row in rows:
        if row["id"] in ids or row["title"] in names:
            raise ValueError("互动知识库动作身份重复")
        ids.add(row["id"])
        names.add(row["title"])
        if not re.fullmatch(r"[\u3400-\u9fff]{2}", row["title"]):
            raise ValueError("互动知识库名称必须为两个汉字")
        if row["kind"] not in {"reply", "clarify", "pause", "recall"} or not row["jev_meaning"].strip():
            raise ValueError("互动知识库动作格式无效")
        if len(row["examples"]) < 3:
            raise ValueError("每种互动需要至少三组原创例子")
        patterns = row.get("style_patterns", {})
        if any(
            not isinstance(patterns.get(field), list)
            or len(patterns[field]) < 2
            or any(not isinstance(x, str) or not x.strip() for x in patterns[field])
            for field in ("analysis", "detail", "reply")
        ):
            raise ValueError("每种互动需要分析、描述与回复的原创句型")
        for example in row["examples"]:
            for key in ("trigger", "context", "analysis", "detail", "reply"):
                if not isinstance(example.get(key), str):
                    raise ValueError("互动示例格式无效")
            if example.get("kind") != "original":
                raise ValueError("运行时示例只允许明确标注的原创文本")
    return value


def _grams(text):
    compact = re.sub(r"[^\u3400-\u9fffA-Za-z0-9]", "", str(text).lower())
    return set(compact[i : i + 2] for i in range(max(0, len(compact) - 1)))


def _similarity(query, example):
    tokens = _grams(example["trigger"] + " " + example["context"])
    return len(query & tokens) / max(1, len(tokens) ** 0.5)


def analysis_examples(bank, subject, conversation="", limit=3):
    """Return only scene/analysis fields, never action IDs, names or drafts."""
    query = _grams(subject)
    examples = [(row, e) for row in bank["actions"] for e in row["examples"]]
    examples.sort(key=lambda item: _similarity(query, item[1]), reverse=True)
    selected, seen = [], set()
    for row, example in examples:
        if _similarity(query, example) == 0:
            continue
        marker = row["id"]
        if marker in seen:
            continue
        seen.add(marker)
        selected.append({"kind": "original_sentence_pattern", "analysis_patterns": row["style_patterns"]["analysis"]})
        if len(selected) >= limit:
            break
    return selected


def writing_examples(bank, selected_actions, subject, conversation="", per_action=2):
    """Lookup examples only after JEV chooses; this function never scores IDs."""
    lookup = {row["id"]: row for row in bank["actions"]}
    query = _grams(subject)
    result = []
    for selected in selected_actions:
        row = lookup[selected["id"]]
        if selected["action"] != row["title"]:
            raise ValueError("文案参考与JEV已选动作不一致")
        examples = sorted(
            (e for e in row["examples"] if _similarity(query, e) > 0), key=lambda e: _similarity(query, e), reverse=True
        )[:per_action]
        # The scenes help lexical retrieval, but their fictional nouns and
        # events never enter the local model's prompt.
        result.append(
            {
                "action_id": row["id"],
                "kind": "original_sentence_pattern",
                "detail_patterns": row["style_patterns"]["detail"][:per_action] if examples else [],
                "reply_patterns": row["style_patterns"]["reply"][:per_action] if examples else [],
            }
        )
    return result


ECHO_PATTERNS = (
    r"(?:俏皮|活泼|人性化|幽默)(?:地|的方式|的语气|风格)",
    r"(?:采用|运用|使用|保持|突出|体现).{0,10}(?:风格|互动感|人性化)",
    r"接住对方(?:的)?(?:进度|情绪|处境|期待|话题)",
    r"推进(?:对话|互动|交流|事项|下一步|当前事项)",
    r"自然(?:地)?回应对方|给.{0,5}有温度的回应",
    r"(?:用|通过).{0,18}(?:比喻|猜测|语气).{0,12}(?:延续|承接|表现|体现)",
    r"(?:用|通过).{0,10}(?:语调|语气).{0,10}(?:回应|回答)",
    r"前文第?\s*\d+\s*条(?:消息|原文)",
)


def instruction_echo(text):
    return next((pattern for pattern in ECHO_PATTERNS if re.search(pattern, text)), None)


def validate_display_text(text, source_text=""):
    for index, pattern in enumerate(ECHO_PATTERNS):
        match = re.search(pattern, text)
        if not match:
            continue
        # A real discussion of a poster or literary style may contain the same
        # descriptor. Preserve literal topic words; workflow slogans still fail.
        if index == 0 and match.group() in source_text:
            continue
        raise ValueError("展示文案复述了写作指令；请写这次原文中具体的人、事、物或一句可直接选择的话")
    if re.search(r"[（(]\s*(?:self|other)\s*[)）]|(?:self|other)\s*[:：]", text):
        raise ValueError("不要展示内部角色标签；用我、你、对方表达")
    if re.search(r"(?:并|而|但|或者|因为|所以|然后)\s*[，、]?\s*$", text):
        raise ValueError("句子尚未说完；补全意思再结束")
    for slot in re.findall(r"【[^】]{1,12}】", text):
        if slot not in source_text:
            raise ValueError("把句型变量换成本次原文的具体内容，不显示占位符")
    if text.count("“") != text.count("”"):
        raise ValueError("引语尚未说完；请写完整句子")


def knowledge_hash(bank, analysis_prompt, detail_prompt):
    authored = json.dumps(bank, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256((authored + analysis_prompt + detail_prompt).encode()).hexdigest()
