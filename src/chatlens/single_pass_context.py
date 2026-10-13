"""Token-exact raw evidence packing; never calls a language model.

JEV has already searched the source history. This module preserves the selected
anchor, prioritizes its contiguous neighbors, then fits selected original quotes.
Source scan coverage is distinct from the smaller set actually sent to MiniCPM.
"""

import copy
from .context_budget import prompt_tokens, ContextBudgetError
from .history_compressor import format_rows

PROMPT_TOKEN_LIMIT = 3000
THINKING_BUDGET = 128
INITIAL_INPUT_LIMIT = PROMPT_TOKEN_LIMIT - THINKING_BUDGET - 32


def prepare_single_payload(tokenizer, header, payload, selected_actions, emotion):
    rows = payload["_context_rows"]
    subject = payload["analysis_subject"]
    anchor = rows[-1]
    if anchor["role"] != "other" or anchor["text"] != subject["text"]:
        raise ValueError("单次生成的锚点与所选对方原文不一致")
    selected = payload.get("jev_selected_rows", [])
    originals = {row["source_index"]: row for row in rows}
    for row in selected:
        original = originals.get(row.get("source_index"))
        if original is None or row.get("role") != original["role"] or row.get("text") != original["text"]:
            raise ValueError("检索引文不属于当前锚点之前的原始对话")
    base = {
        "conversation_message_count": len(rows),
        "current_other_message": anchor["text"],
        "selected_source_index": anchor["source_index"],
        "evidence_to_copy": anchor["text"][:60],
        "reply_perspective": {"speaker": "self", "recipient": "other"},
        "reference_resolution": {
            "requires_resolution": bool(payload.get("requires_reference_resolution")),
            "weak_topic_hint": payload.get("focus_topics", []),
        },
        "selected_actions": [
            {"id": r["id"], "action": r["action"], "meaning": r.get("strategy_meaning", r["detail"])}
            for r in selected_actions
        ],
        "jev_emotion_order": [r["label"] for r in emotion.get("ranked", [])[:3]],
        "conversation": format_rows([anchor]),
        "matched_original_messages": [],
    }
    names = {"self": "我", "other": "对方", "unknown": "发言方待核对"}

    def quote(row):
        fragment = row.get("retrieval_fragment") or {}
        text = fragment.get("text", row["text"])
        if not isinstance(text, str) or text not in row["text"]:
            raise ValueError("JEV选中的引文不在对应原文中")
        value = {"index": row["source_index"], "speaker": names.get(row["role"], "发言方待核对"), "text": text}
        if text != row["text"]:
            value["is_partial_quote"] = True
        return value

    def count(body):
        return prompt_tokens(tokenizer, header, body)

    anchor_tokens = count(base)
    if anchor_tokens > INITIAL_INPUT_LIMIT:
        raise ContextBudgetError("mandatory_anchor_too_large", anchor_tokens, INITIAL_INPUT_LIMIT)

    # Reserve room for older matches, rather than allowing a verbose recent
    # suffix to consume the entire input. Never split a neighboring raw message.
    reserve = min(650, max(0, INITIAL_INPUT_LIMIT - anchor_tokens - 200)) if selected else 0
    recent = [anchor]
    for row in reversed(rows[-12:-1]):
        candidate = {**base, "conversation": format_rows([row] + recent)}
        if count(candidate) > INITIAL_INPUT_LIMIT - reserve:
            break
        recent.insert(0, row)
        base = candidate

    retained = {row["source_index"] for row in recent}
    matches = []
    skipped = []
    priority_indices = []
    # Search winners retain relevance order. Add whole original matches first;
    # an explicitly split candidate can be represented by its exact raw fragment.
    for row in selected:
        index = row["source_index"]
        if index in retained:
            continue
        item = quote(row)
        candidate = {**base, "matched_original_messages": matches + [item]}
        if count(candidate) > INITIAL_INPUT_LIMIT:
            skipped.append(index)
            continue
        matches.append(item)
        priority_indices.append(index)
        retained.add(index)
        base = candidate

    # With remaining room include a selected match's immediate neighbors. This
    # lets a number or a bare acknowledgement retain its local referent.
    lookup = {row["source_index"]: row for row in rows}
    neighbor_indices = []
    for index in priority_indices:
        for neighbor in (index - 1, index + 1):
            row = lookup.get(neighbor)
            if row is None or neighbor in retained:
                continue
            item = quote(row)
            candidate = {**base, "matched_original_messages": matches + [item]}
            if count(candidate) > INITIAL_INPUT_LIMIT:
                continue
            matches.append(item)
            neighbor_indices.append(neighbor)
            retained.add(neighbor)
            base = candidate
    base["matched_original_messages"] = sorted(matches, key=lambda row: row["index"])
    # If selected matches were short, spend any unclaimed reserve on additional
    # contiguous neighbors of the anchor (up to the protected 12-row window).
    while len(recent) < min(12, len(rows)):
        row = rows[len(rows) - len(recent) - 1]
        candidate = {**base, "conversation": format_rows([row] + recent)}
        if count(candidate) > INITIAL_INPUT_LIMIT:
            break
        recent.insert(0, row)
        retained.add(row["source_index"])
        base = candidate
    scope = {
        "input_form": "jev_selected_original_quotes+contiguous_recent_raw",
        "source_messages_count": len(rows),
        "all_sources_covered": bool(payload.get("jev_search_metrics", {}).get("all_sources_covered")),
        "source_coverage_method": "jev_slice_search_and_raw_state",
        "all_original_messages_sent_to_llm": len(retained) == len(rows)
        and not any(x.get("is_partial_quote") for x in matches),
        "raw_source_indices": sorted(retained),
        "raw_messages_count": len(retained),
        "recent_raw_count": len(recent),
        "recent_window_count": min(12, len(rows)),
        "near_context_complete": len(recent) == min(12, len(rows)),
        "retrieved_raw_count": len(priority_indices),
        "retrieved_neighbor_count": len(neighbor_indices),
        "retrieved_rows_omitted_for_budget": skipped,
        "partial_quote_count": sum(bool(x.get("is_partial_quote")) for x in matches),
        "initial_input_tokens": count(base),
        "initial_input_limit": INITIAL_INPUT_LIMIT,
        "thinking_reserve_tokens": THINKING_BUDGET,
        "prompt_token_limit": PROMPT_TOKEN_LIMIT,
        "llm_compression_calls": 0,
        "future_messages_excluded": True,
    }
    if scope["initial_input_tokens"] > INITIAL_INPUT_LIMIT:
        raise ContextBudgetError("selected_context_too_large", scope["initial_input_tokens"], INITIAL_INPUT_LIMIT)
    return copy.deepcopy(base), scope
