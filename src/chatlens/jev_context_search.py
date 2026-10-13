"""Source-preserving, batched native-choice retrieval; no generative model calls.

The selected message and a contiguous fitting recent suffix share one native
state. Earlier originals are complete candidates, split only when necessary.
Each group is compared locally; finalists are compared together rather than
sorting incomparable probabilities from different candidate sets.
"""

import json
import math
import re
import time

VERSION = "jev-original-choice-search-1.0"
MAX_SUFFIX = 1024
QUERY = (
    "找出最能解释当前对方原话的前文：补全它省略的具体事件、对象、数量、比较条件、"
    "指代或最新更正。优先同一件事的直接原文；只有关键词相同但事件不同的不要优先。"
    "不猜心理，不选回复策略。"
)


def _ids(tok, text):
    try:
        return tok.encode(text, add_special_tokens=False)
    except TypeError:
        return tok.encode(text)


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _render(decider, query, labels):
    custom = getattr(decider, "render_choice_suffix", None)
    if callable(custom):
        return custom(query, labels)
    from typical_ai.native import _render_row

    return _render_row(decider.tok, getattr(decider.model, "render", "semif"), query, labels)[0]


def _state_tokens(decider, state):
    custom = getattr(decider, "state_token_count", None)
    if callable(custom):
        return custom(state)
    render = getattr(getattr(decider, "model", None), "render", "semif")
    if render == "semif":
        from typical_ai.native import _semif_head_tail

        return len(_ids(decider.tok, _semif_head_tail(decider.tok)[0] + state))
    return len(_ids(decider.tok, state)) + 1


def _scoped_rows(rows, focus):
    copied = []
    previous = -1
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError("检索需要原始消息文本")
        index = row.get("source_index", row.get("index", position))
        if not isinstance(index, int) or isinstance(index, bool) or index <= previous:
            raise ValueError("检索原文行号必须递增且唯一")
        if row.get("role") not in ("self", "other", "unknown"):
            raise ValueError("检索原文发言方无效")
        copied.append({**row, "source_index": index})
        previous = index
    if not copied:
        raise ValueError("检索需要所选对方原文")
    index = focus.get("source_index", focus.get("index"))
    if index is None:
        matches = [row["source_index"] for row in copied if row["role"] == "other" and row["text"] == focus.get("text")]
        if len(matches) != 1:
            raise ValueError("所选消息缺少可确定的原文行号")
        index = matches[0]
    selected = next((row for row in copied if row["source_index"] == index), None)
    if selected is None or selected["role"] != "other" or selected["text"] != focus.get("text"):
        raise ValueError("所选对方原文与检索截止点不一致")
    return [row for row in copied if row["source_index"] <= index], selected


def _make_state(decider, rows, focus, grounding, recent_count):
    selected = rows[-1]
    topics = grounding.get("focus_topics", []) if isinstance(grounding, dict) else []
    body = {
        "selected_other_message": {"source_index": selected["source_index"], "text": selected["text"]},
        "near_dialogue": [],
        "focus_topics": [value[:32] for value in topics[:4] if isinstance(value, str)],
    }
    limit = min(1024, int(getattr(decider, "max_state", 1024)))
    state = _dump(body)
    if _state_tokens(decider, state) > limit:
        raise ValueError("所选原文超过决策状态预算，不能截断当前句")
    near = []
    for row in reversed(rows[max(0, len(rows) - recent_count) : -1]):
        candidate = [row] + near
        packed = [{key: item[key] for key in ("source_index", "role", "text")} for item in candidate]
        value = _dump({**body, "near_dialogue": packed})
        if _state_tokens(decider, value) > limit:
            break
        near, state = candidate, value
    return state, near + [selected], limit


def _label(item):
    names = {"self": "我", "other": "对方", "unknown": "发言方待核对"}
    return (
        str(item["source_index"])
        + "["
        + str(item["char_start"])
        + ":"
        + str(item["char_end"])
        + "] "
        + names[item["role"]]
        + "："
        + item["text"]
    )


def _count(decider, items):
    return len(_ids(decider.tok, _render(decider, QUERY, [_label(item) for item in items])))


def _parts(decider, rows, budget):
    # Keep enough room for at least two competing originals in a tournament.
    # Short originals remain whole; long ones have exhaustive character offsets.
    empty = _count(decider, [])
    per_candidate = max(16, min(256, (budget - empty) // 3))
    result = []
    for row in rows:
        text, offset = row["text"], 0
        if not text:
            result.append({**row, "char_start": 0, "char_end": 0})
            continue
        while offset < len(text):

            def item(end):
                return {**row, "text": text[offset:end], "char_start": offset, "char_end": end}

            complete = _count(decider, [item(len(text))])
            if complete <= budget and complete - empty <= per_candidate:
                result.append(item(len(text)))
                break
            low, high, best = offset + 1, len(text), None
            while low <= high:
                middle = (low + high) // 2
                count = _count(decider, [item(middle)])
                length = count - empty
                if length <= per_candidate and count <= budget:
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if best is None:
                raise ValueError("原文片段的元数据超过候选后缀预算")
            if best < len(text):
                # Prefer nearby punctuation/space boundaries while keeping every
                # character in its original source interval.
                stops = [match.end() for match in re.finditer(r"[。！？；，\n\s]", text[offset:best])]
                if stops and stops[-1] >= (best - offset) // 2:
                    best = offset + stops[-1]
            result.append(item(best))
            offset = best
    return result


def _pack_groups(decider, items, budget):
    groups, pending = [], []
    for item in items:
        if pending and _count(decider, pending + [item]) > budget:
            groups.append(pending)
            pending = []
        if _count(decider, [item]) > budget:
            raise ValueError("完整候选片段超过决策后缀预算")
        pending.append(item)
    if pending:
        groups.append(pending)
    return groups


def _probabilities(value, count):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    if len(value) != count + 1:
        raise ValueError("原文匹配分数数量不完整")
    probabilities = [float(number) for number in value]
    if any(not math.isfinite(number) or not 0 <= number <= 1 for number in probabilities):
        raise ValueError("原文匹配分数无效")
    if abs(sum(probabilities) - 1) > 0.002:
        raise ValueError("原文匹配分数未归一")
    mass = sum(probabilities[:-1])
    if mass <= 0:
        return None, probabilities[-1]
    return [number / mass for number in probabilities[:-1]], probabilities[-1]


def search_context(
    decider,
    rows,
    focus,
    grounding,
    *,
    score_batch=None,
    suffix_budget=MAX_SUFFIX,
    recent_count=12,
    max_selected=8,
    batch_size=4,
):
    """Return local source rows plus text-free runtime metrics.

    ``score_batch(state, queries)`` is an optional offline scoring adapter. Its
    result is one K+1 probability vector per query, including null LAST, matching
    ``native_kv_decide`` exactly. No candidate is silently token-truncated.
    """
    if not 128 <= suffix_budget <= MAX_SUFFIX or recent_count < 1 or not 1 <= max_selected <= 8:
        raise ValueError("原文检索预算无效")
    if not 1 <= batch_size <= 8:
        raise ValueError("原文检索批次大小无效")
    started = time.perf_counter()
    scoped, selected = _scoped_rows(rows, focus)
    state, near, state_limit = _make_state(decider, scoped, focus, grounding, recent_count)
    recent_indices = {row["source_index"] for row in near}
    candidates = [row for row in scoped if row["source_index"] not in recent_indices]
    parts = _parts(decider, candidates, suffix_budget)
    initial_groups = _pack_groups(decider, parts, suffix_budget)
    suffix_counts, group_sizes, nulls = [], [], []
    batch_calls, hierarchy_rounds = 0, 0
    scorer = score_batch or getattr(decider, "score_choice_batch", None)
    cache = None
    if initial_groups and scorer is None:
        cache = decider._state_kv_for(state)

    def compare(groups):
        nonlocal batch_calls
        if not groups:
            return []
        queries = [(QUERY, [_label(item) for item in group]) for group in groups]
        if scorer is not None:
            result = []
            for start in range(0, len(queries), batch_size):
                result.extend(scorer(state, queries[start : start + batch_size]))
                batch_calls += 1
        else:
            from typical_ai.native import native_kv_decide

            result = native_kv_decide(
                decider.head,
                decider.model,
                state,
                queries,
                chunk=batch_size,
                max_state=state_limit,
                max_suffix=suffix_budget,
                state_cache=cache,
                max_option_tokens=None,
            )
            batch_calls += (len(queries) + batch_size - 1) // batch_size
        if len(result) != len(groups):
            raise ValueError("原文匹配批次结果数量不完整")
        ranked = []
        for group, raw in zip(groups, result):
            probabilities, null = _probabilities(raw, len(group))
            nulls.append(null)
            suffix_counts.append(_count(decider, group))
            group_sizes.append(len(group))
            if probabilities is None:
                ranked.append([])
            else:
                order = sorted(range(len(group)), key=lambda i: (-probabilities[i], -group[i]["source_index"]))
                ranked.append([group[i] for i in order])
        return ranked

    if len(initial_groups) == 1:
        finalists = compare(initial_groups)[0]
    elif initial_groups:
        finalists = [item for group in compare(initial_groups) for item in group[:2]]
        while finalists and _count(decider, finalists) > suffix_budget:
            groups = _pack_groups(decider, finalists, suffix_budget)
            narrowed = [item for group in compare(groups) for item in group[:2]]
            if len(narrowed) >= len(finalists):
                raise ValueError("原文候选无法在完整后缀预算内进一步比较")
            finalists = narrowed
            hierarchy_rounds += 1
        if finalists:
            finalists = compare([finalists])[0]
    else:
        finalists = []

    source_lookup = {row["source_index"]: row for row in scoped}
    selected_indices = []
    matched_parts = {}
    primary_limit = min(6, max_selected)
    for item in finalists:
        index = item["source_index"]
        if index not in selected_indices:
            selected_indices.append(index)
            matched_parts[index] = item
        if len(selected_indices) >= primary_limit:
            break
    primary_indices = list(selected_indices)
    candidate_indices = {row["source_index"] for row in candidates}
    positions = {row["source_index"]: position for position, row in enumerate(scoped)}
    for index in primary_indices:
        for position in (positions[index] - 1, positions[index] + 1):
            if 0 <= position < len(scoped):
                neighbor = scoped[position]["source_index"]
                if neighbor in candidate_indices and neighbor not in selected_indices:
                    selected_indices.append(neighbor)
                if len(selected_indices) >= max_selected:
                    break
        if len(selected_indices) >= max_selected:
            break
    # Never manufacture a match when the head assigned all mass to null.
    selected_rows = [
        {
            **source_lookup[index],
            "relevance_order": order,
            "selection_reason": "jev_choice" if index in primary_indices else "adjacent_original",
        }
        for order, index in enumerate(selected_indices)
    ]
    for row in selected_rows:
        match = matched_parts.get(row["source_index"])
        if match is not None and (match["char_start"] != 0 or match["char_end"] != len(row["text"])):
            row["retrieval_fragment"] = {key: match[key] for key in ("text", "char_start", "char_end")}
    covered = {item["source_index"] for item in parts} | recent_indices
    metrics = {
        "version": VERSION,
        "source_messages_count": len(scoped),
        "selected_source_index": selected["source_index"],
        "candidate_source_indices": sorted(candidate_indices),
        "state_source_indices": sorted(recent_indices),
        "covered_source_indices": sorted(covered),
        "all_sources_covered": covered == set(source_lookup),
        "candidate_parts_count": len(parts),
        "initial_chunks": len(initial_groups),
        "choice_questions": len(group_sizes),
        "candidate_counts": group_sizes,
        "hierarchy_rounds": hierarchy_rounds,
        "global_comparison_performed": bool(finalists),
        "state_tokens": _state_tokens(decider, state),
        "state_budget_tokens": state_limit,
        "state_recent_messages_count": len(near),
        "requested_recent_messages_count": recent_count,
        "suffix_token_counts": suffix_counts,
        "max_suffix_tokens": max(suffix_counts, default=0),
        "suffix_budget_tokens": suffix_budget,
        "option_texts_truncated": 0,
        "suffix_forward_batches": batch_calls,
        "batch_size": batch_size,
        "prefix_forward_count_upper_bound": int(bool(initial_groups) and scorer is None),
        "null_only_questions": sum(value >= 1 for value in nulls),
        "selected_source_indices": selected_indices,
        "primary_selected_source_indices": primary_indices,
        "selected_rows_count": len(selected_rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return {
        "selected_rows": selected_rows,
        "metrics": metrics,
        "selection_kind": "native_choice_hierarchical_source_matching",
    }
