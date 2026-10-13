"""Exact prompt budgets and lossless source-aware context planning.

This module never calls a model. The caller summarizes ``earlier_messages`` and
calls ``fit_context_payload`` again with those chronological summaries. Input
records must already stop at the selected message; ``subject_index`` can enforce
that boundary when a larger history is supplied. No action or prompt field is
removed to make the mandatory payload fit.
"""

import copy
import json


class ContextBudgetError(ValueError):
    """A required prompt cannot fit; the exception contains only token metadata."""

    def __init__(self, reason, required_tokens, limit):
        self.reason = reason
        self.required_tokens = required_tokens
        self.limit = limit
        super().__init__(f"{reason}: {required_tokens} tokens exceed budget {limit}")


def _positive_budget(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("token budget must be a positive integer")
    return value


def prompt_text(header, payload):
    """Serialize exactly as the product's generation path does."""
    return header + json.dumps(payload, ensure_ascii=False)


def prompt_tokens(tokenizer, header, payload):
    """Count the full user chat template, including the assistant thinking prefix.

    A tokenizer without ``apply_chat_template`` is supported for offline fakes;
    deployed tokenizers must use their actual chat template. Counting an already
    rendered template with encode matches ``local_models.generate_reply``.
    """
    content = prompt_text(header, payload)
    template = getattr(tokenizer, "apply_chat_template", None)
    if callable(template):
        rendered = template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    else:
        rendered = content
    return len(tokenizer.encode(rendered))


def indexed_messages(messages, *, subject_index=None):
    """Copy records while keeping supplied source indices and the anchor boundary."""
    if subject_index is not None and (
        not isinstance(subject_index, int) or isinstance(subject_index, bool) or subject_index < 0
    ):
        raise ValueError("selected source index must be a non-negative integer")
    result = []
    previous = -1
    for position, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("text"), str):
            raise ValueError("each message requires an original text string")
        index = message.get("source_index", message.get("index", position))
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index <= previous:
            raise ValueError("source indices must be unique increasing non-negative integers")
        previous = index
        if subject_index is not None and index > subject_index:
            break
        row = copy.deepcopy(message)
        row["source_index"] = index
        result.append(row)
    if subject_index is not None and (not result or result[-1]["source_index"] != subject_index):
        raise ValueError("selected source index is absent from supplied history")
    return result


def split_earlier(
    messages, tokenizer, header, max_input_tokens=2048, *, payload_key="messages", base_payload=None, subject_index=None
):
    """Pack chronological raw records into prompts with exact token limits.

    Returns chunks containing ``payload``, ``messages``, ``source_indices`` and
    ``prompt_tokens``. A single oversize text is split losslessly into consecutive
    character ranges; all parts keep the same ``source_index`` and supplied IDs.
    Part numbers are one based. Offsets refer to the original Python string.
    ``base_payload`` is copied in every chunk, without dropping mandatory keys.
    """
    limit = _positive_budget(max_input_tokens)
    base = copy.deepcopy(base_payload or {})
    if payload_key in base:
        raise ValueError("chunk payload key must not already exist in base payload")
    rows = indexed_messages(messages, subject_index=subject_index)

    def payload(items):
        return {**base, payload_key: items}

    empty_tokens = prompt_tokens(tokenizer, header, payload([]))
    if empty_tokens > limit:
        raise ContextBudgetError("chunk_header_too_large", empty_tokens, limit)
    chunks = []
    pending = []

    def emit(items):
        body = payload(items)
        count = prompt_tokens(tokenizer, header, body)
        if count > limit:
            raise ContextBudgetError("chunk_exceeds_budget", count, limit)
        chunks.append(
            {
                "payload": body,
                "messages": items,
                "source_indices": list(dict.fromkeys(x["source_index"] for x in items)),
                "prompt_tokens": count,
            }
        )

    for row in rows:
        if prompt_tokens(tokenizer, header, payload(pending + [row])) <= limit:
            pending.append(row)
            continue
        if pending:
            emit(pending)
            pending = []
        if prompt_tokens(tokenizer, header, payload([row])) <= limit:
            pending = [row]
            continue
        text = row["text"]
        offset = 0
        part_number = 1
        if not text:
            raise ContextBudgetError(
                "message_metadata_too_large", prompt_tokens(tokenizer, header, payload([row])), limit
            )
        while offset < len(text):

            def part(end):
                return {**row, "text": text[offset:end], "part": part_number, "char_start": offset, "char_end": end}

            smallest = prompt_tokens(tokenizer, header, payload([part(offset + 1)]))
            if smallest > limit:
                raise ContextBudgetError("message_metadata_too_large", smallest, limit)
            # Bisection finds a useful prefix. Every accepted prefix is measured
            # exactly; tokenization need not be strictly monotonic to stay safe.
            low, high = offset + 1, len(text)
            best = low
            while low <= high:
                middle = (low + high) // 2
                if prompt_tokens(tokenizer, header, payload([part(middle)])) <= limit:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            emit([part(best)])
            offset = best
            part_number += 1
    if pending:
        emit(pending)
    return chunks


def fit_context_payload(
    base_payload,
    indexed_lines,
    summaries,
    tokenizer,
    header,
    max_input_tokens=2712,
    *,
    min_recent_messages=4,
    conversation_key="conversation",
    summaries_key="conversation_summaries",
    conversation_formatter=None,
    subject_index=None,
):
    """Keep the anchor and the largest fitting suffix of original messages.

    ``base_payload`` is immutable mandatory context. Summaries are included in
    their supplied order, never silently shortened. The returned ``earlier_messages``
    must be summarized by the caller; they have not been forgotten or analyzed.
    If fewer than ``min_recent_messages`` fit, this is explicit in the returned
    metadata. A too-large anchor or summaries produces ``ContextBudgetError``.
    ``conversation_formatter`` lets the product keep its existing string format.
    """
    limit = _positive_budget(max_input_tokens)
    if not isinstance(min_recent_messages, int) or isinstance(min_recent_messages, bool) or min_recent_messages < 1:
        raise ValueError("min_recent_messages must be a positive integer")
    if conversation_key in base_payload or summaries_key in base_payload:
        raise ValueError("context fields must not already exist in base payload")
    rows = indexed_messages(indexed_lines, subject_index=subject_index)
    if not rows:
        raise ValueError("selected message is required")
    copied_base = copy.deepcopy(base_payload)
    copied_summaries = copy.deepcopy(list(summaries))
    for memory in copied_summaries:
        if not isinstance(memory, dict):
            continue
        source_range = memory.get("range")
        if source_range is not None:
            if (
                not isinstance(source_range, (list, tuple))
                or len(source_range) != 2
                or any(not isinstance(index, int) or isinstance(index, bool) for index in source_range)
                or not 0 <= source_range[0] <= source_range[1] <= rows[-1]["source_index"]
            ):
                raise ValueError("summary range exceeds selected source boundary or is invalid")
    formatter = conversation_formatter or (lambda items: items)

    def payload(recent, memories):
        return {**copied_base, conversation_key: formatter(copy.deepcopy(recent)), summaries_key: memories}

    anchor_only = payload(rows[-1:], [])
    anchor_tokens = prompt_tokens(tokenizer, header, anchor_only)
    if anchor_tokens > limit:
        raise ContextBudgetError("mandatory_anchor_too_large", anchor_tokens, limit)
    chosen = rows[-1:]
    body = payload(chosen, copied_summaries)
    count = prompt_tokens(tokenizer, header, body)
    if count > limit:
        raise ContextBudgetError("summaries_too_large", count, limit)
    start = len(rows) - 1
    # Keep the most recent contiguous originals. Stop at the first overflow so
    # chronological adjacency is preserved, rather than cherry-picking short rows.
    while start > 0:
        candidate = payload(rows[start - 1 :], copied_summaries)
        candidate_tokens = prompt_tokens(tokenizer, header, candidate)
        if candidate_tokens > limit:
            break
        start -= 1
        body, count = candidate, candidate_tokens
    recent = rows[start:]
    earlier = rows[:start]
    return {
        "payload": body,
        "earlier_messages": earlier,
        "recent_messages": recent,
        "prompt_tokens": count,
        "max_input_tokens": limit,
        "needs_compression": bool(earlier),
        "anchor_source_index": rows[-1]["source_index"],
        "recent_message_count": len(recent),
        "minimum_recent_satisfied": len(recent) >= min(min_recent_messages, len(rows)),
        "compressed_source_indices": [row["source_index"] for row in earlier],
    }
