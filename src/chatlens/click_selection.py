"""Resolve structured AX/OCR fragments to database IDs without guessing duplicates."""

import re
import unicodedata
from difflib import SequenceMatcher


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def similarity(fragment, text):
    a, b = normalized(fragment), normalized(text)
    if len(a) < 2 or not b:
        return 0
    if a in b:
        return 1.0
    if len(a) >= 8:
        ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
        if ratio >= 0.9:
            return ratio
    return 0


def candidate_rows(db, talker, fragments, limit=2000):
    """Indexed normalized substring search; fuzzy OCR narrows by intact segments."""
    found = {}
    truncated = False
    for fragment in fragments:
        text = normalized(fragment)
        if len(text) == 1:
            rows = db.execute(
                "SELECT m.* FROM message_display d JOIN messages m ON m.id=d.id WHERE d.talker=? AND d.normalized=? LIMIT ?",
                (talker, text, limit + 1),
            ).fetchall()
        elif len(text) >= 3:
            phrase = '"' + text.replace('"', '""') + '"'
            rows = db.execute(
                "SELECT m.* FROM display_search f JOIN messages m ON m.id=f.id "
                "WHERE display_search MATCH ? AND m.talker=? LIMIT ?",
                (phrase, talker, limit + 1),
            ).fetchall()
            if not rows and len(text) >= 8:
                pieces = {text[i : i + 4] for i in range(0, len(text) - 3, max(4, len(text) // 4))}
                query = " OR ".join('"' + piece.replace('"', '""') + '"' for piece in pieces)
                rows = db.execute(
                    "SELECT m.* FROM display_search f JOIN messages m ON m.id=f.id "
                    "WHERE display_search MATCH ? AND m.talker=? LIMIT ?",
                    (query, talker, limit + 1),
                ).fetchall()
        else:
            # Two-character fragments cannot use FTS trigram indexes. Keep the
            # fallback bounded, and never report a unique match after truncation.
            escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = db.execute(
                "SELECT m.* FROM message_display d JOIN messages m ON m.id=d.id "
                "WHERE d.talker=? AND d.normalized LIKE ? ESCAPE '\\' LIMIT ?",
                (talker, "%" + escaped + "%", limit + 1),
            ).fetchall()
        truncated = truncated or len(rows) > limit
        for row in rows[:limit]:
            found[row["id"]] = dict(row)
    return list(found.values()), truncated


def resolve(store, target, locator):
    groups = {}
    for key in ("clicked_texts", "before_texts", "after_texts"):
        values = locator.get(key, [])
        if (
            not isinstance(values, list)
            or len(values) > 8
            or any(not isinstance(x, str) or len(x) > 800 for x in values)
        ):
            raise ValueError("点击定位文本格式错误")
        minimum = 1 if locator.get("source") == "ax" else 2
        groups[key] = [x for x in values if len(normalized(x)) >= minimum]
    matches = []
    with store.connect() as db:
        c = store.contact(db, target)
        rows, truncated = candidate_rows(db, c["talker"], groups["clicked_texts"])
        for row in rows:
            base = max(
                (
                    1.0
                    if locator.get("source") == "ax" and normalized(t) == normalized(row["text"])
                    else similarity(t, row["text"])
                    for t in groups["clicked_texts"]
                ),
                default=0,
            )
            if not base:
                continue
            score = 10 * base
            neighbor_hits = 0
            boundary = (row["timestamp"], row["sort_seq"], row["id"])
            for key, op, order in (("before_texts", "<", "DESC"), ("after_texts", ">", "ASC")):
                neighbors = db.execute(
                    f"SELECT text FROM messages WHERE talker=? AND "
                    f"(timestamp,sort_seq,id){op}(?,?,?) ORDER BY timestamp {order},sort_seq {order},id {order} LIMIT 15",
                    (c["talker"], *boundary),
                ).fetchall()
                for text in groups[key]:
                    best = max((similarity(text, n["text"]) for n in neighbors), default=0)
                    if best:
                        score += 2 * best
                        neighbor_hits += 1
            matches.append(
                {
                    "message": {k: row[k] for k in ("id", "timestamp", "sort_seq", "role", "text", "kind")},
                    "match_score": score,
                    "neighbor_hits": neighbor_hits,
                }
            )
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    unique = len(matches) == 1 and not truncated
    separated = (
        not truncated
        and len(matches) > 1
        and matches[0]["neighbor_hits"] > 0
        and matches[0]["match_score"] - matches[1]["match_score"] >= 2
    )
    return {
        "status": "matched" if unique or separated else ("ambiguous" if matches or truncated else "no_match"),
        "message": matches[0]["message"] if unique or separated else None,
        "candidates": matches[:5],
        "matched_count": len(matches),
        "candidate_count": len(rows),
        "truncated": truncated,
        "method": "indexed-display-fragment-to-database-id",
    }
