"""Persist numeric decision telemetry, without chat text or generated drafts."""

import hashlib
import json
import os
import time


def record_decision(path, request, result, status, version):
    decision = result.get("decision", {})
    emotion = result.get("emotion", {})
    selected = request.get("selected_message") or {}
    row = {
        "created_at": time.time(),
        "status": status,
        "workflow_version": version,
        "source": request.get("source"),
        "contact_hash": hashlib.sha256(str(request.get("target", "")).encode()).hexdigest(),
        "anchor_hash": hashlib.sha256(str(selected.get("id", request.get("through_id", ""))).encode()).hexdigest(),
        "actions": [
            {"id": r["id"], "label": r["action"], "score10": r.get("score10")} for r in decision.get("ranked", [])[:3]
        ],
        "emotions": [{"label": r["label"], "score10": r.get("score10")} for r in emotion.get("ranked", [])[:3]],
        "llm_calls": result.get("llm_call_count"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "writing_errors": [m["validation_error"] for m in result.get("generations", []) if m.get("validation_error")],
    }
    if not row["actions"]:
        return
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "w") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        # Optional statistics must not turn a completed card into a UI error.
        return False
    return True
