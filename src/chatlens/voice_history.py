"""Extract only the selected contact's local voice BLOBs and run offline ASR."""

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parent
from .history_connector import PRIVATE, VENDOR, stable_copy
from .history_store import HistoryStore


def voice_inventory(store, target):
    with store.connect() as db:
        contact = store.contact(db, target)
        voices = [
            dict(r)
            for r in db.execute(
                "SELECT m.*, t.message_id IS NOT NULL AS has_transcript FROM messages m "
                "LEFT JOIN voice_transcripts t ON t.message_id=m.id "
                "WHERE m.talker=? AND m.kind=34 ORDER BY m.timestamp,m.sort_seq,m.id",
                (contact["talker"],),
            )
        ]
    return contact, voices


def media_sources(config):
    official = Path(config["db_dir"])
    account = official.parent.name
    debug = (
        Path.home()
        / "Library/Containers/io.github.timfang4162.wechat4-macos-frida-export.sip/Data/Documents/xwechat_files"
        / account
        / "db_storage"
    )
    sources = [("official-wechat-database", official)]
    if debug != official:
        sources.append(("isolated-wechat-database", debug))
    return sources


def read_media_databases(source, label, keys, audio, verify, decrypt, merge, stats):
    """Verify existing keys against stable copies; never touch the source DB."""
    databases = []
    originals = sorted(source.rglob("media*.db")) if source.is_dir() else []
    stats.update(
        databases_found=len(originals), validated_databases=0, missing_keys=0, database_errors={}, voices_found=0
    )
    for original in originals:
        identity = hashlib.sha256((label + ":" + str(original.relative_to(source))).encode()).hexdigest()
        encrypted = audio / (identity + ".encrypted")
        plain = audio / (identity + ".plain")
        try:
            stable_copy(original, encrypted)
            with encrypted.open("rb") as f:
                page = f.read(4096)
            key = next((k for k in keys if verify(k, page)), None)
            if key is None:
                stats["missing_keys"] += 1
                continue
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if not decrypt(str(encrypted), str(plain), key):
                    raise ValueError("media-decryption-failed")
                merge(str(encrypted), str(plain), key)
            plain.chmod(0o600)
            with sqlite3.connect(plain.as_uri() + "?mode=ro", uri=True) as db:
                if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("media-integrity-check-failed")
            databases.append(plain)
            stats["validated_databases"] += 1
        except Exception as exc:
            kind = type(exc).__name__
            stats["database_errors"][kind] = stats["database_errors"].get(kind, 0) + 1
    return databases


def voice_blob(databases, talker, server_id):
    """Exact contact plus 64-bit server ID, with conflicting BLOBs rejected."""
    blobs = set()
    for plain in databases:
        with sqlite3.connect(plain.as_uri() + "?mode=ro", uri=True) as db:
            try:
                rows = db.execute(
                    "SELECT v.voice_data FROM VoiceInfo v JOIN Name2Id n ON n.rowid=v.chat_name_id "
                    "WHERE n.user_name=? AND v.svr_id=?",
                    (talker, int(server_id)),
                )
                blobs.update(bytes(r[0]) for r in rows if r[0])
            except sqlite3.OperationalError:
                continue
    if len(blobs) != 1:
        return None
    blob = blobs.pop()
    return blob if blob.startswith((b"\x02#!SILK_V3", b"#!SILK_V3")) else None


def extract(store, target, metadata=None):
    """Official media first; use the already-existing debug media only as fallback."""
    contact, voices = voice_inventory(store, target)
    pending = [v for v in voices if not v["has_transcript"]]
    info = metadata if metadata is not None else {}
    info.update(skipped_cached=len(voices) - len(pending), pending_voice_messages=len(pending), media_sources={})
    if not pending:
        return voices, [], 0
    cfg = json.loads((PRIVATE / "active-source.json").read_text())
    if VENDOR is not None:
        sys.path.insert(0, str(VENDOR / "src"))
    from wxexport.keys.map import page1_hmac_ok
    from wxexport.decrypt import decrypt_database, merge_wal

    keys = []
    keyfile = PRIVATE / "runtime-candidates.txt"
    for value in keyfile.read_text().splitlines() if keyfile.exists() else []:
        try:
            key = bytes.fromhex(value)
            if len(key) == 32 and key not in keys:
                keys.append(key)
        except ValueError:
            pass
    audio = PRIVATE / "audio"
    audio.mkdir(mode=0o700, exist_ok=True)
    found = []
    missing_keys = 0
    for label, source in media_sources(cfg):
        if not pending:
            break
        stats = info["media_sources"][label] = {}
        databases = read_media_databases(source, label, keys, audio, page1_hmac_ok, decrypt_database, merge_wal, stats)
        missing_keys += stats["missing_keys"]
        remaining = []
        for voice in pending:
            try:
                blob = voice_blob(databases, contact["talker"], voice["server_id"])
            except (ValueError, sqlite3.Error):
                blob = None
            if blob is None:
                remaining.append(voice)
                continue
            path = audio / (voice["id"] + ".silk")
            path.write_bytes(blob)
            path.chmod(0o600)
            found.append((voice, path))
            stats["voices_found"] += 1
        pending = remaining
    return voices, found, missing_keys


def report_paths(target):
    stem = "contact-" + hashlib.sha256(target.encode()).hexdigest()[:12]
    return PRIVATE / ("reports/" + stem + "-voice-history.json"), PRIVATE / ("audio/" + stem + "-errors.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-contact local offline voice transcription")
    parser.add_argument("--target", required=True)
    args = parser.parse_args(argv)
    target = args.target.strip()
    if not target or len(target) > 100:
        raise ValueError("invalid-contact-name")
    os.umask(0o077)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from .server import install_local_network_guard

    install_local_network_guard()
    store = HistoryStore(PRIVATE / "memory.sqlite3")
    reportpath, errorpath = report_paths(target)
    reportpath.parent.mkdir(parents=True, exist_ok=True)
    errorpath.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    began = time.perf_counter()
    report = {
        "target": target,
        "voice_messages": 0,
        "local_audio_found": 0,
        "missing_local_audio": 0,
        "missing_database_keys": 0,
        "skipped_cached": 0,
        "pending_voice_messages": 0,
        "transcribed": 0,
        "failed": 0,
        "empty": 0,
        "audio_seconds": 0,
        "elapsed_seconds": 0,
        "offline_inference": True,
        "model": "mlx-community/whisper-large-v3-turbo",
        "asr_verified_by_human": False,
        "status": "running",
    }
    errors = []

    def save():
        report["elapsed_seconds"] = round(time.perf_counter() - began, 2)
        temporary = reportpath.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        temporary.chmod(0o600)
        os.replace(temporary, reportpath)

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            voices, found, missing_keys = extract(store, target, report)
        report.update(
            voice_messages=len(voices),
            local_audio_found=len(found),
            missing_local_audio=report["pending_voice_messages"] - len(found),
            missing_database_keys=missing_keys,
        )
        save()
        # Repeat runs with cached transcripts do not load the model or decode audio.
        if found:
            from .config import MODELS_DIR

            modelpath = MODELS_DIR / "whisper-large-v3-turbo-mlx"
            if not (modelpath / "weights.safetensors").is_file():
                raise ValueError("local-asr-model-unavailable")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                import numpy as np
                import pilk
                import mlx_whisper
                import mlx.core as mx
            durations = 0
            for number, (voice, path) in enumerate(found, 1):
                pcm = path.with_suffix(".pcm")
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        pilk.decode(str(path), str(pcm), pcm_rate=16000)
                        waveform = np.frombuffer(pcm.read_bytes(), dtype="<i2").astype(np.float32) / 32768.0
                        duration = len(waveform) / 16000.0
                        result = mlx_whisper.transcribe(
                            waveform,
                            path_or_hf_repo=str(modelpath),
                            language="zh",
                            verbose=None,
                            temperature=0.0,
                            condition_on_previous_text=False,
                        )
                    text = result["text"].strip()
                    segments = result.get("segments", [])
                    metrics = {
                        "segments": len(segments),
                        "avg_logprob": [s.get("avg_logprob") for s in segments],
                        "no_speech_prob": [s.get("no_speech_prob") for s in segments],
                        "human_verified": False,
                    }
                    if text:
                        store.save_transcript(
                            voice["id"],
                            text,
                            report["model"],
                            hashlib.sha256(path.read_bytes()).hexdigest(),
                            duration,
                            metrics,
                        )
                        report["transcribed"] += 1
                    else:
                        report["empty"] += 1
                    durations += duration
                except Exception as exc:
                    errors.append({"message_id": voice["id"], "error_type": type(exc).__name__})
                    report["failed"] += 1
                finally:
                    pcm.unlink(missing_ok=True)
                report["audio_seconds"] = round(durations, 2)
                save()
                print(
                    json.dumps(
                        {
                            "processed": number,
                            "total": len(found),
                            "transcribed": report["transcribed"],
                            "failed": report["failed"],
                            "elapsed_seconds": report["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
                mx.clear_cache()
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
    errorpath.write_text(json.dumps(errors))
    errorpath.chmod(0o600)
    save()
    # Public CLI logs contain only aggregate metadata; text, keys, IDs stay private.
    public = {k: v for k, v in report.items() if k not in ("target", "media_sources", "model")}
    print(json.dumps(public, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), flush=True)
        sys.exit(1)
