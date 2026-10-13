"""Download pinned public weights and verify every LFS file against its SHA-256."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")

from huggingface_hub import HfApi, snapshot_download

from .config import MODELS_DIR

ROOT = MODELS_DIR
REPOS = {
    "OzLabs/typical-small": ("typical-small-1.7b", "0ff732eafcc22d879446628433c920466e913283"),
    "Qwen/Qwen3-1.7B-Base": ("qwen3-1.7b-base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"),
    "openbmb/MiniCPM5-2B-MLX": ("minicpm5-2b-mlx", "8a9ad7539ac86281d0ac2b017ba04a5de53fe9a3"),
}


def download(repo, name, revision):
    api = HfApi(token=False)
    info = api.model_info(repo, revision=revision, files_metadata=True)
    destination = ROOT / name
    expected = sum(file.size or 0 for file in info.siblings)
    print(
        json.dumps(
            {"event": "start", "repo": repo, "revision": info.sha, "expected_bytes": expected}, ensure_ascii=False
        ),
        flush=True,
    )
    snapshot_download(repo, revision=info.sha, local_dir=destination, token=False, max_workers=2)
    files = []
    for file in info.siblings:
        path = destination / file.rfilename
        if not path.is_file():
            raise RuntimeError(f"Missing file: {path}")
        size = path.stat().st_size
        if file.size is not None and size != file.size:
            raise RuntimeError(f"Size mismatch: {path}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        lfs_hash = file.lfs.get("sha256") if file.lfs else None
        if lfs_hash and digest != lfs_hash:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
        files.append(
            {"path": file.rfilename, "bytes": size, "sha256": digest, "upstream_lfs_sha256_verified": bool(lfs_hash)}
        )
    result = {
        "repo": repo,
        "revision": info.sha,
        "directory": name,
        "total_bytes": sum(file["bytes"] for file in files),
        "files": files,
    }
    print(
        json.dumps({"event": "verified", "repo": repo, "bytes": result["total_bytes"]}, ensure_ascii=False), flush=True
    )
    (destination / "DOWNLOAD_MANIFEST.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(download, repo, name, revision) for repo, (name, revision) in REPOS.items()]
        for future in as_completed(pending):
            results.append(future.result())
    manifest = {
        "download_elapsed_seconds": time.time() - started,
        "models": sorted(results, key=lambda item: item["repo"]),
    }
    (ROOT / "models-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"event": "complete", "total_bytes": sum(item["total_bytes"] for item in results)}, ensure_ascii=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
