"""Task queue helpers: stage uploads, push jobs, update status.

Both the web process (producers) and the worker process (consumer) use
this module. All paths are absolute and validated against the configured
ROOT — never let an untrusted filename drive a path.

Why a single shared module: there is exactly one job lifecycle (push →
status hash → consume → status update → sentinel), and the format must
match between the two processes byte-for-byte.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

import redis

from config import (
    DATA,
    JOB_HASH_PREFIX,
    JOB_QUEUE,
    IMAGES_DIR,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_FILES,
    REDIS_URL,
    STAGING_DIR,
    UPLOAD_ACCEPT_EXTS,
    UPLOAD_ACCEPT_MIME,
)

log = logging.getLogger("tasks")

# Lazy connection — both processes call get_redis() and we don't want to
# block import on a missing broker (worker recovers via retry; web serves
# reads even if the broker is briefly down).
_REDIS: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    global _REDIS
    if _REDIS is None:
        # A16 fix: bounded socket timeouts so /api/health cannot hang for
        # the full TCP default (~120s) on a broker outage. health_check
        # # interval keeps idle connections alive.
        _REDIS = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
            health_check_interval=30,
        )
    return _REDIS


def _safe_name(name: str) -> str:
    """Return a name that is safe to use as a path component.

    Reject path separators, control chars, `.` and `..` (the only
    names that cause path traversal), and empty strings. Cap length.
    Always returns a non-empty string or raises ValueError — never
    returns a path-traversal-friendly name.

    A41 fix: previously we rejected ALL leading-dot names. That killed
    legit files like `.DS_Store` / `.gitkeep` (macOS folder uploads
    commonly contain these) and aborted the whole batch. The traversal
    risk is only for the names `.` and `..`.
    """
    if not name:
        raise ValueError("empty name")
    base = os.path.basename(name.replace("\\", "/"))
    if base in {".", ".."}:
        raise ValueError(f"rejected name: {name!r}")
    if any(ord(c) < 0x20 for c in base):
        raise ValueError(f"control char in name: {name!r}")
    if len(base) > 200:
        raise ValueError(f"name too long: {len(base)}")
    return base


def stage_uploads(files) -> tuple[str, int, int, list[str]]:
    """Save uploaded files to staging/<job_id>/, return (job_id, count,
    total_bytes, list_of_safe_names).

    `files` is an iterable of (filename, content_type, file-like) tuples,
    matching the shape FastAPI gives for `UploadFile`. The web layer
    already enforced MAX_UPLOAD_FILES and MAX_UPLOAD_BYTES limits, but
    we re-check here as defence in depth.

    Raises ValueError on any validation problem.
    """
    job_id = uuid.uuid4().hex
    job_dir = STAGING_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    saved: list[str] = []
    total = 0
    # A6 fix: track names already seen in this batch. If the same name
    # comes in twice (e.g. user dropped two folders with the same
    # IMG_001.png), the second one would silently overwrite the first
    # on os.replace. We now append a numeric suffix instead.
    seen: dict[str, int] = {}
    try:
        for filename, content_type, fp in files:
            if len(saved) >= MAX_UPLOAD_FILES:
                raise ValueError(
                    f"too many files: limit is {MAX_UPLOAD_FILES} per request"
                )
            name = _safe_name(filename or "")
            ext = os.path.splitext(name)[1].lower()
            if ext not in UPLOAD_ACCEPT_EXTS:
                raise ValueError(f"unsupported file type: {ext or '(none)'}")
            if content_type and content_type not in UPLOAD_ACCEPT_MIME:
                # Be lenient: many browsers send application/octet-stream.
                if content_type != "application/octet-stream":
                    raise ValueError(f"unsupported content type: {content_type}")

            # A6: in-batch dedup by appending a numeric suffix on collision.
            base_for_write = name
            if name in seen:
                seen[name] += 1
                stem = os.path.splitext(name)[0]
                base_for_write = f"{stem}-{seen[name]}{ext}"
            else:
                seen[name] = 0

            # Stream-copy to a temp file, enforce size cap, then rename.
            tmp_path = job_dir / (".part-" + base_for_write)
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = fp.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise ValueError(
                            f"upload too large: limit is "
                            f"{MAX_UPLOAD_BYTES // (1024*1024)} MB per request"
                        )
                    out.write(chunk)
            final_path = job_dir / base_for_write
            os.replace(tmp_path, final_path)
            saved.append(base_for_write)
    except BaseException:
        # Don't leave a half-baked job dir on disk if anything fails.
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    return job_id, len(saved), total, saved


def push_job(job_id: str, names: list[str]) -> None:
    """Atomically LPUSH a job and initialize its status hash.

    A consumer (worker) doing BLPOP on JOB_QUEUE receives the JSON.
    The status hash is read by the web on GET /api/upload/<job_id>.

    A21 fix: status hashes get a 7-day TTL so the queue doesn't grow
    unbounded. Operators who want longer history can bump JOB_HASH_TTL_S.
    """
    r = get_redis()
    payload = json.dumps({"job_id": job_id, "names": names, "ts": time.time()})
    pipe = r.pipeline()
    pipe.lpush(JOB_QUEUE, payload)
    pipe.hset(JOB_HASH_PREFIX + job_id, mapping={
        "status": "queued",
        "total": str(len(names)),
        "done": "0",
        "failed": "0",
        "created": str(time.time()),
    })
    # 7-day TTL on the status hash.
    pipe.expire(JOB_HASH_PREFIX + job_id, 7 * 86400)
    pipe.execute()


def update_job(job_id: str, **fields) -> None:
    r = get_redis()
    r.hset(JOB_HASH_PREFIX + job_id, mapping={k: str(v) for k, v in fields.items()})


def get_job(job_id: str) -> Optional[dict]:
    r = get_redis()
    h = r.hgetall(JOB_HASH_PREFIX + job_id)
    if not h:
        return None
    return {
        "job_id": job_id,
        "status": h.get("status", "unknown"),
        "total": int(h.get("total", "0") or 0),
        "done": int(h.get("done", "0") or 0),
        "failed": int(h.get("failed", "0") or 0),
        "created": float(h.get("created", "0") or 0.0),
        "updated": float(h.get("updated", "0") or 0.0),
        "error": h.get("error", ""),
    }


def queue_depth() -> int:
    r = get_redis()
    return int(r.llen(JOB_QUEUE) or 0)


def job_dir(job_id: str) -> Path:
    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        raise ValueError("bad job_id")
    return STAGING_DIR / job_id


def cleanup_job_dir(job_id: str) -> None:
    d = job_dir(job_id)
    shutil.rmtree(d, ignore_errors=True)


def touch_reload_sentinel() -> None:
    """Worker calls this after a successful reindex. Web polls mtime."""
    from config import RELOAD_SENTINEL
    RELOAD_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    RELOAD_SENTINEL.touch()
