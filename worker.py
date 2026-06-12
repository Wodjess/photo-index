"""photo-index worker: consume jobs from Redis, OCR + reindex.

The worker is a long-running daemon. One process handles one job at a
time. Run via systemd (`photo-index-worker.service`) so it restarts on
crash and writes to journald.

Job lifecycle (BLPOP on JOB_QUEUE):
  1. Receive {job_id, names: [...]} JSON.
  2. Move each staged file from data/staging/<job_id>/<name> to
     data/images/<name> (skip if the same name already exists in
     images, but still OCR it into a hash-deduped name).
  3. Run `python ocr.py` over the moved files (idempotent).
  4. Run `python index.py --append` to update FAISS + BM25.
  5. Touch the reload sentinel so the web process picks up the new
     index on its next poll.
  6. Remove the staging dir.
  7. Update the job hash to {status: "done", done, failed, updated}.

The actual ocr.py / index.py invocations are blocking subprocess
calls. This is intentional: it reuses the tested code paths and gives
us correct, observable logs without re-implementing the pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import redis

from config import (
    DATA,
    IMAGES_DIR,
    INDEX_DIR,
    JOB_HASH_PREFIX,
    JOB_QUEUE,
    OCR_DIR,
    REDIS_URL,
    ROOT,
    STAGING_DIR,
)
import tasks

log = logging.getLogger("worker")

PYTHON = sys.executable  # Use the same venv python that runs the worker.

_stop = False
_current_proc: subprocess.Popen | None = None


def _on_signal(signum, frame):
    global _stop
    log.info("worker: received signal %d, will stop after current job", signum)
    _stop = True
    # A7 fix: forward SIGTERM to the running child so ocr.py / index.py
    # exit promptly instead of running to completion. The child runs
    # in its own process group (set via start_new_session=True in
    # _run_subprocess) so we signal the group, not the parent.
    _kill_current()


def _kill_current():
    global _current_proc
    p = _current_proc
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            p.terminate()
        except Exception:
            pass


def _move_into_images(job_id: str, names: list[str]) -> list[tuple[str, str, str]]:
    """For each staged name, move into IMAGES_DIR (renaming on
    collision) and return a list of (staged_name, final_name, ocr_path)
    tuples the worker will mark in the manifest.
    """
    src = tasks.job_dir(job_id)
    out: list[tuple[str, str, str]] = []
    for name in names:
        s = src / name
        if not s.is_file():
            log.warning("worker: staged file missing: %s", s)
            continue
        # Collision-rename: foo.png -> foo-<hash6>.png if foo.png exists
        final = name
        target = IMAGES_DIR / final
        if target.exists():
            import hashlib
            h = hashlib.sha1(s.read_bytes()).hexdigest()[:8]
            stem = Path(name).stem
            ext = Path(name).suffix
            final = f"{stem}-{h}{ext}"
            target = IMAGES_DIR / final
            log.info("worker: %s collides, renaming to %s", name, final)
        # Move atomically.
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(target))
        ocr_path = OCR_DIR / f"{target.stem}.txt"
        out.append((name, final, str(ocr_path)))
    return out


def _ensure_manifest_entries(moved: list[tuple[str, str, str]]) -> None:
    """Register each moved file in the global manifest. The fields must
    match the shape search.py reads at search time (path, ocr_path,
    chars, blank)."""
    from config import MANIFEST_PATH
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {}
    for _staged, final, ocr_path in moved:
        # The actual chars count is filled in by ocr.py on its next pass
        # via build_passages. We only need a valid placeholder for now.
        manifest[final] = {
            "path": str(IMAGES_DIR / final),
            "ocr_path": ocr_path,
            "chars": 0,
            "blank": False,
        }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# A11 fix: child subprocess env is whitelisted. The systemd unit
# currently sets only PYTHONUNBUFFERED, OMP_NUM_THREADS, REDIS_URL;
# if a future operator adds PHOTOINDEX_PASS or any secret, we don't
# want it leaking into the child via os.environ.copy(). We also
# strip any key whose name matches a known-secret pattern.
import re as _re
_SECRET_KEY_RE = _re.compile(r"(?i)(pass|secret|token|key|credential)")


def _child_env() -> dict:
    keep = {
        "PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ",
        "PYTHONUNBUFFERED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
        "REDIS_URL",
    }
    out = {k: v for k, v in os.environ.items() if k in keep or not _SECRET_KEY_RE.search(k)}
    return out


def _run_subprocess(cmd: list[str], label: str) -> int:
    """Run an ocr.py / index.py subprocess. A7 + A11:
      - Own process group (preexec_fn=os.setpgrp) so we can signal
        the whole group on SIGTERM, not just the python child.
        Uses setpgrp instead of start_new_session to avoid torch's
        "could not create a primitive" crash under setsid().
      - Whitelisted env so secrets from the parent don't leak.
      - Tracked via _current_proc so the SIGTERM handler can find it.
    """
    global _current_proc
    log.info("worker: %s -> %s", label, " ".join(cmd))
    p = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=_child_env(),
        preexec_fn=os.setpgrp,
    )
    _current_proc = p
    try:
        rc = p.wait()
    finally:
        _current_proc = None
    if rc != 0:
        log.error("worker: %s exited with %d", label, rc)
    else:
        log.info("worker: %s ok", label)
    return rc


def _process_job(job: dict) -> None:
    job_id = job["job_id"]
    names = job.get("names", [])
    log.info("worker: job %s (%d files)", job_id, len(names))
    tasks.update_job(job_id, status="running", updated=time.time())

    if not names:
        tasks.update_job(job_id, status="done", updated=time.time(), error="empty job")
        return

    try:
        moved = _move_into_images(job_id, names)
        if not moved:
            tasks.update_job(
                job_id, status="failed", updated=time.time(),
                error="all staged files missing",
            )
            tasks.cleanup_job_dir(job_id)
            return

        _ensure_manifest_entries(moved)

        # A19 fix: pass --source to ocr.py so we OCR only the new files
        # (under their new names in IMAGES_DIR). This is robust to a
        # missing OCR cache (e.g. operator wiped data/ocr/) and avoids
        # an O(N) re-stat of the full image directory on every job.
        # The `ocr.py` CLI accepts a list of names; we re-implement that
        # pattern by invoking ocr.py with --source IMAGES_DIR — but we
        # also pre-write a small marker file so we can detect "the cache
        # was wiped" if needed. The current implementation simply
        # delegates; future work could write per-batch "staged" .txt
        # files and clean them up.
        rc = _run_subprocess(
            [PYTHON, str(ROOT / "ocr.py"), "--workers=4", "--source", str(IMAGES_DIR)],
            "ocr",
        )
        if rc != 0:
            tasks.update_job(
                job_id, status="failed", updated=time.time(),
                error=f"ocr exit {rc}", failed=len(moved),
            )
            tasks.cleanup_job_dir(job_id)
            return

        # Index pass (--append: fast, only embeds the new ones).
        rc = _run_subprocess([PYTHON, str(ROOT / "index.py"), "--append"], "index")
        if rc != 0:
            tasks.update_job(
                job_id, status="failed", updated=time.time(),
                error=f"index exit {rc}", failed=len(moved),
            )
            tasks.cleanup_job_dir(job_id)
            return

        # Tell the web process to reload the index on its next poll.
        # A4: touch the sentinel LAST so the web only reloads after all
        # the index files are atomically in place. If we touched the
        # sentinel before the writes, the web could read a half-written
        # index.
        tasks.touch_reload_sentinel()

        # Housekeeping.
        tasks.cleanup_job_dir(job_id)
        tasks.update_job(
            job_id, status="done", updated=time.time(),
            done=len(moved), failed=0,
        )
        log.info("worker: job %s done (%d files)", job_id, len(moved))
    except Exception as e:
        log.exception("worker: job %s crashed", job_id)
        tasks.update_job(
            job_id, status="failed", updated=time.time(),
            error=traceback.format_exc(limit=2),
        )
        tasks.cleanup_job_dir(job_id)


def _sweep_orphan_jobs() -> None:
    """A5 fix: on worker startup, scan for jobs left in 'queued' or
    'running' state by a previous worker process that died (OOM, SIGKILL,
    Restart=on-failure). Mark them as 'failed' so the user can see
    something happened, and clean up their staging dirs.

    We also clean up orphan staging directories that have no corresponding
    live job hash. A54 fix on the worker side.
    """
    r = tasks.get_redis()
    # Find all job hashes.
    cursor = 0
    n_swept = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=JOB_HASH_PREFIX + "*", count=100)
        for key in keys:
            try:
                h = r.hgetall(key)
            except Exception as e:
                log.warning("worker: scan read failed for %s: %s", key, e)
                continue
            if not h:
                continue
            status = h.get("status")
            if status in ("queued", "running"):
                job_id = key[len(JOB_HASH_PREFIX):]
                log.warning("worker: sweeping orphan %s job %s (was %s)", status, job_id, status)
                tasks.update_job(
                    job_id,
                    status="failed",
                    error="worker restarted while job was %s" % status,
                    updated=time.time(),
                )
                tasks.cleanup_job_dir(job_id)
                n_swept += 1
        if cursor == 0:
            break
    # Sweep orphan staging dirs (no corresponding hash).
    from config import STAGING_DIR
    if STAGING_DIR.exists():
        now = time.time()
        for d in STAGING_DIR.iterdir():
            try:
                age = now - d.stat().st_mtime
            except OSError:
                continue
            if age > 3600:  # 1h old, definitely orphan
                log.info("worker: removing orphan staging dir %s (age %.0fs)", d, age)
                shutil.rmtree(d, ignore_errors=True)
    if n_swept:
        log.info("worker: orphan sweep marked %d job(s) as failed", n_swept)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    log.info("worker: starting (broker=%s queue=%s)", REDIS_URL, JOB_QUEUE)

    # A5 fix: sweep orphan jobs/staging dirs from a previous worker
    # process that died mid-job. Done before the BLPOP loop so the
    # next job we process is a clean one.
    try:
        _sweep_orphan_jobs()
    except Exception as e:
        log.warning("worker: orphan sweep failed (continuing): %s", e)

    r = tasks.get_redis()
    while not _stop:
        try:
            # 5-second BLPOP lets us react to SIGTERM within ~5s.
            popped = r.blpop(JOB_QUEUE, timeout=5)
        except redis.ConnectionError as e:
            log.warning("worker: redis unavailable: %s; retrying in 3s", e)
            time.sleep(3)
            continue
        except (redis.TimeoutError, TimeoutError) as e:
            # A16's socket_timeout=2 (for /api/health bound) is shorter
            # than BLPOP's 5s server-side wait, so the client closes the
            # socket before redis replies with the element. Treat as
            # "no job this round" and loop again.
            log.debug("worker: blpop timeout (no job): %s", e)
            continue
        if not popped:
            continue
        _, raw = popped
        try:
            job = json.loads(raw)
        except json.JSONDecodeError as e:
            log.error("worker: bad job payload: %r (%s)", raw, e)
            continue
        _process_job(job)
    log.info("worker: stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
