"""Web UI for the photo-index semantic search.

Bound to 0.0.0.0:7860 by default (see WEB_HOST / WEB_PORT env vars).
SSR via Jinja2 for the initial paint; the front-end then drives
/api/search from JS so the 5-card grid + cinema mode feel instant.

Round 0 (settings+upload):
  - POST /api/upload  (multipart, up to 300 files / 200 MB)
  - GET  /api/upload/{job_id}  (job status from Redis hash)
  - GET  /api/health  (liveness + queue depth + indexed count)
  - /api/search accepts `k` 1..MAX_TOP_K (default MAX_TOP_K, capped
    server-side).
  - Background asyncio task polls data/index/.reload mtime and calls
    ensure_loaded() on change.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    BASIC_AUTH_PASS,
    BASIC_AUTH_USER,
    EMBEDDINGS_PATH,
    FAISS_PATH,
    IMAGES_DIR,
    INDEX_DIR,
    MANIFEST_PATH,
    MAX_TOP_K,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_FILES,
    OCR_DIR,
    RELOAD_POLL_S,
    RELOAD_SENTINEL,
    DEFAULT_TOP_K,
)
import search as search_mod
import tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("web")

_ROOT = Path(__file__).parent
_TEMPLATES = Jinja2Templates(directory=str(_ROOT / "templates"))
_STATIC_DIR = _ROOT / "templates" / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_loaded() -> None:
    """Load (or reload after a sentinel touch) the in-memory index.

    Wraps search_mod.ensure_loaded so the web can force a fresh read from
    disk. search_mod itself caches the index; this wrapper drops the
    cache by calling search_mod.reload() if it exists, otherwise clears
    the module-level singletons.
    """
    reload_fn = getattr(search_mod, "reload", None)
    if callable(reload_fn):
        reload_fn()
        return
    # Fallback: poke the module's private cache.
    for name in ("_INDEX", "_META", "_MANIFEST", "_MODEL", "_BM25"):
        if hasattr(search_mod, name):
            setattr(search_mod, name, None)
    search_mod.ensure_loaded()


app = FastAPI(title="photo-index", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── background sentinel poller (reloads FAISS/BM25/model on worker write) ──
_sentinel_mtime: float = 0.0
_sentinel_task: asyncio.Task | None = None


async def _reload_watcher() -> None:
    """Poll the .reload sentinel mtime. When it advances, re-load the
    in-memory index. Sleep is cancellation-friendly.

    A3 fix: serialize reloads with an asyncio lock. Two concurrent
    reloads would race on search_mod._INDEX / _BM25 and (worse) load
    the 3 GB bge-m3 model twice, which can OOM the 8 GB container cap.
    """
    global _sentinel_mtime
    _reload_lock = asyncio.Lock()
    if not RELOAD_SENTINEL.exists():
        # First run: create the sentinel so the worker has a target.
        try:
            RELOAD_SENTINEL.touch()
        except OSError:
            pass
        _sentinel_mtime = RELOAD_SENTINEL.stat().st_mtime
    else:
        _sentinel_mtime = RELOAD_SENTINEL.stat().st_mtime

    while True:
        try:
            await asyncio.sleep(RELOAD_POLL_S)
            if not RELOAD_SENTINEL.exists():
                continue
            m = RELOAD_SENTINEL.stat().st_mtime
            if m > _sentinel_mtime:
                # A3: skip if a reload is in progress. We re-read mtime
                # after acquiring the lock; if the worker touched again
                # in the meantime, we'll pick it up on the next poll.
                if _reload_lock.locked():
                    log.info("web: reload already in progress, skipping this tick")
                    continue
                async with _reload_lock:
                    # Re-check inside the lock: mtime may have advanced.
                    try:
                        m2 = RELOAD_SENTINEL.stat().st_mtime
                    except OSError:
                        m2 = m
                    if m2 <= _sentinel_mtime:
                        continue
                    _sentinel_mtime = m2
                    log.info("web: reload sentinel advanced; reloading index")
                    try:
                        await asyncio.to_thread(_ensure_loaded)
                        ntotal = search_mod._INDEX.ntotal if search_mod._INDEX else 0
                        log.info("web: index reloaded OK (ntotal=%d)", ntotal)
                    except Exception as e:
                        log.exception("web: reload failed: %s", e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("web: reload watcher error: %s", e)
            await asyncio.sleep(RELOAD_POLL_S)


@app.on_event("startup")
async def _warm_up() -> None:
    global _sentinel_task
    try:
        await asyncio.to_thread(_ensure_loaded)
    except BaseException as e:
        log.exception("startup warmup failed: %s", e)
        raise
    _sentinel_task = asyncio.create_task(_reload_watcher())


@app.on_event("shutdown")
async def _shut_down() -> None:
    if _sentinel_task and not _sentinel_task.done():
        _sentinel_task.cancel()
        try:
            await _sentinel_task
        except (asyncio.CancelledError, Exception):
            pass


# ── auth ────────────────────────────────────────────────────────────────────
def _require_auth(request: Request) -> None:
    """HTTP basic auth via Authorization header.

    Configured via PHOTOINDEX_USER + PHOTOINDEX_PASS env vars.

    A2 fix: previously, when the env vars were unset, this function
    silently returned (no auth) — which matched the plan's "auth is
    OFF by default" decision, but the code comment in config.py
    claimed the endpoint would 503. The behavior was correct, the
    docs were wrong. The default-off decision is preserved for
    *reads*; writes (POST /api/upload) now always require auth, with
    a 503 if env is unset (so an operator who hasn't configured auth
    is forced to opt-in rather than accidentally exposing writes).
    """
    if not (BASIC_AUTH_USER and BASIC_AUTH_PASS):
        # Write paths enforce: deny if auth is unconfigured. Read paths
        # (caller should skip this check) opt out via _require_auth_opt.
        if request.url.path.startswith("/api/upload"):
            raise HTTPException(
                503,
                "auth not configured: set PHOTOINDEX_USER and PHOTOINDEX_PASS in "
                "set PHOTOINDEX_USER and PHOTOINDEX_PASS environment variables to enable uploads",
                headers={"WWW-Authenticate": "Basic"},
            )
        return
    import base64
    h = request.headers.get("Authorization", "")
    if not h.startswith("Basic "):
        raise HTTPException(401, "auth required", headers={"WWW-Authenticate": "Basic"})
    try:
        decoded = base64.b64decode(h.split(" ", 1)[1], validate=True).decode("utf-8", "replace")
    except Exception:
        raise HTTPException(401, "bad auth", headers={"WWW-Authenticate": "Basic"})
    if ":" not in decoded:
        raise HTTPException(401, "bad auth", headers={"WWW-Authenticate": "Basic"})
    user, _, pwd = decoded.partition(":")
    import hmac
    if not hmac.compare_digest(user, BASIC_AUTH_USER) or not hmac.compare_digest(pwd, BASIC_AUTH_PASS):
        raise HTTPException(401, "bad creds", headers={"WWW-Authenticate": "Basic"})


# ── helpers ────────────────────────────────────────────────────────────────
def _image_url(name: str) -> str:
    return f"/image/{name}"


def _do_search(query: str, top_k: int) -> list[dict]:
    if not query.strip():
        return []
    try:
        raw = search_mod.search(query, top_k=int(top_k))
    except Exception:
        log.exception("search failed for query=%r top_k=%d", query, top_k)
        return []
    out = []
    for i, r in enumerate(raw, 1):
        out.append(
            {
                "rank": i,
                "name": r["name"],
                "score": r["score"],
                "chars": r.get("chars", 0),
                "image_url": _image_url(r["name"]),
                "snippet": r.get("snippet", "")[:2000],
            }
        )
    return out


# ── routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return _TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": [],
            "query": "",
            "default_k": DEFAULT_TOP_K,
            "max_k": MAX_TOP_K,
            "max_files": MAX_UPLOAD_FILES,
            "max_bytes": MAX_UPLOAD_BYTES,
        },
    )


@app.post("/search", response_class=HTMLResponse)
def search_post(
    request: Request,
    q: str = Form("", max_length=512),
    k: int = Form(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
):
    results = _do_search(q, top_k=k)
    return _TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results,
            "query": q,
            "default_k": DEFAULT_TOP_K,
            "max_k": MAX_TOP_K,
            "max_files": MAX_UPLOAD_FILES,
            "max_bytes": MAX_UPLOAD_BYTES,
        },
    )


@app.get("/api/search")
def api_search(
    q: str = Query(..., min_length=1, max_length=512),
    k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
):
    return {"query": q, "results": _do_search(q, k)}


@app.get("/api/health")
def api_health():
    n = 0
    try:
        if search_mod._INDEX is not None:
            n = search_mod._INDEX.ntotal
    except Exception:
        pass
    depth = 0
    redis_ok = False
    try:
        depth = tasks.queue_depth()
        redis_ok = True
    except Exception as e:
        log.warning("health: redis unavailable: %s", e)
    return {
        "status": "ok",
        "indexed": n,
        "queue_depth": depth,
        "redis": "ok" if redis_ok else "down",
    }


@app.post("/api/upload")
async def api_upload(
    request: Request,
    files: list[UploadFile] = File(...),
):
    """Accept up to MAX_UPLOAD_FILES files, up to MAX_UPLOAD_BYTES total.
    Returns 202 + {job_id, count, bytes} on success.
    """
    _require_auth(request)

    if not files:
        raise HTTPException(400, "no files in request")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(413, f"too many files (limit {MAX_UPLOAD_FILES})")

    try:
        file_triples = [(f.filename, f.content_type, f.file) for f in files]
        job_id, count, total_bytes, names = await asyncio.to_thread(
            tasks.stage_uploads, file_triples
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("upload: staging failed: %s", e)
        raise HTTPException(500, "staging failed")

    try:
        await asyncio.to_thread(tasks.push_job, job_id, names)
    except Exception as e:
        # Roll back the staged files if we can't enqueue.
        tasks.cleanup_job_dir(job_id)
        log.exception("upload: push_job failed: %s", e)
        raise HTTPException(503, "broker unavailable")

    log.info("upload: job=%s files=%d bytes=%d", job_id, count, total_bytes)
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "count": count,
            "bytes": total_bytes,
            "max_files": MAX_UPLOAD_FILES,
            "max_bytes": MAX_UPLOAD_BYTES,
        },
    )


@app.get("/api/upload/{job_id}")
async def api_upload_status(request: Request, job_id: str):
    # A32 fix: GET status endpoint was missing _require_auth when auth is
    # enabled, leaking job metadata to unauthenticated callers. Symmetrize
    # with POST /api/upload so the entire upload surface is gated.
    _require_auth(request)
    j = await asyncio.to_thread(tasks.get_job, job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


@app.delete("/api/image/{name:path}")
def api_delete_image(request: Request, name: str):
    """Delete an image and all its artifacts (image, OCR, manifest, index)."""
    _require_auth(request)

    img_path = (IMAGES_DIR / name).resolve()
    try:
        img_path.relative_to(IMAGES_DIR.resolve())
    except ValueError:
        raise HTTPException(404, "not found")

    if not img_path.is_file():
        raise HTTPException(404, f"image not found: {name}")

    # Load manifest, remove entry, save.
    import json
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {}
    if name not in manifest:
        raise HTTPException(404, f"image not in manifest: {name}")

    ocr_path = Path(manifest[name].get("ocr_path", ""))
    del manifest[name]
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Delete image + OCR files.
    import os
    for p in (img_path, ocr_path):
        if p and p.is_file():
            try:
                p.unlink()
            except OSError as e:
                log.warning("delete failed %s: %s", p, e)

    # Rebuild FAISS + BM25 from remaining manifest entries.
    _rebuild_index(manifest)

    # Touch sentinel so web worker reloads.
    RELOAD_SENTINEL.write_text("", encoding="utf-8")
    return {"ok": True, "deleted": name, "remaining": len(manifest)}


def _rebuild_index(manifest: dict) -> None:
    """Rebuild FAISS + BM25 from manifest entries (runs in-process)."""
    import faiss
    import hashlib
    import io
    import json as _json
    import numpy as np
    import os
    import pickle
    import re
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    from config import EMBED_BATCH, EMBED_DIM, EMBED_MODEL

    names, texts, tokens = [], [], []
    for name, meta in manifest.items():
        text_path = Path(meta.get("ocr_path", ""))
        if not text_path.exists() or meta.get("blank"):
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        names.append(name)
        texts.append(text[:20000])
        import re
        tokens.append([w.lower() for w in re.findall(r"[А-Яа-яёЁA-Za-z0-9]+", text) if len(w) > 1])  # noqa: E501

    if not names:
        # Empty index — remove artifacts.
        for p in (FAISS_PATH, EMBEDDINGS_PATH, INDEX_DIR / "bm25.pkl",
                  INDEX_DIR / "bm25.pkl.sig", INDEX_DIR / "index_meta.json"):
            if p.exists():
                p.unlink()
        log.info("rebuild: empty manifest, cleared index")
        return

    model = SentenceTransformer(EMBED_MODEL)
    dense = model.encode(
        texts, batch_size=EMBED_BATCH, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype("float32")
    dim = dense.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(dense)

    # Atomic writes.
    def _atomic_write(path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    faiss_buf = faiss.VectorIOWriter()
    faiss.write_index(index, faiss_buf)
    _atomic_write(FAISS_PATH, faiss.vector_to_array(faiss_buf.data).tobytes())

    emb_buf = io.BytesIO()
    np.save(emb_buf, dense, allow_pickle=False)
    _atomic_write(EMBEDDINGS_PATH, emb_buf.getvalue())

    bm25 = BM25Okapi(tokens)
    payload = pickle.dumps(
        {"bm25": bm25, "names": names, "token_lists": tokens},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    _atomic_write(INDEX_DIR / "bm25.pkl", payload)
    (INDEX_DIR / "bm25.pkl.sig").write_text(
        hashlib.sha256(payload).hexdigest(), encoding="utf-8"
    )

    meta = {"model": EMBED_MODEL, "dim": dim, "count": len(names),
            "names": names, "has_bm25": True}
    meta_path = INDEX_DIR / "index_meta.json"
    meta_tmp = meta_path.with_suffix(".json.tmp")
    meta_tmp.write_text(_json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(meta_tmp, meta_path)

    log.info("rebuild: %d vectors, dim=%d", len(names), dim)


@app.get("/image/{name:path}")
def image(name: str):
    from fastapi.responses import FileResponse
    p = (IMAGES_DIR / name).resolve()
    try:
        p.relative_to(IMAGES_DIR.resolve())
    except ValueError:
        log.warning("path traversal blocked: %s -> %s", name, p)
        raise HTTPException(404, "image not found")
    if not p.is_file():
        raise HTTPException(404, f"image not found: {name}")
    return FileResponse(p, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    from config import WEB_HOST, WEB_PORT
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level="info")
