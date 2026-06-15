"""Web UI for the photo-index semantic search.

Bound to 0.0.0.0:7860 by default (see WEB_HOST / WEB_PORT env vars).
SSR via Jinja2 for the initial paint; the front-end then drives
/api/search from JS so the 5-card grid + cinema mode feel instant.

Auth (single user, no registration):
  - Defaults: PHOTOINDEX_USER=admin, PHOTOINDEX_PASS=admin.
  - Set both to "" to disable auth entirely (legacy behaviour).
  - Frontend gates both search and upload behind a login modal that
    verifies the session cookie set by /api/login.
  - Direct API consumers can still use HTTP Basic auth.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
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
from starlette.responses import JSONResponse as StarletteJSONResponse

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
    REDIS_URL,
    RELOAD_POLL_S,
    RELOAD_SENTINEL,
    DEFAULT_TOP_K,
    SESSION_COOKIE_NAME,
    SESSION_COOKIE_SECURE,
    SESSION_KEY,
    SESSION_TTL_DAYS,
    WEB_HOST,
    WEB_PORT,
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


# ── request logging middleware ─────────────────────────────────────────────
@app.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    log.info(
        "web: %s %s -> %s (%.0fms)",
        request.method,
        request.url.path + (("?" + request.url.query) if request.url.query else ""),
        response.status_code,
        duration_ms,
    )
    return response


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

    # Startup diagnostics
    ntotal = search_mod._INDEX.ntotal if search_mod._INDEX else 0
    log.info("startup: REDIS_URL=%s", REDIS_URL)
    log.info("startup: WEB_HOST=%s WEB_PORT=%d", WEB_HOST, WEB_PORT)
    log.info("startup: auth_enabled=%s", bool(BASIC_AUTH_USER))
    log.info("startup: index ntotal=%d", ntotal)


@app.on_event("shutdown")
async def _shut_down() -> None:
    if _sentinel_task and not _sentinel_task.done():
        _sentinel_task.cancel()
        try:
            await _sentinel_task
        except (asyncio.CancelledError, Exception):
            pass


# ── auth ────────────────────────────────────────────────────────────────────
AUTH_ENABLED = bool(BASIC_AUTH_USER) and bool(BASIC_AUTH_PASS)


def _b64u_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64u_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _issue_session_cookie(user: str) -> tuple[str, int]:
    """Build a signed session token. Returns (cookie_value, expires_at_epoch)."""
    expires = int(time.time()) + SESSION_TTL_DAYS * 86400
    payload = json.dumps({"u": user, "exp": expires}, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64u_encode(payload)
    sig = hmac.new(SESSION_KEY, payload_b64.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64u_encode(sig)}", expires


def _verify_session_cookie(token: str) -> str | None:
    """Return the username if the token is valid and unexpired, else None."""
    if not token or "." not in token:
        return None
    payload_b64, _, sig_b64 = token.partition(".")
    try:
        expected_sig = hmac.new(SESSION_KEY, payload_b64.encode("ascii"), hashlib.sha256).digest()
        actual_sig = _b64u_decode(sig_b64)
    except Exception:
        return None
    if not hmac.compare_digest(expected_sig, actual_sig):
        return None
    try:
        payload = json.loads(_b64u_decode(payload_b64).decode("utf-8"))
    except Exception:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    user = payload.get("u")
    if not isinstance(user, str) or not user:
        return None
    # Defense-in-depth: never trust a payload field outside of what the
    # current auth config declares as valid. The HMAC already prevents
    # forgery, so this is a guardrail for any future multi-user reuse
    # of the signing key.
    if not hmac.compare_digest(user, BASIC_AUTH_USER):
        return None
    return user


def _set_session_cookie(response: JSONResponse, user: str) -> None:
    value, expires = _issue_session_cookie(user)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=value,
        max_age=SESSION_TTL_DAYS * 86400,
        expires=expires,
        path="/",
        httponly=True,
        samesite="lax",
        secure=SESSION_COOKIE_SECURE,
    )


def _clear_session_cookie(response: JSONResponse) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=SESSION_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )


def _require_auth(request: Request) -> str | None:
    """Authenticate the request. Returns the username or raises 401.

    Auth sources, in order:
      1. The signed HttpOnly session cookie set by /api/login.
      2. HTTP Basic auth header (for direct API consumers and scripts).
    Returns None immediately if auth is disabled (one or both env vars
    empty). To re-enable the original "no auth, no problem" behaviour
    (uploads public), set both PHOTOINDEX_USER and PHOTOINDEX_PASS to "".
    """
    if not AUTH_ENABLED:
        return None

    # 1) Cookie auth (browser modal flow).
    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie:
        user = _verify_session_cookie(cookie)
        if user is not None:
            return user
        # Bad cookie — fall through to basic, then 401 below.

    # 2) HTTP Basic auth (curl, scripts, direct API consumers).
    h = request.headers.get("Authorization", "")
    if h.startswith("Basic "):
        try:
            decoded = base64.b64decode(h.split(" ", 1)[1]).decode("utf-8", "replace")
        except Exception:
            decoded = ""
        if ":" in decoded:
            user, _, pwd = decoded.partition(":")
            if hmac.compare_digest(user, BASIC_AUTH_USER) and hmac.compare_digest(pwd, BASIC_AUTH_PASS):
                return user

    raise HTTPException(401, "auth required", headers={"WWW-Authenticate": "Basic"})


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
def _is_authed(request: Request) -> bool:
    """Return True if the request carries a valid session cookie."""
    if not AUTH_ENABLED:
        return False
    return _verify_session_cookie(request.cookies.get(SESSION_COOKIE_NAME) or "") is not None


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
            "auth_enabled": AUTH_ENABLED,
            "authed": _is_authed(request),
        },
    )


@app.post("/search", response_class=HTMLResponse)
def search_post(
    request: Request,
    q: str = Form("", max_length=512),
    k: int = Form(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
):
    _require_auth(request)
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
            "auth_enabled": AUTH_ENABLED,
            "authed": _is_authed(request),
        },
    )


@app.get("/api/search")
def api_search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=512),
    k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
):
    _require_auth(request)
    return {"query": q, "results": _do_search(q, k)}


# ── auth endpoints ──────────────────────────────────────────────────────────
@app.post("/api/login")
async def api_login(request: Request):
    """Verify username + password, then set the session cookie.

    Accepts both JSON (`{"user": "...", "pass": "..."}`) and form-encoded
    (`user=...&pass=...`) bodies. The form-encoded path is the no-JS
    fallback: the SSR `<form>` posts natively to /api/login with the
    default `application/x-www-form-urlencoded` content type. The JS
    modal uses JSON via `fetch()` to avoid a page reload.

    The frontend never sees the cookie value directly (it is HttpOnly);
    it only needs the 200 vs 401 response to know it can unlock the UI.
    """
    if not AUTH_ENABLED:
        return JSONResponse(
            status_code=400,
            content={"detail": "auth is disabled on this server"},
        )
    content_type = (request.headers.get("content-type") or "").lower()
    creds: dict = {}
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            creds = body
    elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception:
            form = None
        if form is not None:
            creds = {k: form.get(k) for k in ("user", "pass")}
    else:
        # Unknown content type. Try JSON first, then form. This is a
        # defensive fallback for clients that omit the content-type header.
        try:
            body = await request.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            creds = body
        else:
            try:
                form = await request.form()
            except Exception:
                form = None
            if form is not None:
                creds = {k: form.get(k) for k in ("user", "pass")}
    user = creds.get("user", "") or ""
    pwd  = creds.get("pass", "")  or ""
    if not isinstance(user, str) or not isinstance(pwd, str):
        return JSONResponse(status_code=400, content={"detail": "user and pass must be strings"})
    if not (hmac.compare_digest(user, BASIC_AUTH_USER) and hmac.compare_digest(pwd, BASIC_AUTH_PASS)):
        # Identical message for unknown user vs wrong password to avoid
        # user-enumeration leaks. We do log the attempt so the operator
        # can spot brute-force scans after the fact.
        client = request.client
        log.warning("auth: failed login for user=%r from %s", user, client.host if client else "?")
        return JSONResponse(status_code=401, content={"detail": "bad creds"})
    resp = JSONResponse(content={"user": user, "ok": True})
    _set_session_cookie(resp, user)
    log.info("auth: login ok user=%r from %s", user, request.client.host if request.client else "?")
    return resp


@app.post("/api/logout")
async def api_logout():
    """Clear the session cookie. Idempotent — works even if no cookie was set."""
    resp = JSONResponse(content={"ok": True})
    _clear_session_cookie(resp)
    return resp


@app.get("/api/whoami")
def api_whoami(request: Request):
    """Return the current user if a valid session cookie is present.

    Always returns 401 when auth is enabled and the caller is not signed
    in. When auth is disabled, returns 200 with `auth_enabled: false`
    so the frontend can skip showing the modal.
    """
    if not AUTH_ENABLED:
        return {"auth_enabled": False, "user": None}
    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    user = _verify_session_cookie(cookie or "")
    if not user:
        raise HTTPException(401, "auth required")
    # Decode the exp to surface a friendly hint to the UI.
    exp = 0
    if cookie:
        try:
            payload_b64 = cookie.split(".", 1)[0]
            payload = json.loads(_b64u_decode(payload_b64).decode("utf-8"))
            exp = int(payload.get("exp", 0))
        except Exception:
            exp = 0
    return {"auth_enabled": True, "user": user, "expires_at": exp}


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
    body = {
        "status": "ok" if redis_ok else "degraded",
        "indexed": n,
        "queue_depth": depth,
        "redis": "ok" if redis_ok else "down",
    }
    if not redis_ok:
        return StarletteJSONResponse(content=body, status_code=503)
    return body


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

    # Atomic manifest write.
    manifest_data = json.dumps(manifest, ensure_ascii=False, indent=2)
    manifest_tmp = MANIFEST_PATH.with_suffix(MANIFEST_PATH.suffix + ".tmp")
    manifest_tmp.write_text(manifest_data, encoding="utf-8")
    os.replace(manifest_tmp, MANIFEST_PATH)

    # Rebuild FAISS + BM25 from remaining manifest entries.
    _rebuild_index(manifest)

    # Delete image + OCR files only after successful rebuild.
    for p in (img_path, ocr_path):
        if p and p.is_file():
            try:
                p.unlink()
            except OSError as e:
                log.warning("delete failed %s: %s", p, e)

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
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level="info")
