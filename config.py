"""Configuration for photo-index."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

ROOT = Path(os.environ.get("PHOTOINDEX_ROOT", Path(__file__).resolve().parent))
# PHOTOINDEX_DATA lets operators override the data directory without the
# implicit "ROOT/data" wrapping. Defaults to the historical layout so bare
# metal users are unaffected. The Docker image sets this to /data so the
# ./data:/data volume mount lands images at ./data/images/ on the host
# (not the surprising ./data/data/images/ that PHOTOINDEX_ROOT=/data would
# otherwise produce).
DATA = Path(os.environ.get("PHOTOINDEX_DATA", ROOT / "data"))
IMAGES_DIR = DATA / "images"
OCR_DIR = DATA / "ocr"
INDEX_DIR = DATA / "index"
STAGING_DIR = DATA / "staging"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
FAISS_PATH = INDEX_DIR / "faiss.index"

# Sentinel file: worker touches this after a successful reindex; the web
# process polls its mtime every few seconds and reloads the index.
RELOAD_SENTINEL = INDEX_DIR / ".reload"

OCR_LANGS = os.environ.get("OCR_LANGS", "rus+eng")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = 1024
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "8"))

WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "7860"))

# Upload limits (per request, enforced in the web layer).
MAX_UPLOAD_FILES = int(os.environ.get("MAX_UPLOAD_FILES", "300"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
UPLOAD_ACCEPT_MIME = frozenset({
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/bmp", "image/tiff", "image/heic", "image/heif",
})
UPLOAD_ACCEPT_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
    ".heic", ".heif",
})

# Top-k cap. Client may request 1..MAX_TOP_K.
MAX_TOP_K = int(os.environ.get("MAX_TOP_K", "25"))
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "5"))

# Message broker (Redis). Worker and web both connect.
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
JOB_QUEUE = "photoindex:jobs"
JOB_HASH_PREFIX = "photoindex:job:"

# Web reload polling interval (seconds).
RELOAD_POLL_S = float(os.environ.get("RELOAD_POLL_S", "5"))

# Basic auth. Defaults to admin/admin so the modal is enabled out of the
# box. Set PHOTOINDEX_USER="" AND PHOTOINDEX_PASS="" together to disable
# auth entirely (the modal is then not served and all endpoints become
# public). The user is fixed at "admin" — there is exactly one account.
BASIC_AUTH_USER = os.environ.get("PHOTOINDEX_USER", "admin")
BASIC_AUTH_PASS = os.environ.get("PHOTOINDEX_PASS", "admin")

# Session cookie. The signing key is generated on first start and
# persisted to SESSION_KEY_PATH (mode 600). Backup this file if you
# want to preserve sessions across container recreates; deleting it
# invalidates all existing session cookies.
SESSION_COOKIE_NAME = "photoindex_session"
SESSION_KEY_PATH = DATA / ".session_key"
SESSION_TTL_DAYS = int(os.environ.get("PHOTOINDEX_SESSION_TTL_DAYS", "7"))
# Set PHOTOINDEX_SESSION_SECURE=1 to require HTTPS for the cookie.
SESSION_COOKIE_SECURE = os.environ.get("PHOTOINDEX_SESSION_SECURE", "0") == "1"


def _load_or_create_session_key() -> bytes:
    """Read the persistent HMAC key, or create one if it does not exist.

    Persisted to SESSION_KEY_PATH (mode 600) inside PHOTOINDEX_ROOT.
    """
    try:
        if SESSION_KEY_PATH.is_file():
            data = SESSION_KEY_PATH.read_bytes()
            if len(data) >= 32:
                return data
    except OSError:
        pass
    key = secrets.token_bytes(32)
    try:
        SESSION_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SESSION_KEY_PATH.write_bytes(key)
        try:
            os.chmod(SESSION_KEY_PATH, 0o600)
        except OSError:
            pass
    except OSError:
        # Best-effort: if we cannot persist, fall back to an in-process
        # key. Sessions will be invalidated on restart.
        return key
    return key


SESSION_KEY = _load_or_create_session_key()

for d in (IMAGES_DIR, OCR_DIR, INDEX_DIR, STAGING_DIR):
    d.mkdir(parents=True, exist_ok=True)
