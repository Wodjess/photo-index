"""Configuration for photo-index."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("PHOTOINDEX_ROOT", Path(__file__).resolve().parent))
DATA = ROOT / "data"
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

# Basic auth. Set PHOTOINDEX_USER + PHOTOINDEX_PASS to enable. If unset,
# web returns 503 on /api/upload (writes are not exposed) but reading
# endpoints still work for backwards compat.
BASIC_AUTH_USER = os.environ.get("PHOTOINDEX_USER", "")
BASIC_AUTH_PASS = os.environ.get("PHOTOINDEX_PASS", "")

for d in (IMAGES_DIR, OCR_DIR, INDEX_DIR, STAGING_DIR):
    d.mkdir(parents=True, exist_ok=True)
