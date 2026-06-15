"""Hybrid search: dense (bge-m3 via FAISS) + sparse (BM25) merged via RRF.

Reciprocal Rank Fusion: score(d) = sum(1 / (k0 + rank_d)) for each ranker.
Tunable: K_DENSE, K_SPARSE (top-N from each), K0 (RRF constant, default 60).

The model, FAISS index, and BM25 are loaded once at module level via
`ensure_loaded()` (called from web.py startup). Search calls reuse the
module-level singletons.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBED_DIM,
    EMBED_MODEL,
    FAISS_PATH,
    INDEX_DIR,
    OCR_DIR,
)
import ocr as ocr_mod

log = logging.getLogger("search")

# RRF configuration
K_DENSE = 50
K_SPARSE = 15
K0 = 60.0

_TOKEN_RE = re.compile(r"[А-Яа-яёЁA-Za-z0-9]+")

# Module-level caches (A2 + A3 fix: no reloads)
_INDEX: Optional[faiss.Index] = None
_META: Optional[dict] = None
_MANIFEST: Optional[dict] = None
_MODEL: Optional[SentenceTransformer] = None
_BM25 = None  # tuple (bm25_obj, names) or None


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text) if len(w) > 1]


def _load_bm25():
    global _BM25
    if _BM25 is not None:
        return _BM25
    pkl_path = INDEX_DIR / "bm25.pkl"
    sig_path = INDEX_DIR / "bm25.pkl.sig"
    if not pkl_path.exists():
        return None
    import hashlib
    import pickle
    with open(pkl_path, "rb") as f:
        payload = f.read()
    # A9 fix: integrity check. If the sig file is missing (older index run)
    # we refuse to unpickle rather than silently accept a tampered file.
    if not sig_path.exists():
        log.warning(
            "missing %s — skipping BM25 load (sparse search disabled). "
            "Re-run `python index.py` to regenerate the BM25 index with an integrity signature.",
            sig_path,
        )
        _BM25 = None
        return None
    expected = sig_path.read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(payload).hexdigest()
    if expected != actual:
        log.warning(
            "BM25 pickle signature mismatch: expected %s…, got %s… — "
            "skipping BM25 load (sparse search disabled). Re-run `python index.py`.",
            expected[:12],
            actual[:12],
        )
        _BM25 = None
        return None
    data = pickle.loads(payload)
    _BM25 = (data["bm25"], data["names"])
    return _BM25


def _load_index_from_disk():
    """Read FAISS + meta + manifest from disk.

    Returns an empty corpus (no images indexed) on a fresh deploy —
    the service must boot even with zero indexed images. The first
    upload through /api/upload triggers the worker to build the
    index, and the sentinel-mtime poll in web.py will reload it.
    """
    if not FAISS_PATH.exists():
        log.info("no FAISS index at %s — starting with empty corpus", FAISS_PATH)
        return None, {"names": [], "count": 0, "dim": EMBED_DIM, "has_bm25": False}, {}
    index = faiss.read_index(str(FAISS_PATH))
    meta = json.loads((FAISS_PATH.parent / "index_meta.json").read_text(encoding="utf-8"))
    manifest = ocr_mod.load_manifest()
    return index, meta, manifest


def ensure_loaded() -> None:
    """Idempotent loader for index, BM25, and embedding model."""
    global _INDEX, _META, _MANIFEST, _MODEL
    if _INDEX is None:
        log.info("loading FAISS + meta + manifest from %s", FAISS_PATH)
        _INDEX, _META, _MANIFEST = _load_index_from_disk()
    if _BM25 is None:
        log.info("loading BM25 from %s/bm25.pkl", INDEX_DIR)
        _load_bm25()
    if _MODEL is None:
        log.info("loading embed model %s", EMBED_MODEL)
        _MODEL = SentenceTransformer(EMBED_MODEL)
    n = _INDEX.ntotal if _INDEX is not None else 0
    log.info("ensure_loaded: index=%d, bm25=%s, model=%s",
             n, _BM25 is not None, EMBED_MODEL)



def reload() -> None:
    """Force-reload the in-memory index from disk.

    Called by the web process after the worker touches the .reload
    sentinel (i.e. a successful reindex). The FAISS index is read
    into memory once; the BM25 pickle is re-read and re-verified
    against its SHA256 sidecar. The embed model is kept in memory
    between reloads because it is the slowest thing to load.
    """
    global _INDEX, _META, _MANIFEST, _BM25
    log.info("search.reload: dropping in-memory index/BM25 caches")
    _INDEX = None
    _META = None
    _MANIFEST = None
    _BM25 = None
    ensure_loaded()
def load_index():
    """Backwards-compat shim for CLI — same effect as ensure_loaded()."""
    ensure_loaded()
    return _INDEX, _META, _MANIFEST


def _make_snippet(text: str, query: str, head_len: int = 400, window: int = 250) -> str:
    """A6 fix: center snippet on the first query-token occurrence.
    Falls back to the first `head_len` chars if no token is found.
    """
    q_tokens = tokenize(query)
    if not q_tokens or not text:
        return text[:head_len].replace("\n", " ")
    text_lower = text.lower()
    best_idx = -1
    for tok in q_tokens:
        i = text_lower.find(tok)
        if i >= 0 and (best_idx < 0 or i < best_idx):
            best_idx = i
            break
    if best_idx < 0:
        return text[:head_len].replace("\n", " ")
    start = max(0, best_idx - window // 2)
    end = min(len(text), start + head_len)
    snippet = text[start:end].replace("\n", " ")
    if start > 0:
        snippet = "… " + snippet
    if end < len(text):
        snippet = snippet + " …"
    return snippet


def _rrf_merge(dense_ranks: list[int], sparse_ranks: list[int], k0: float) -> dict[int, float]:
    scores: dict[int, float] = {}
    for r, idx in enumerate(dense_ranks):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k0 + r)
    for r, idx in enumerate(sparse_ranks):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k0 + r)
    return scores


def search(query: str, top_k: int = 5) -> list[dict]:
    ensure_loaded()
    # Empty corpus (fresh deploy, no uploads yet): the embed model
    # is still loaded so the user gets a 200 with an empty result set
    # rather than a 500. The first upload will trigger the worker to
    # build the index, and the sentinel-mtime poll in web.py will
    # reload it for subsequent queries.
    if _INDEX is None or _META is None or _MANIFEST is None or _MODEL is None:
        log.info("search: empty corpus, returning [] for query=%r", query)
        return []
    names = _META["names"]

    # A5 fix: clamp dense k to ntotal.
    if _INDEX.ntotal == 0:
        return []
    k_dense = min(K_DENSE, _INDEX.ntotal)

    # Dense
    q_emb = _MODEL.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    _, dense_ids = _INDEX.search(q_emb, k_dense)
    dense_ranks = [int(i) for i in dense_ids[0] if i >= 0]

    # Sparse
    sparse_ranks: list[int] = []
    if _BM25 is not None:
        bm25_obj, _ = _BM25
        q_tokens = tokenize(query)
        if q_tokens:
            scores = bm25_obj.get_scores(q_tokens)
            order = np.argsort(scores)[::-1][:K_SPARSE]
            sparse_ranks = [int(i) for i in order if scores[i] > 0]

    # RRF merge
    rrf = _rrf_merge(dense_ranks, sparse_ranks, K0)
    sorted_idx = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]

    out = []
    for idx, rrf_score in sorted_idx:
        if idx < 0 or idx >= len(names):
            continue
        name = names[idx]
        entry = _MANIFEST.get(name)
        if entry is None:
            log.warning("manifest desync: name %r not in manifest, skipping", name)
            continue
        ocr_path = Path(entry["ocr_path"])
        # A17 fix: read_text in try/except, fall back to "" on any error.
        try:
            text = ocr_path.read_text(encoding="utf-8", errors="replace") if ocr_path.exists() else ""
        except OSError as e:
            log.warning("OCR read failed for %s: %s", name, e)
            text = ""
        out.append(
            {
                "name": name,
                "path": entry["path"],
                "score": float(rrf_score),
                "chars": entry.get("chars", 0),
                "snippet": _make_snippet(text, query),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="+", help="search query")
    p.add_argument("--top-k", "-k", type=int, default=5)
    args = p.parse_args()
    query = " ".join(args.query)
    results = search(query, top_k=args.top_k)
    if not results:
        print("no results")
        return
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] rrf={r['score']:.4f}  chars={r['chars']}  {r['name']}")
        print(f"    {r['path']}")
        if r["snippet"]:
            print(f"    {r['snippet']}")


if __name__ == "__main__":
    main()
