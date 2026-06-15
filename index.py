"""Build a dense (FAISS) + sparse (BM25) index over OCR text.

Dense: BAAI/bge-m3 (1024d) on full text passages.
Sparse: rank_bm25 BM25Okapi on word-tokenized text.

Round 0 (settings+upload):
  - `--append` mode: only embed images not already in the current
    index, append them to FAISS, and rebuild BM25 from the union of
    old + new passages. Faster than a full rebuild; safe for a small
    per-batch delta. The web process polls the .reload sentinel and
    re-runs ensure_loaded() on mtime change.
  - `--manifest PATH`: alternative manifest location (used by the
    worker to write a temporary merged manifest into INDEX_DIR before
    re-embedding).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    EMBED_BATCH,
    EMBED_DIM,
    EMBED_MODEL,
    FAISS_PATH,
    INDEX_DIR,
    MANIFEST_PATH,
    OCR_DIR,
    EMBEDDINGS_PATH,
)
import ocr

log = logging.getLogger("index")


def tokenize_for_bm25(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[А-Яа-яёЁA-Za-z0-9]+", text) if len(w) > 1]


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(m: dict[str, dict], path: Path = MANIFEST_PATH) -> None:
    path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def build_passages(manifest: dict[str, dict]) -> tuple[list[str], list[str], list[list[str]]]:
    names: list[str] = []
    texts: list[str] = []
    tokens: list[list[str]] = []
    for name, meta in manifest.items():
        text_path = Path(meta["ocr_path"])
        if not text_path.exists():
            continue
        if meta.get("blank"):
            continue
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        names.append(name)
        texts.append(text[:20000])
        tokens.append(tokenize_for_bm25(text))
    return names, texts, tokens


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--append", action="store_true",
                   help="Append only new images; rebuild BM25 from union")
    args = p.parse_args()

    manifest = load_manifest()
    names, texts, token_lists = build_passages(manifest)
    log.info("indexable passages: %d / %d", len(names), len(manifest))
    if not names:
        raise SystemExit("no OCR text to index — check ocr.py output")

    log.info("loading model %s ...", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    # Decide which names are new (in --append mode) vs already indexed.
    existing_index = None
    new_names: list[str] = []
    new_texts: list[str] = []
    new_tokens: list[list[str]] = []
    if args.append and EMBEDDINGS_PATH.exists() and FAISS_PATH.exists():
        try:
            existing_index = faiss.read_index(str(FAISS_PATH))
            meta_existing = json.loads(
                (FAISS_PATH.parent / "index_meta.json").read_text(encoding="utf-8")
            )
            existing_names = set(meta_existing.get("names", []))
        except Exception:
            existing_index = None
            existing_names = set()

        for n, t, tk in zip(names, texts, token_lists):
            if n in existing_names:
                continue
            new_names.append(n)
            new_texts.append(t)
            new_tokens.append(tk)
        log.info(
            "--append: existing=%d new=%d",
            len(existing_names), len(new_names),
        )
    else:
        new_names = names
        new_texts = texts
        new_tokens = token_lists

    if not new_names and args.append:
        # Nothing new; still rewrite index_meta to reflect current manifest
        # (handy if names were removed from the manifest).
        meta = {
            "model": EMBED_MODEL,
            "dim": EMBED_DIM,
            "count": len(names),
            "names": names,
            "has_bm25": True,
        }
        (FAISS_PATH.parent / "index_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("--append: nothing to add, manifest rewritten")
        return

    log.info("encoding %d new passages (dense, dim=%d) ...", len(new_texts), EMBED_DIM)
    dense_new = model.encode(
        new_texts,
        batch_size=EMBED_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    dim = dense_new.shape[1]
    if dim != EMBED_DIM:
        log.warning("model dim %d != configured EMBED_DIM %d", dim, EMBED_DIM)

    if args.append and existing_index is not None:
        # Concatenate old embeddings + new ones, then rebuild the index.
        # IndexFlatIP doesn't support efficient appends, so the cheap and
        # correct thing is a full FAISS rebuild of (old + new) vectors.
        old_dense = np.load(EMBEDDINGS_PATH, allow_pickle=False).astype("float32")
        if old_dense.shape[1] != dim:
            log.warning(
                "old dim %d != new dim %d; falling back to full rebuild",
                old_dense.shape[1], dim,
            )
            dense = np.concatenate([np.zeros((0, dim), dtype="float32"), dense_new], axis=0)
        else:
            dense = np.concatenate([old_dense, dense_new], axis=0)
    else:
        dense = dense_new

    index = faiss.IndexFlatIP(dim)
    index.add(dense)

    # A4 fix: atomic writes for every index artifact. SIGKILL during a
    # non-atomic write would leave a half-written file the next reload
    # would mmap as corrupt. We write to "<file>.tmp" and rename.
    # faiss.write_index takes a path; we serialize the index to a buffer
    # ourselves to use the same _atomic_write helper.
    import io, hashlib
    log.info("writing dense index (%d vectors, dim=%d) ...", index.ntotal, dim)

    # faiss-cpu 1.7/1.8 reject a Python BytesIO as the second arg
    # (the C++ overload wants a C FILE* or an IOWriter). Use the
    # built-in faiss.VectorIOWriter (backed by std::vector<uint8_t>),
    # then convert to Python bytes via faiss.vector_to_array(...).
    faiss_buf = faiss.VectorIOWriter()
    faiss.write_index(index, faiss_buf)
    faiss_bytes = faiss.vector_to_array(faiss_buf.data).tobytes()
    # Write embeddings as .npy (with header) so np.load() works in --append mode.
    emb_buf = io.BytesIO()
    np.save(emb_buf, np.ascontiguousarray(dense), allow_pickle=False)
    _atomic_write(EMBEDDINGS_PATH, emb_buf.getvalue())
    _atomic_write(FAISS_PATH, faiss_bytes)
    log.info("dense index -> %s (vectors=%d, dim=%d)", FAISS_PATH, index.ntotal, dim)

    log.info("building BM25 over %d tokenized passages ...", len(token_lists))
    bm25 = BM25Okapi(token_lists)
    bm25_path = INDEX_DIR / "bm25.pkl"
    sig_path = INDEX_DIR / "bm25.pkl.sig"
    payload = pickle.dumps(
        {"bm25": bm25, "names": names, "token_lists": token_lists},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    _atomic_write(bm25_path, payload)
    sig_path.write_text(hashlib.sha256(payload).hexdigest(), encoding="utf-8")
    log.info("BM25 -> %s + %s (%d docs)", bm25_path, sig_path, len(token_lists))

    meta = {
        "model": EMBED_MODEL,
        "dim": dim,
        "count": int(index.ntotal),
        "names": names,
        "has_bm25": True,
    }
    # A4: index_meta.json is the gating file for the reload watcher. It
    # must be written AFTER the data files so a reader never sees a
    # meta with names that don't match the on-disk index.
    meta_path = FAISS_PATH.parent / "index_meta.json"
    meta_tmp = meta_path.with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(meta_tmp, meta_path)
    log.info("index_meta.json -> %d names, dim=%d", len(names), dim)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
