"""OCR pipeline v4:
  - Russian Tesseract on multi-channel preprocessed images (text body)
  - English Tesseract on inverted variants, keep only URL/email-like tokens
  - Strip noise: single non-word symbols (°*|~© etc), keep multi-char patterns
  - Skip blank images (mean<8 or std<4)
  - Filter results by noise ratio (>45% single-char words)

Round 0 (settings+upload):
  - `--source DIR` to OCR a directory other than the main IMAGES_DIR
    (used by the worker to OCR the staging dir without polluting the
    global manifest until the files are moved into IMAGES_DIR).
  - idempotent: files that already have an OCR .txt in the chosen
    output dir are skipped (unless --force).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageOps

from config import (
    IMAGES_DIR,
    MANIFEST_PATH,
    OCR_DIR,
)

log = logging.getLogger("ocr")

# A16 fix: probe tesseract availability at module import.
try:
    _tess_version = pytesseract.get_tesseract_version()
except pytesseract.TesseractNotFoundError as e:
    raise SystemExit(
        f"tesseract binary not found on PATH: {e}. "
        "Install with: apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng"
    ) from e

# A15 fix: per-image Tesseract timeout (seconds).
TESSERACT_TIMEOUT_S = 60

OCR_LANGS_RUS = "rus"
OCR_LANGS_ENG = "eng"
TESS_CONFIG = "--psm 6"

BLANK_STD_THRESHOLD = 4.0
BLANK_MEAN_THRESHOLD = 8.0

URL_RE = re.compile(r"https?://[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
DOMAIN_RE = re.compile(r"\b[A-Za-z0-9\-]+\.(?:ru|com|org|net|io|co|ai|gov|edu)(?:/[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*)?\b", re.IGNORECASE)

WORD_RE = re.compile(r"[А-Яа-яёЁA-Za-z0-9][А-Яа-яёЁA-Za-z0-9\-_/.,:;()%]*", re.UNICODE)
TOKEN_RE = re.compile(r"[А-Яа-яёЁA-Za-z0-9][А-Яа-яёЁA-Za-z0-9\-_/.,:;()%]+", re.UNICODE)
NOISE_CHAR_RE = re.compile(r"(?:(?<=\s)|^)([^\sА-Яа-яёЁA-Za-z0-9]|[^\sА-Яа-яёЁA-Za-z0-9][^\sА-Яа-яёЁA-Za-z0-9]*)(?=\s|$)", re.UNICODE)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images() -> list[Path]:
    return sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def is_blank(img: Image.Image) -> bool:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    if arr.mean() < BLANK_MEAN_THRESHOLD:
        return True
    if arr.std() < BLANK_STD_THRESHOLD:
        return True
    return False


def variants_for_rus(img: Image.Image) -> dict[str, Image.Image]:
    rgb = img.convert("RGB")
    r, g, b = rgb.split()
    g_lum = ImageOps.grayscale(rgb)
    return {
        "gray": ImageOps.autocontrast(g_lum),
        "R": ImageOps.autocontrast(r),
        "G": ImageOps.autocontrast(g),
        "B": ImageOps.autocontrast(b),
        "inv": ImageOps.autocontrast(ImageOps.invert(g_lum)),
    }


def variants_for_eng(img: Image.Image) -> dict[str, Image.Image]:
    rgb = img.convert("RGB")
    g_lum = ImageOps.grayscale(rgb)
    return {
        "inv": ImageOps.autocontrast(ImageOps.invert(g_lum)),
        "gray": ImageOps.autocontrast(g_lum),
    }


def otsu_variant(img: Image.Image) -> Image.Image | None:
    arr = np.asarray(ImageOps.grayscale(img), dtype=np.uint8)
    if arr.size == 0:
        return None
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    if hist.sum() == 0:
        return None
    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))
    global_mean = cum_mean[-1] / total
    omega = cum_sum / total
    mu = cum_mean / total
    sigma_b2 = (global_mean * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    thr = int(np.argmax(sigma_b2))
    return Image.fromarray(((arr < thr) * 255).astype(np.uint8))


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def is_noisy(text: str) -> bool:
    ws = words(text)
    if len(ws) < 3:
        return True
    short = sum(1 for w in ws if len(w) <= 1)
    return short / len(ws) > 0.45


def clean_noise(text: str) -> str:
    return NOISE_CHAR_RE.sub(" ", text)


def extract_url_tokens(text: str) -> set[str]:
    found: set[str] = set()
    for m in URL_RE.findall(text):
        found.add(m.rstrip(".,;:)]}'\""))
    for m in EMAIL_RE.findall(text):
        found.add(m)
    for m in DOMAIN_RE.findall(text):
        m = m.rstrip(".,;:)]}'\"")
        if len(m) > 4:
            found.add(m)
    return found


def ocr_one(path: Path) -> tuple[Path, str, int, bool]:
    with Image.open(path) as raw:
        img = raw.convert("RGB")
        if is_blank(img):
            return path, "", 0, True

        vs = variants_for_rus(img)
        otsu = otsu_variant(img)
        if otsu is not None:
            vs["otsu"] = otsu

        rus_texts: list[str] = []
        for v in vs.values():
            try:
                t = pytesseract.image_to_string(v, lang=OCR_LANGS_RUS, config=TESS_CONFIG, timeout=TESSERACT_TIMEOUT_S)
            except Exception:
                continue
            t = t.strip()
            if not t or is_noisy(t):
                continue
            rus_texts.append(clean_noise(t))

        eng_tokens: set[str] = set()
        for v in variants_for_eng(img).values():
            try:
                t = pytesseract.image_to_string(v, lang=OCR_LANGS_ENG, config=TESS_CONFIG, timeout=TESSERACT_TIMEOUT_S)
            except Exception:
                continue
            t = t.strip()
            if t:
                eng_tokens.update(extract_url_tokens(t))

    merged_words: list[str] = []
    seen: set[str] = set()
    for t in rus_texts:
        for w in words(t):
            key = w.lower()
            if key in seen:
                continue
            seen.add(key)
            merged_words.append(w)

    for tok in eng_tokens:
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        merged_words.append(tok)

    merged = " ".join(merged_words)
    return path, merged, len(merged), False


def load_manifest() -> dict[str, dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(m: dict[str, dict]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(
    workers: int = 4,
    force: bool = False,
    source_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, dict]:
    """OCR every image in `source_dir`, write .txt to `out_dir`,
    merge results into the global manifest. Both dirs default to the
    main IMAGES_DIR / OCR_DIR.
    """
    source_dir = Path(source_dir) if source_dir else IMAGES_DIR
    out_dir = Path(out_dir) if out_dir else OCR_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    manifest = load_manifest()
    todo: list[Path] = []
    for p in images:
        out_file = out_dir / f"{p.stem}.txt"
        if force or not out_file.exists():
            todo.append(p)
    log.info(
        "source=%s out=%s total=%d to_process=%d cached=%d",
        source_dir, out_dir, len(images), len(todo), len(images) - len(todo),
    )

    if not todo:
        return manifest

    blanks = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(ocr_one, p): p for p in todo}
        done = 0
        for fut in as_completed(futures):
            try:
                path, text, n, blank = fut.result()
            except Exception as e:
                p = futures[fut]
                log.error("  ! %s: %s", p.name, e)
                continue
            if blank:
                blanks += 1
            # A28 fix: write to .tmp, then atomic-rename. A SIGKILL
            # mid-write leaves only a .tmp; the next run sees the
            # final name missing and re-OCRs.
            out_path = out_dir / f"{path.stem}.txt"
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_text(text, encoding="utf-8")
            os.replace(tmp_path, out_path)
            # If the source is a non-default dir, the file is named by stem
            # and may collide with an existing manifest entry. Only register
            # the global manifest when the source IS the main images dir.
            if source_dir.resolve() == IMAGES_DIR.resolve():
                manifest[path.name] = {
                    "path": str(IMAGES_DIR / path.name),
                    "ocr_path": str(OCR_DIR / f"{path.stem}.txt"),
                    "chars": n,
                    "blank": blank,
                }
            done += 1
            if done % 25 == 0 or done == len(todo):
                log.info("  ocr %d/%d (blanks=%d)", done, len(todo), blanks)

    if source_dir.resolve() == IMAGES_DIR.resolve():
        save_manifest(manifest)
        log.info(
            "manifest -> %s (entries=%d, blanks=%d)",
            MANIFEST_PATH, len(manifest), blanks,
        )
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Re-OCR files that already have a .txt output")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of Tesseract worker threads")
    p.add_argument("--source", type=Path, default=None,
                   help="Directory of images to OCR (default: data/images)")
    p.add_argument("--out", type=Path, default=None,
                   help="Where to write .txt outputs (default: data/ocr)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    run(
        workers=args.workers,
        force=args.force,
        source_dir=args.source,
        out_dir=args.out,
    )
