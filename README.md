# Photo Index

Semantic and keyword search for your photo library. Upload a folder of images and find them by describing what is in them. Built with FAISS, BM25, and a multilingual text encoder. Everything runs locally in Docker.

## What it does

Drop in hundreds of images at once. The app extracts text from each one with OCR, encodes the text with a multilingual sentence transformer, and stores the vectors in a FAISS index. When you type a query, you get the most relevant images back, ranked by a hybrid of dense (semantic) and sparse (BM25 keyword) scores.

* Pure local. No external API calls. No cloud.
* Handles 300 images and 200 MB per upload.
* Top-k results are configurable from 1 to 25.
* Each result card opens a fullscreen viewer with keyboard navigation.
* Delete individual images from the search results.

## Quick start with Docker

Requirements: Docker and Docker Compose.

```bash
git clone <this-repo>
cd photo-index
cp .env.example .env
docker compose up -d --build
```

Open http://localhost:7860

The first run downloads the embedding model (about 2 GB) and the OCR language data. After that, restarts are fast.

A login screen appears on first open. Enter `admin` / `admin` to proceed. To change the password or disable auth, edit `.env` and restart (see [Authentication](#authentication) below).

To stop and clean up:

```bash
docker compose down
```

The `data/` directory on the host holds your images, OCR text, and the FAISS index. Back it up if you care about the contents.

## Running on a clean Ubuntu 24.04 server

This is the bare-metal setup. It works too if you do not want Docker.

### 1. System packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip \
  tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
  libgl1 libglib2.0-0 curl
```

### 2. Redis

```bash
sudo apt install -y redis-server
sudo systemctl enable --now redis-server
```

### 3. Application

```bash
git clone <this-repo>
cd photo-index
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Start the web and worker

Open two terminals.

Terminal 1, the web UI:

```bash
python web.py
```

Terminal 2, the OCR and indexing worker:

```bash
python worker.py
```

Both processes need `REDIS_URL=redis://127.0.0.1:6379/0` in the environment. They communicate through Redis. The web serves on port 7860 by default.

### 5. Open it

Visit http://your-server-ip:7860

## How it works

Web process (FastAPI):
* Serves the UI on port 7860.
* Accepts uploads at `POST /api/upload`.
* Exposes `GET /api/search?q=...&k=...` for the frontend.
* Exposes `DELETE /api/image/{name}` for removing images.
* Exposes `POST /api/login`, `POST /api/logout`, `GET /api/whoami` for auth.
* Polls a sentinel file every few seconds and reloads the in-memory index when the worker finishes a job.

Worker process (standalone):
* Pulls jobs from a Redis queue (`BLPOP`).
* Runs OCR with Tesseract on every new image.
* Encodes the OCR text with `BAAI/bge-m3`.
* Rebuilds FAISS and BM25 indexes from the union of old and new entries.
* Touches the sentinel file so the web process reloads.

Storage layout:

```
data/
  images/         # original uploads
  ocr/            # one .txt per image
  index/          # faiss.index, bm25.pkl, manifest.json, embeddings.npy
  staging/        # in-flight upload chunks
  .cache/         # Hugging Face model weights
  .session_key    # HMAC-SHA256 signing key for auth cookies (mode 600)
```

## Configuration

All settings are environment variables. Sensible defaults are baked in.

| Variable             | Default                | What it does                           |
|----------------------|------------------------|----------------------------------------|
| `WEB_PORT`           | `7860`                 | Port the web UI listens on             |
| `REDIS_URL`          | `redis://127.0.0.1:6379/0` | Redis connection string           |
| `PHOTOINDEX_ROOT`    | script directory       | Where the data directory's parent is   |
| `PHOTOINDEX_DATA`    | `$PHOTOINDEX_ROOT/data` | Override the data directory location  |
| `PHOTOINDEX_USER`    | `admin`                | Username for login (always `admin`)    |
| `PHOTOINDEX_PASS`    | `admin`                | Password for login (set both to empty to disable auth) |
| `PHOTOINDEX_SESSION_TTL_DAYS` | `7`           | How long the session cookie lasts      |
| `PHOTOINDEX_SESSION_SECURE`  | `0`           | Set to `1` to mark the cookie Secure (HTTPS) |
| `MAX_UPLOAD_FILES`   | `300`                  | Max images per upload                  |
| `MAX_UPLOAD_BYTES`   | `209715200`            | Max bytes per upload (200 MB)          |
| `MAX_TOP_K`          | `25`                   | Max results per search                 |
| `DEFAULT_TOP_K`      | `5`                    | Default results per search             |
| `EMBED_MODEL`        | `BAAI/bge-m3`          | Sentence transformer model name        |
| `OCR_LANGS`          | `rus+eng`              | Tesseract languages, plus-separated    |
| `OMP_NUM_THREADS`    | `4`                    | CPU threads for torch                  |
| `RELOAD_POLL_S`      | `5`                    | Seconds between index reload checks    |

## Search quality

The hybrid score combines a dense vector similarity (cosine) and a sparse BM25 score. The combination is done with Reciprocal Rank Fusion. In practice this means exact words in your query still help, but you do not need to use the exact words that appear in the image. Russian and English are both supported by the default model.

## Authentication

Photo Index has a single-user auth system. There is exactly one account (`admin`), no registration, and no way to add more users. By default, auth is **enabled** with credentials `admin` / `admin`.

### How it works

On first visit, a login modal appears over the dimmed search interface. The modal blocks all interaction (search, upload, settings) until valid credentials are entered. After login the modal fades out and the app becomes interactive. The session is stored in an HttpOnly cookie (`photoindex_session`) signed with HMAC-SHA256. The cookie expires after 7 days (configurable).

The signing key is a 32-byte random secret stored at `data/.session_key` (mode 600). It is created automatically on first start. Deleting this file logs out all sessions. Back it up if you want sessions to survive container recreates.

### Changing the password

Edit `.env` and restart:

```bash
PHOTOINDEX_USER=admin
PHOTOINDEX_PASS=your-new-password
```

The username is always `admin`. Only the password can be changed.

### Disabling auth

Set both variables to empty strings in `.env` and restart:

```bash
PHOTOINDEX_USER=
PHOTOINDEX_PASS=
```

The login modal disappears and all endpoints become public. Use this for local-only setups or when you have your own reverse-proxy auth in front.

### API access (curl, scripts)

HTTP Basic auth works for all API endpoints. Example upload:

```bash
curl -u admin:admin -F "files=@photo.jpg" https://your-domain/api/upload
```

Example search:

```bash
curl -u admin:admin "https://your-domain/api/search?q=cat&k=5"
```

### HTTPS and secure cookies

Behind HTTPS (nginx, Caddy, etc.), set in `.env`:

```bash
PHOTOINDEX_SESSION_SECURE=1
```

This marks the cookie `Secure` so browsers only send it over HTTPS. Leave at `0` for local development over plain HTTP.

### Session lifetime

Default is 7 days. Override with:

```bash
PHOTOINDEX_SESSION_TTL_DAYS=14
```

After expiry the user must log in again.

### Important

Change the default `admin` / `admin` password before exposing the app to anyone else. There is no rate limit on the login endpoint, so a weak password on a public host is easily brute-forced.

## Limitations

* OCR quality depends on the image. Scans of typed text work well. Handwriting does not.
* The default model is `BAAI/bge-m3` (1024 dimensions, about 2 GB on disk). Cold start takes a few seconds.
* No video, audio, or PDF support yet. Just still images.
* No automatic re-OCR if you change the OCR languages. Delete the image and re-upload.

## License

MIT. Do whatever you want.
