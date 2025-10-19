# RAG-Eval-Box (Mistral + FAISS) — README

> **Goal**: a tiny, reproducible RAG pipeline to ingest docs → build a FAISS index → retrieve → answer with strict bracket citations → run end-to-end eval (Recall/MRR + EM + Groundedness).

✅ This repo already works end-to-end; follow the copy-and-paste commands below.

## 0) Prereqs
* Python 3.10+
* macOS/Linux (Windows WSL is fine)
* A Mistral API key (set it in your shell or `.env`)

```sh
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Create a .env file at the repo root:

```sh
# .env
MISTRAL_API_KEY=YOUR_MISTRAL_KEY_HERE
# optional but recommended
EMB_CACHE_PATH=/tmp/emb_cache.sqlite
MISTRAL_RATE_LIMIT_SECONDS=1
```

Tip: if you ever change cache schema or see SQLite errors, simply delete the file in /tmp and re-run.

