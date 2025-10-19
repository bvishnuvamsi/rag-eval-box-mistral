# RAG-Eval-Box (Mistral + FAISS) — README

> **Goal**: a tiny, reproducible RAG pipeline to ingest docs → build a FAISS index → retrieve → answer with strict bracket citations → run end-to-end eval (Recall/MRR + EM + Groundedness).

This repo already works end-to-end; follow the copy-and-paste commands below.

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

## 1) Sanity Checks

```sh
# Basic environment check
python -m src.cli ping

# Verify your API key + list models
python -m src.cli api-check

# Show embedding-capable models
python -m src.cli list-embed-models
```
## 2) Use the prepared web dataset (fast path)
This repo ships with parsed Stripe + Google Eng Practices data under `data/real/`:

* `data/real/chunks_web.csv` – chunk rows (chunk_id, doc_id, page_num, text)
* `data/real/chunk_meta_web.csv` – same columns, often used as both chunks+meta
* `data/real/faiss_web.index` – you can build this yourself (see below)

### Build (or rebuild) the FAISS index
Uses an on-disk embedding cache (`EMB_CACHE_PATH`) and request pacing (`MISTRAL_RATE_LIMIT_SECONDS`) to avoid rate limits.

```sh
export EMB_CACHE_PATH=/tmp/emb_cache.sqlite
export MISTRAL_RATE_LIMIT_SECONDS=1

# Option A: build from chunks_web.csv
python -m src.cli build-faiss \
  --embed-model mistral-embed-2312 \
  --chunks-csv data/real/chunks_web.csv \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv

# Option B: if chunks_web.csv is missing, reuse the meta CSV as input
python -m src.cli build-faiss \
  --embed-model mistral-embed-2312 \
  --chunks-csv data/real/chunk_meta_web.csv \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv
  ```

You should see cache logs like:

```sh
[cache] build hits=199 misses=0  (/tmp/emb_cache.sqlite)
Built index with 199 chunks -> data/real/faiss_web.index
```
## 3) Search & Answer (strict bracket citations)

### Search (top-k)
```sh
python -m src.cli query "What is a PaymentIntent used for?" \
  --embed-model mistral-embed-2312 \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv \
  --k 5
  ```
You’ll see top-k snippets and L2 scores.

Answer with citations
Citation rule: the model must copy exact tokens like [doc_id pX] that we provide.

We print the acceptable tokens before the answer to force correct citations.

```sh
python -m src.cli answer "What does GET /v1/customers return?" \
  --embed-model mistral-embed-2312 \
  --chat-model mistral-medium-latest \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv \
  --k 5
  ```

Expected pattern:
```sh
Acceptable tokens: [docs.stripe.com__api__customers p1] [docs.stripe.com__api__metadata p1]

=== ANSWER ===
GET /v1/customers returns a list of Customer objects … [docs.stripe.com__api__customers p1]
```
