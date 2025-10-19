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

## 4) Evaluate
We include a small labelset `src/evals/qa_labelset_eng.jsonl`. Each line has:
* `question`
* one or more reference `answers`
* `gold_doc_ids` (or other gold fields) — used to compute retrieval hit

***

### End-to-end eval (retrieval + answer)
```sh
python -m src.cli eval-end2end \
  --embed-model mistral-embed-2312 \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv \
  --labelset src/evals/qa_labelset_eng.jsonl \
  --chat-model mistral-medium-latest \
  --k 5
  ```

You’ll see, per question:

* Recall@k / MRR: did retrieval hit a gold doc?
* EM (exact-match proxy): does the generated answer contain any reference answer substring?
* Grounded: did the model include at least one exact bracket token from context?

Example success:
```sh
Summary: N=4  avg_Recall@k=1.00  avg_MRR=1.00  avg_EM=1.00  avg_Grounded=1.00
```

Retrieval-only eval

```sh
python -m src.cli eval-retrieval \
  --embed-model mistral-embed-2312 \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv \
  --labelset src/evals/qa_labelset_eng.jsonl \
  --k 5
```

## 5) (Optional) Build your own dataset from PDFs or URLs

A) Tiny seed PDFs (quick demo)

```sh
# Generate two tiny PDFs
python -m src.cli make-seed --out-dir data/seed_docs

# Parse PDFs -> docs.csv (page-level text)
python -m src.cli ingest-pdf --input-dir data/seed_docs --out-csv data/docs.csv

# Chunk -> chunks.csv
python -m src.cli chunk --docs-csv data/docs.csv --out-csv data/chunks.csv --chunk-size 800

# Build index
python -m src.cli build-faiss \
  --embed-model mistral-embed-2312 \
  --chunks-csv data/chunks.csv \
  --index-path data/faiss.index \
  --meta-csv data/chunk_meta.csv
```

B) Fetch & parse web pages

Put URLs (one per line) in data/real/sources_e.txt, then:

```sh
# Download HTML locally
python -m src.cli fetch-web --sources data/real/sources_e.txt --out-dir data/real/raw

# Parse HTML -> docs CSV
python -m src.cli ingest-web --raw-dir data/real/raw --out-csv data/real/docs_web.csv

# Chunk and build index as above…
```

## 6) Labels: format (JSONL)

Example src/evals/qa_labelset_eng.jsonl:

```sh
JSON

{"question":"What does GET /v1/customers return?","answers":["a list of customer objects","a paginated list of customers"],"gold_doc_ids":["docs.stripe.com__api__customers","docs.stripe.com__api__metadata"]}
{"question":"What is a PaymentIntent used for?","answers":["to track the lifecycle of a payment, including authentication/confirmation, ensuring at most one successful charge"],"gold_doc_ids":["docs.stripe.com__api__payment_intents"]}
{"question":"What should a reviewer look for in a code review?","answers":["design, functionality, complexity, tests, naming, comments, style, documentation"],"gold_doc_ids":["google.github.io__eng-practices__review","google.github.io__eng-practices__review__reviewer__standard.html"]}
{"question":"Where are Google’s code review guidelines split?","answers":["overview, how to do a code review, the cl author's guide"],"gold_doc_ids":["google.github.io__eng-practices__review","google.github.io__eng-practices__review__reviewer__standard.html","google.github.io__eng-practices__review__developer"]}

```
Flexible loader: the eval code tries several common field names (answers, gold_doc_ids, etc.) and falls back to scanning for strings that look like our doc IDs.

## 7) How strict citations work (important)

* During answer and eval-end2end, we render the context like:

```sh
[docs.stripe.com__api__customers p1] <chunk text…>
[docs.stripe.com__api__metadata p1] <chunk text…>
```

* We also show “Acceptable citation tokens” (e.g., [docs.stripe.com__api__customers p1]).
* The system prompt forces the model to copy one of those tokens exactly.
* A post-check appends a token if the model forgot one.
* Groundedness metric = 1.0 iff at least one exact token is present.