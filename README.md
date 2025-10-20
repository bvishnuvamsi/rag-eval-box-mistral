# RAG-Eval-Box (Mistral + FAISS) — README

## What this repo shows:
- End-to-end RAG on Mistral (ingest → FAISS → retrieve → generate with strict citations → eval).
- Groundedness enforced at the sentence level; dev/test label sets; Recall/MRR, F1, semantic similarity reported.
- Ablation over k ∈ {3,5,8} and two Mistral chat models; optional BM25+dense hybrid.
- Default recommendation: k=3, cost-efficient mistral-medium-latest (similar F1 to large; often better grounding at higher k).
- (Optional table) Ops snapshot: p50/p95, tokens/sec, $/100Q (model@k=3).

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
> **Optional (for hybrid search)**  
If you plan to use `--bm25`, make sure the BM25 dependency is installed:
```sh
pip install rank-bm25==0.2.2
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

### Answer with citations (strict bracket tokens)

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

### 3b) Hybrid retrieval (BM25 + dense, optional)

You can enable hybrid retrieval (dense + BM25 with rank-fusion) by adding `--bm25` on eval commands.

Under the hood this uses RRF (reciprocal rank fusion) with defaults:
- `rrf_k=60`
- `dense_weight=0.5`
- `bm25_weight=0.5`

**Retrieval-only (hybrid on)**
```sh
python -m src.cli eval-retrieval \
  --embed-model mistral-embed-2312 \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv \
  --labelset src/evals/qa_labelset_eng.jsonl \
  --k 5 \
  --bm25
```

End-to-end (hybrid on)
```sh
python -m src.cli eval-end2end \
  --embed-model mistral-embed-2312 \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv \
  --labelset src/evals/qa_labelset_eng.jsonl \
  --chat-model mistral-medium-latest \
  --k 5 \
  --bm25
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

- Expected (dev, k=3, dense-only): Recall@k ≈ 0.88, MRR ≈ 0.79
- End-to-end (dev, k=3, mistral-medium-latest): F1 ≈ 0.19, SentG ≈ 0.74, Gnd ≈ 0.65
- Hybrid (BM25+dense) on this small corpus didn’t improve F1; keep dense-only by default.


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
# JSON

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

- Groundedness is computed at the **sentence level**. A sentence is grounded iff it contains ≥1 exact bracket token that appears in the retrieved context.  
- **AnswerGrounded = 1 only if all sentences are grounded.** We never append citations post-hoc; missing citations count against both sentence- and answer-level scores.


## 8) Troubleshooting

SQLite OperationalError: disk I/O error or no such column: text_hash
* Remove any old cache and use /tmp:
```sh
rm -f /tmp/emb_cache.sqlite data/emb_cache.sqlite
export EMB_CACHE_PATH=/tmp/emb_cache.sqlite
```

* 429 Too Many Requests
    * Slow down:
```sh
export MISTRAL_RATE_LIMIT_SECONDS=1
```

* zsh “command not found: #”
    * Don’t paste # lines into zsh directly; they’re comments. Remove them or put them on their own line.

* zsh “no matches found: /tmp/emb_cache.sqlite”*
    * Quote globs:
```sh
rm -f "/tmp/emb_cache.sqlite"*
```

* Stop a running command
```sh
Ctrl + C  # stop immediately.
Ctrl + Z  #suspend; then 
kill %1   #to terminate the suspended job
```

## 9) CLI Reference (implemented in src/cli.py)

```sh
ping                            # env + FAISS sanity
api-check                       # calls /v1/models; shows model IDs
list-embed-models               # filters visible models for "embed"
build-faiss                     # build FAISS from chunks CSV (uses cache)
query                           # top-k retrieval
answer                          # RAG answer w/ strict bracket citations
eval-retrieval                  # retrieval metrics only
eval-end2end                    # retrieval + EM + groundedness
make-seed                       # generate tiny PDFs for demo
ingest-pdf                      # parse PDFs -> docs.csv
chunk                           # split page text -> chunks.csv
fetch-web                       # download curated URLs
ingest-web                      # parse HTML -> docs_web.csv
```
Common Flags
* --embed-model mistral-embed-2312
* --chat-model mistral-medium-latest
* --index-path data/real/faiss_web.index
* --meta-csv data/real/chunk_meta_web.csv
* --k 5
* --bm25  # enable hybrid retrieval (BM25 + dense with RRF)

## 10) Why you’re seeing the cache logs

This project includes a tiny SQLite embedding cache so repeated queries are instant:

* On first run you’ll see misses=1, then hits=1 on the second run.
* File path is EMB_CACHE_PATH (default: data/emb_cache.sqlite, we recommend /tmp/emb_cache.sqlite).

## 11) Repo layout (key bits)
```sh
src/
  cli.py                     # all commands live here
  models/client_mistral.py   # minimal Mistral HTTP client
  index/
    build_faiss.py           # build_and_save()
    search_faiss.py          # FAISS search (+ cached embed calls)
    emb_cache.py             # sqlite cache helpers
  evals/
    run_eval.py              # retrieval + end-to-end eval helpers
    qa_labelset_eng.jsonl    # sample labelset
  ingest/
    fetch_web.py             # download URLs
    html_parse.py            # parse HTML to CSV
    make_seed_pdfs.py        # tiny demo PDFs
    pdf_parse.py             # parse PDFs to CSV
    chunkers.py              # fixed-size chunking
data/
  real/                      # prepared dataset & outputs
```

## 12) Example: one-liner quickstart (copy/paste)
```sh
# 1) Env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo 'MISTRAL_API_KEY=YOUR_KEY' > .env
export EMB_CACHE_PATH=/tmp/emb_cache.sqlite
export MISTRAL_RATE_LIMIT_SECONDS=1

# 2) Build from provided web chunks
python -m src.cli build-faiss \
  --embed-model mistral-embed-2312 \
  --chunks-csv data/real/chunks_web.csv \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv

# 3) Query & Answer
python -m src.cli query "What is a PaymentIntent used for?" \
  --embed-model mistral-embed-2312 \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv \
  --k 5

python -m src.cli answer "What does GET /v1/customers return?" \
  --embed-model mistral-embed-2312 \
  --chat-model mistral-medium-latest \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv \
  --k 5

# 4) End-to-end eval
python -m src.cli eval-end2end \
  --embed-model mistral-embed-2312 \
  --index-path data/real/faiss_web.index \
  --meta-csv data/real/chunk_meta_web.csv \
  --labelset src/evals/qa_labelset_eng.jsonl \
  --chat-model mistral-medium-latest \
  --k 5 \
  --bm25
```

## 13) Dataset & Source References

This project uses public web documentation for small-scale RAG evaluation. All content is © the respective owners and used here for educational/testing purposes only. See each site’s terms for reuse policies.

Stripe API documentation
* Stripe API Reference (root) — https://docs.stripe.com/api
    * Payment Intents API — https://docs.stripe.com/api/payment_intents
    * Customers API — https://docs.stripe.com/api/customers
    * Metadata — https://docs.stripe.com/api/metadata

Google Engineering Practices
* Overview (Code Review: Introduction & Index) — https://google.github.io/eng-practices/review/
* How To Do A Code Review (for reviewers) — https://google.github.io/eng-practices/review/reviewer/
* The CL Author’s Guide (for authors) — https://google.github.io/eng-practices/review/developer/

Provenance: Original URLs listed in data/real/sources_e.txt; downloaded HTML stored in data/real/raw/; parsed text to data/real/docs_web.csv; chunked to data/real/chunks_web.csv.

Usage notes
* We store only small text chunks for retrieval evaluation (see data/real/chunk_meta_web.csv).

> If you are the content owner and want a URL removed from the example set, open an issue or PR and I’ll remove it immediately.

## 14) Ablation: `k` × model on dev/test

**Setup.** 
All runs use:
- Embed model: `mistral-embed-2312`
- Chat models: `mistral-large-latest`, `mistral-medium-latest`
- Index: `data/real/faiss_web.index`, metadata: `data/real/chunk_meta_web.csv`
- Temp: 0.0, strict citations (no post-hoc appends)
- Splits: `dev` (N=17), `test` (N=8)

#### dev (N=17)
| k | Model                 | F1   | Sem  | SentG | Gnd  |
|---|-----------------------|------|------|-------|------|
| 3 | mistral-large-latest  | 0.24 | 0.80 | 0.85  | 0.71 |
| 3 | mistral-medium-latest | 0.24 | 0.80 | 0.85  | 0.71 |
| 5 | mistral-large-latest  | 0.21 | 0.79 | 0.83  | 0.71 |
| 5 | mistral-medium-latest | 0.25 | 0.80 | 0.88  | 0.76 |
| 8 | mistral-large-latest  | 0.25 | 0.80 | 0.77  | 0.59 |
| 8 | mistral-medium-latest | 0.23 | 0.80 | 0.86  | 0.71 |

#### test (N=8)
| k | Model                 | F1   | Sem  | SentG | Gnd  |
|---|-----------------------|------|------|-------|------|
| 3 | mistral-large-latest  | 0.21 | 0.79 | 1.00  | 1.00 |
| 3 | mistral-medium-latest | 0.20 | 0.79 | 0.94  | 0.88 |
| 5 | mistral-large-latest  | 0.21 | 0.79 | 0.94  | 0.88 |
| 5 | mistral-medium-latest | 0.23 | 0.80 | 0.88  | 0.75 |
| 8 | mistral-large-latest  | 0.22 | 0.80 | 0.94  | 0.88 |
| 8 | mistral-medium-latest | 0.22 | 0.80 | 0.94  | 0.88 |

> **Metric notes.** *Sem* = semantic similarity; *SentG* = fraction of answer sentences with grounded citations; *Gnd* = answer-level groundedness.

*Sem* = cosine similarity between sentence embeddings of the prediction and the closest gold answer, averaged over answers.
- Encoder: `mistral-embed-2312` (mean-pooled)
- Preprocessing: lowercase, strip punctuation, collapse whitespace
- Score: `Sem = mean_i max_j cosine(emb(pred_i), emb(gold_j))`


Outputs are written to `evals/out/*.txt`.

### Retrieval-only

#### dev (N=17)
| k | Recall@k | MRR |
|---|----------|-----|
| 3 |   0.94   | 0.94|
| 5 |   1.00   | 0.96|
| 8 |   1.00   | 0.96|

#### test (N=8)
| k | Recall@k | MRR |
|---|----------|---- |
| 3 |   1.00   | 0.94|
| 5 |   1.00   | 0.94|
| 8 |   1.00   | 0.94|

---

### Reproduce

```bash
# End-to-end grid (k ∈ {3,5,8} × {dev,test} × {large,medium})
for SPLIT in dev test; do
  for K in 3 5 8; do
    for MODEL in mistral-large-latest mistral-medium-latest; do
      python -m src.cli eval-end2end \
        --embed-model mistral-embed-2312 \
        --chat-model $MODEL \
        --index-path data/real/faiss_web.index \
        --meta-csv data/real/chunk_meta_web.csv \
        --labelset src/evals/qa_labelset_${SPLIT}.jsonl \
        --k $K | tee evals/out/${SPLIT}_${MODEL}_k${K}.txt
    done
  done
done

# Retrieval-only (same grid)
for SPLIT in dev test; do
  for K in 3 5 8; do
    python -m src.cli eval-retrieval \
      --embed-model mistral-embed-2312 \
      --index-path data/real/faiss_web.index \
      --meta-csv data/real/chunk_meta_web.csv \
      --labelset src/evals/qa_labelset_${SPLIT}.jsonl \
      --k $K | tee evals/out/${SPLIT}_retrieval_k${K}.txt
  done
done
```

## Summary (dev N=17, test N=8)

- Retrieval is strong:
  - Dev: Recall@k hits 0.94 at k=3 and 1.00 by k≥5.
  - Test: Recall@k is 1.00 already at k=3.
    - Increasing k past 5 doesn’t buy coverage; it can even hurt downstream grounding.

- Answer quality is similar across models:
  - F1 clusters in the 0.20–0.25 range for both mistral-large-latest and mistral-medium-latest.
  - SemScore is stable around 0.79–0.80 for all settings.

- Grounding & citation behavior:
  - On dev, mistral-medium-latest tends to keep SentGrounded/Grounded higher as k grows (e.g., k=5 Medium: SentG 0.88 / Gnd 0.76), while Large degrades at k=8 (SentG 0.77, Gnd 0.59), suggesting context overload.
  - On test, Large @ k=3 shows perfect grounding (SentG 1.00 / Gnd 1.00), but both models converge to 0.94/0.88 by k=5–8.

- Best single points (by F1):
  - Dev: Large @ k=8 0.25, Medium @ k=5 0.25 (tie; Medium keeps better grounding).
  - Test: Medium @ k=5 0.23 (highest), with others close (0.21–0.22).

## Conclusion / Recommendation

- Default config: use k=3. It delivers near-max retrieval with less context noise and equal or better grounding than larger k.
- Model choice: mistral-medium-latest is as accurate as large on this labelset and often better grounded at higher k, making it a solid cost-efficient default.
- If you must pick one “best dev” point: Medium @ k=5 (F1 0.25, Sem 0.80, SentG 0.88, Gnd 0.76) balances output quality and citations well.
- If you want strongest test grounding with minimal context: Large @ k=3 (SentG 1.00, Gnd 1.00), though F1 is similar to Medium.

## Notes / Next steps

- The gap between high retrieval and modest F1/EM suggests generation-side tweaks could help:
  - Tighten the system prompt to “quote or paraphrase only from cited spans; answer in one factual sentence when possible”.
  - Add a short answer template (e.g., “Answer: …\nSources: [doc p#]”).
  - Consider re-ranking retrieved chunks or lowering k to reduce dilution.
- The test set is small (N=8), so treat per-row differences as directional rather than definitive.