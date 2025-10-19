# src/cli.py
# Purpose: your CLI front door. We start with two commands:
#  - ping: verify your environment and FAISS work
#  - api-check: verify your Mistral API key and list available models

from __future__ import annotations
#from models.client_mistral import MistralClient
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from src.index.build_faiss import build_and_save


import os
import sys
import typer  # zero-boilerplate CLI
from dotenv import load_dotenv  # load .env into env vars
import httpx  # only for friendly error printing in api-check

from typing import Optional

# import client class (we already fixed relative imports earlier)
try:
    from .models.client_mistral import MistralClient
    from .index.build_faiss import build_and_save as build_index_and_meta
    from .index.search_faiss import search as faiss_search
    from .evals.run_eval import run_retrieval_eval, run_end2end_eval
    from .ingest.fetch_web import fetch_all
    from .ingest.html_parse import parse_dir as parse_html_dir
except ImportError:
    from models.client_mistral import MistralClient
    from index.build_faiss import build_and_save as build_index_and_meta
    from index.search_faiss import search as faiss_search
    from evals.run_eval import run_retrieval_eval, run_end2end_eval
    from ingest.fetch_web import fetch_all
    from ingest.html_parse import parse_dir as parse_html_dir


app = typer.Typer(help="RAG Eval Box CLI")


@app.command()
def ping() -> None:
    """
    Sanity check: shows Python/FAISS versions and whether MISTRAL_API_KEY is set.
    Exits non-zero if FAISS is broken so you don't waste time later.
    """
    load_dotenv()  # read .env if present (safe no-op if missing)

    # Don't import FAISS at module import time; fail fast only when you run ping.
    try:
        import faiss  # noqa: F401
        faiss_version = getattr(faiss, "__version__", "unknown")
    except Exception as e:
        typer.secho(f"FAISS import failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    key_set = bool(os.getenv("MISTRAL_API_KEY"))
    typer.echo(f"Python: {sys.version.split()[0]}")
    typer.echo(f"FAISS:  {faiss_version}")
    typer.echo("MISTRAL_API_KEY set: " + ("yes" if key_set else "no"))


@app.command("api-check")
def api_check() -> None:
    """
    Calls Mistral /v1/models to verify your API key and connectivity.
    Prints up to 10 model IDs on success.
    """
    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        typer.secho("Missing MISTRAL_API_KEY in .env (or environment).", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Import here so unit tests can stub the client without importing the world.
    from .models.client_mistral import MistralClient

    client = MistralClient(api_key=key)

    try:
        models = client.list_models()
    except httpx.HTTPStatusError as e:
        # Show status code and a slice of server text so debugging is fast.
        status = e.response.status_code
        body = e.response.text[:300].replace("\n", " ")
        typer.secho(f"API error {status}: {body}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except httpx.RequestError as e:
        typer.secho(f"Network error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"OK. {len(models)} model(s) visible:", fg=typer.colors.GREEN)
    for m in models[:10]:
        typer.echo(f" - {m}")

# --- add near the top (after existing imports) ---
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Safe local imports with relative fallbacks (like we fixed before)
try:
    from .ingest.make_seed_pdfs import make_seed_pdfs
    from .ingest.pdf_parse import parse_dir
    from .ingest.chunkers import make_chunks
except ImportError:
    from ingest.make_seed_pdfs import make_seed_pdfs
    from ingest.pdf_parse import parse_dir
    from ingest.chunkers import make_chunks

# --- add these commands at the bottom, before the if __name__ == "__main__": block ---

@app.command("make-seed")
def cmd_make_seed(out_dir: Optional[str] = typer.Option("data/seed_docs", help="Where to write seed PDFs")):
    """
    Create two tiny PDFs (policy + API guide) for quick testing.
    """
    load_dotenv()
    path = Path(out_dir)
    make_seed_pdfs(path)
    typer.secho(f"Seed PDFs written to: {path.resolve()}", fg=typer.colors.GREEN)


@app.command("ingest-pdf")
def cmd_ingest_pdf(input_dir: Optional[str] = typer.Option("data/seed_docs", help="Folder with PDFs"),
                   out_csv: Optional[str] = typer.Option("data/docs.csv", help="Output CSV for page texts")):
    """
    Parse all PDFs in input_dir and write page-level text to out_csv.
    """
    load_dotenv()
    n = parse_dir(Path(input_dir), Path(out_csv))
    typer.secho(f"Wrote {n} page rows to {out_csv}", fg=typer.colors.GREEN)


@app.command("chunk")
def cmd_chunk(docs_csv: Optional[str] = typer.Option("data/docs.csv", help="Input CSV from ingest-pdf"),
              out_csv: Optional[str] = typer.Option("data/chunks.csv", help="Output CSV of chunks"),
              chunk_size: Optional[int] = typer.Option(800, help="Chunk size (chars), v1 no overlap")):
    """
    Split page text into fixed-size chunks; write to out_csv.
    """
    load_dotenv()
    n = make_chunks(Path(docs_csv), Path(out_csv), chunk_size=chunk_size)
    typer.secho(f"Wrote {n} chunks to {out_csv}", fg=typer.colors.GREEN)

@app.command("list-embed-models")
def cmd_list_embed_models():
    """
    Shows model IDs that likely serve embeddings (filter: 'embed').
    Use one of these for --embed-model in build-index/query.
    """
    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        typer.secho("Missing MISTRAL_API_KEY", fg=typer.colors.RED)
        raise typer.Exit(1)
    client = MistralClient(api_key=key)
    models = client.list_models()
    embeds = [m for m in models if "embed" in m.lower()]
    if not embeds:
        typer.secho("No embedding models visible to this key. You may need a different plan.", fg=typer.colors.YELLOW)
    else:
        typer.secho("Embedding-capable models:", fg=typer.colors.GREEN)
        for m in embeds:
            typer.echo(f" - {m}")


@app.command("build-index")
def cmd_build_index(chunks_csv: str = typer.Option("data/chunks.csv"),
                    index_path: str = typer.Option("data/faiss.index"),
                    meta_csv: str = typer.Option("data/chunk_meta.csv"),
                    embed_model: str = typer.Option(..., prompt=True, help="Embeddings model ID (see list-embed-models)"),
                    batch_size: int = typer.Option(64)):
    """
    Embed chunks and build a FAISS index + metadata CSV.
    """
    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        typer.secho("Missing MISTRAL_API_KEY", fg=typer.colors.RED)
        raise typer.Exit(1)
    client = MistralClient(api_key=key)
    n = build_index_and_meta(Path(chunks_csv), Path(index_path), Path(meta_csv), client, embed_model, batch_size=batch_size)
    typer.secho(f"Indexed {n} chunks -> {index_path} and {meta_csv}", fg=typer.colors.GREEN)


@app.command("query")
def cmd_query(question: str = typer.Argument(...),
              index_path: str = typer.Option("data/faiss.index"),
              meta_csv: str = typer.Option("data/chunk_meta.csv"),
              embed_model: str = typer.Option(..., prompt=True),
              k: int = typer.Option(5, help="Top-k chunks to return")):
    """
    Embed the question, search FAISS, and print top-k chunks with doc/page.
    """
    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        typer.secho("Missing MISTRAL_API_KEY", fg=typer.colors.RED)
        raise typer.Exit(1)
    client = MistralClient(api_key=key)
    results = faiss_search(Path(index_path), Path(meta_csv), client, embed_model, question, k=k)
    if not results:
        typer.secho("No results.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)
    for r in results:
        typer.secho(f"[{r['rank']}] {r['doc_id']} p{r['page_num']}  (L2={r['score_l2']:.4f})", fg=typer.colors.CYAN)
        typer.echo(r["text"].strip()[:300] + ("..." if len(r["text"]) > 300 else ""))


@app.command("answer")
def cmd_answer(
    question: str = typer.Argument(...),
    index_path: str = typer.Option("data/faiss.index"),
    meta_csv: str = typer.Option("data/chunk_meta.csv"),
    embed_model: str = typer.Option(..., prompt=True),
    chat_model: str = typer.Option("mistral-medium-latest", help="Chat model to use"),
    k: int = typer.Option(5),
):
    """
    Retrieve top-k chunks, then call Mistral chat to generate an answer that cites [doc_id pX].
    """
    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        typer.secho("Missing MISTRAL_API_KEY", fg=typer.colors.RED)
        raise typer.Exit(1)

    client = MistralClient(api_key=key)

    # 1) Retrieve
    ctx = faiss_search(Path(index_path), Path(meta_csv), client, embed_model, question, k=k)
    if not ctx:
        typer.secho("No retrieval results; cannot answer.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    # 2) Build context with EXACT citation tokens from meta (no normalization)
    context_blocks: list[str] = []
    cite_tokens: list[str] = []
    seen: set[str] = set()
    for r in ctx:
        did = r["doc_id"]                 # keep EXACT value from CSV
        token = f"[{did} p{r['page_num']}]"
        if token not in seen:
            cite_tokens.append(token)
            seen.add(token)
        context_blocks.append(f"{token} {r['text']}")
    context_str = "\n\n".join(context_blocks)
    acceptable = " ".join(cite_tokens) if cite_tokens else ""

    # (optional) quick debug to verify tokens
    typer.secho(f"Acceptable tokens: {acceptable}", fg=typer.colors.BLUE)

    # 3) Tight prompt: no URLs; must paste one of the tokens exactly; short and direct.
    system_msg = {
        "role": "system",
        "content": (
            "You are a precise RAG assistant. Use ONLY the CONTEXT.\n"
            "Cite sources by copying EXACTLY one or more of the bracket tokens present in CONTEXT "
            "in the form [doc_id pX]. Do NOT include any URLs. "
            "If the answer is not in CONTEXT, reply exactly: I don't know."
        ),
    }
    user_msg = {
    "role": "user",
    "content": (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"Acceptable citation tokens (copy EXACTLY): {acceptable}\n"
        "Write a short answer of 1–2 sentences. "
        "Append one bracket token to the END of **every sentence**. "
        "Do not invent tokens; only use tokens shown above."
        ),
    }


    raw = client.chat(model=chat_model, messages=[system_msg, user_msg], temperature=0.0)
    answer = raw.strip()

    # IMPORTANT: no post-append. If model forgot a token, we show it as-is.
    typer.secho("\n=== ANSWER ===", fg=typer.colors.GREEN)
    typer.echo(answer)

@app.command("eval-retrieval")
def cmd_eval_retrieval(
    labelset: str = typer.Option("src/evals/qa_labelset.jsonl", help="Path to JSONL labelset"),
    index_path: str = typer.Option("data/faiss.index"),
    meta_csv: str = typer.Option("data/chunk_meta.csv"),
    embed_model: str = typer.Option(..., prompt=True),
    k: int = typer.Option(5),
):
    """
    Evaluate retrieval quality (Recall@k, MRR) over the labelset.
    """
    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        typer.secho("Missing MISTRAL_API_KEY", fg=typer.colors.RED)
        raise typer.Exit(1)
    client = MistralClient(api_key=key)

    out = run_retrieval_eval(Path(labelset), Path(index_path), Path(meta_csv), client, embed_model, k=k)

    # Pretty-print
    typer.secho("Per-question results:", fg=typer.colors.GREEN)
    for r in out["rows"]:
        typer.echo(f"- Q: {r['question']}")
        typer.echo(f"  retrieved: {r['retrieved']}")
        typer.echo(f"  Recall@k={r['Recall@k']:.2f}  MRR={r['MRR']:.2f}")
    s = out["summary"]
    typer.secho(f"\nSummary: N={s['n']}  avg_Recall@k={s['avg_Recall@k']:.2f}  avg_MRR={s['avg_MRR']:.2f}", fg=typer.colors.CYAN)


@app.command("eval-end2end")
def cmd_eval_end2end(
    labelset: str = typer.Option("src/evals/qa_labelset.jsonl"),
    index_path: str = typer.Option("data/faiss.index"),
    meta_csv: str = typer.Option("data/chunk_meta.csv"),
    embed_model: str = typer.Option(..., prompt=True),
    chat_model: str = typer.Option("mistral-medium-latest"),
    k: int = typer.Option(5),
):
    """
    End-to-end eval: retrieval + answer quality with
    - EM (substring proxy)
    - F1 (token-level, SQuAD-style)
    - SemScore (embedding cosine vs gold)
    - Sentence-level groundedness (fraction of sentences with a valid token)
    - Answer-level groundedness (all sentences grounded)
    Uses the same strict token-citation prompt as `answer` (no post-append).
    """
    import os, json, re, math
    from pathlib import Path
    from dotenv import load_dotenv
    from src.models.client_mistral import MistralClient
    from src.index.search_faiss import search as faiss_search
    # optional cache for semantic scoring
    try:
        from src.index.emb_cache import EmbeddingCache, get_or_embed
    except Exception:
        EmbeddingCache = None
        get_or_embed = None

    # ---------------- helpers ----------------
    SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+|\n+")
    WORD_RE = re.compile(r"\w+", re.UNICODE)

    def split_sentences(text: str) -> list[str]:
        t = (text or "").strip()
        if not t:
            return []
        return [p.strip() for p in SENT_SPLIT_RE.split(t) if p.strip()]

    def sentence_groundedness(answer: str, valid_tokens: list[str]) -> tuple[float, int]:
        """
        Returns (sent_frac, all_grounded_flag).
        sent_frac = fraction of sentences containing >=1 valid token.
        all_grounded_flag = 1 only if every sentence is grounded (and answer not empty).
        """
        sents = split_sentences(answer)
        if not sents:
            return 0.0, 0
        grounded_count = sum(1 for s in sents if any(tok in s for tok in valid_tokens))
        frac = grounded_count / len(sents)
        all_flag = 1 if grounded_count == len(sents) else 0
        return frac, all_flag

    def _canon_text(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip().lower()

    def tokenize(s: str) -> list[str]:
        return [w for w in WORD_RE.findall((s or "").lower()) if w]

    def f1_max(pred: str, gold_list: list[str]) -> float:
        """
        Token-level F1 vs each gold; take the max.
        """
        ptoks = tokenize(pred)
        if not gold_list or not ptoks:
            return 0.0
        best = 0.0
        for g in gold_list:
            gtoks = tokenize(g)
            if not gtoks:
                continue
            # overlap
            common = {}
            for t in ptoks:
                common[t] = min(ptoks.count(t), gtoks.count(t))
            overlap = sum(common.values())
            if overlap == 0:
                continue
            precision = overlap / len(ptoks)
            recall = overlap / len(gtoks)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            if f1 > best:
                best = f1
        return best

    def cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def embed_texts(client: MistralClient, model: str, texts: list[str]) -> list[list[float]]:
        """
        Embeds texts using Mistral embeddings. Uses on-disk cache if available.
        """
        # Use the same cache path your pipeline already uses (env or default).
        cache_path = Path(os.getenv("EMB_CACHE_PATH", "data/emb_cache.sqlite"))
        if EmbeddingCache and get_or_embed:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with EmbeddingCache(cache_path) as cache:
                embs, _, _ = get_or_embed(
                    cache=cache,
                    model=model,
                    texts=texts,
                    embed_fn=lambda xs: client.embed(model=model, inputs=xs),
                )
            return embs
        # Fallback without cache
        return client.embed(model=model, inputs=texts)

    def semantic_score_max(client: MistralClient, model: str, pred: str, gold_list: list[str]) -> float:
        """
        Max cosine similarity of pred vs any gold answer.
        """
        gold_list = [g for g in gold_list if (g or "").strip()]
        if not gold_list or not (pred or "").strip():
            return 0.0
        # batch embed: pred + golds
        texts = [pred] + gold_list
        vecs = embed_texts(client, model, texts)
        pv = vecs[0]
        best = 0.0
        for gv in vecs[1:]:
            best = max(best, cosine(pv, gv))
        return best

    ANSWER_KEYS = [
        "answers","answer","expected","gold_answers","gold_answer",
        "targets","target","labels","label"
    ]
    GOLD_KEYS = [
        "gold_doc_ids","gold","docs","doc_ids","gold_docs",
        "gold_chunks","gold_chunk_ids","gold_chunk_prefixes","gold_ids",
        "retrieval_ids","gold_retrieval_ids","positive_ids"
    ]

    def _auto_find_gold(obj: dict) -> list[str]:
        """Fallback: scan all fields for values that look like our ids."""
        out = []
        pat = re.compile(r"docs\.[\w\.-]+__\w+(?:__\w+)*(?:__p\d+__c\d+)?", re.I)
        for v in obj.values():
            if isinstance(v, str):
                if pat.search(v): out.append(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and pat.search(x):
                        out.append(x)
        return out

    def _load_labelset(p: Path):
        items = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                ex = json.loads(line)

                q = ex.get("question") or ex.get("q") or ex.get("prompt")
                if not q: continue

                answers = []
                for k in ANSWER_KEYS:
                    if k in ex:
                        v = ex[k]
                        answers = v if isinstance(v, list) else [v]
                        break
                if not answers:
                    for k, v in ex.items():
                        if "answer" in k.lower() and isinstance(v, (str, list)):
                            answers = v if isinstance(v, list) else [v]
                            break

                gold = []
                for k in GOLD_KEYS:
                    if k in ex:
                        v = ex[k]
                        gold = v if isinstance(v, list) else [v]
                        break
                if not gold:
                    gold = _auto_find_gold(ex)

                items.append({"question": q, "answers": answers, "gold": gold})
        return items

    def _exact_match(pred: str, gold_answers: list[str]) -> float:
        if not gold_answers: return 0.0
        p = _canon_text(pred)
        for a in gold_answers:
            aa = _canon_text(a)
            if aa and aa in p:
                return 1.0
        return 0.0

    def _norm_id(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"_{3,}", "__", s)   # collapse ___ to __
        s = re.sub(r"_+$", "", s)       # drop trailing underscores
        return s

    def _chunk_root(chunk_id: str) -> str:
        return re.sub(r"__p\d+__c\d+$", "", _norm_id(chunk_id))

    def _recall_mrr(retrieved_doc_ids: list[str], retrieved_chunk_ids: list[str], gold_ids: list[str]) -> tuple[float, float]:
        if not gold_ids:
            return 0.0, 0.0
        docs = [_norm_id(x) for x in retrieved_doc_ids]
        roots = [_chunk_root(x) for x in retrieved_chunk_ids]
        gold = [_norm_id(x) for x in gold_ids]
        hit_rank = None
        for i in range(len(retrieved_doc_ids)):
            d = docs[i]; r = roots[i]
            for g in gold:
                if d == g or r == g or d.startswith(g) or g.startswith(d) or r.startswith(g) or g.startswith(r):
                    hit_rank = i + 1
                    break
            if hit_rank is not None:
                break
        if hit_rank is None:
            return 0.0, 0.0
        return 1.0, 1.0 / hit_rank

    # ---------------- auth / client ----------------
    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        typer.secho("Missing MISTRAL_API_KEY", fg=typer.colors.RED)
        raise typer.Exit(1)
    client = MistralClient(api_key=key)

    # ---------------- run ----------------
    label_items = _load_labelset(Path(labelset))
    rows = []
    sum_em = sum_f1 = sum_sem = sum_sent_grounded = sum_ans_grounded = sum_recall = sum_mrr = 0.0

    for ex in label_items:
        question = ex["question"]
        gold_ids = ex["gold"]
        if gold_ids:
            typer.secho(f"[gold] {question} -> {gold_ids}", fg=typer.colors.BLUE)
        else:
            typer.secho(f"[gold] {question} -> NONE FOUND", fg=typer.colors.YELLOW)

        # retrieval
        ctx = faiss_search(Path(index_path), Path(meta_csv), client, embed_model, question, k=k)
        retrieved_chunk_ids = [r["chunk_id"] for r in ctx]
        retrieved_doc_ids = [r["doc_id"] for r in ctx]

        # strict token-citation prompt (same as `answer`)
        tokens, blocks, seen = [], [], set()
        for r in ctx:
            tok = f"[{r['doc_id']} p{r['page_num']}]"
            if tok not in seen:
                tokens.append(tok); seen.add(tok)
            blocks.append(f"{tok} {r['text']}")
        context_str = "\n\n".join(blocks)
        acceptable = " ".join(tokens) if tokens else ""

        system_msg = {
            "role": "system",
            "content": (
                "You are a precise RAG assistant. Use ONLY the CONTEXT.\n"
                "Cite sources by copying EXACTLY one or more of the bracket tokens present in CONTEXT "
                "in the form [doc_id pX]. Do NOT include any URLs. "
                "If the answer is not in CONTEXT, reply exactly: I don't know."
            ),
        }
        user_msg = {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\n"
                f"CONTEXT:\n{context_str}\n\n"
                f"Acceptable citation tokens (copy EXACTLY): {acceptable}\n"
                "Write a short answer of 1–2 sentences. "
                "Append one bracket token to the END of **every sentence**. "
                "Do not invent tokens; only use tokens shown above."
            ),
        }


        pred = client.chat(model=chat_model, messages=[system_msg, user_msg], temperature=0.0).strip()

        # metrics
        rec, mrr = _recall_mrr(retrieved_doc_ids, retrieved_chunk_ids, gold_ids)
        em = _exact_match(pred, ex["answers"])
        f1 = f1_max(pred, ex["answers"])
        sem = semantic_score_max(client, embed_model, pred, ex["answers"])

        sent_frac, all_flag = sentence_groundedness(pred, tokens)

        sum_recall += rec; sum_mrr += mrr
        sum_em += em; sum_f1 += f1; sum_sem += sem
        sum_sent_grounded += sent_frac; sum_ans_grounded += float(all_flag)

        rows.append({
            "question": question,
            "retrieved": retrieved_chunk_ids,
            "Recall@k": rec,
            "MRR": mrr,
            "EM": em,
            "F1": f1,
            "SemScore": sem,
            "SentGrounded": sent_frac,
            "Grounded": float(all_flag),
            "answer": pred,
        })

    # ---------------- print ----------------
    typer.secho("Per-question results:", fg=typer.colors.GREEN)
    for r in rows:
        typer.echo(f"- Q: {r['question']}")
        typer.echo(f"  retrieved: {r['retrieved']}")
        typer.echo(
            f"  Recall@k={r['Recall@k']:.2f}  MRR={r['MRR']:.2f}  "
            f"EM={r['EM']:.2f}  F1={r['F1']:.2f}  SemScore={r['SemScore']:.2f}  "
            f"SentGrounded={r['SentGrounded']:.2f}  Grounded={r['Grounded']:.2f}"
        )
        typer.echo(f"  ANSWER: {r['answer']}\n")

    n = max(len(rows), 1)
    typer.secho(
        f"\nSummary: N={n}  "
        f"avg_Recall@k={sum_recall/n:.2f}  avg_MRR={sum_mrr/n:.2f}  "
        f"avg_EM={sum_em/n:.2f}  avg_F1={sum_f1/n:.2f}  avg_SemScore={sum_sem/n:.2f}  "
        f"avg_SentGrounded={sum_sent_grounded/n:.2f}  avg_Grounded={sum_ans_grounded/n:.2f}",
        fg=typer.colors.CYAN,
    )




@app.command("fetch-web")
def cmd_fetch_web(sources: str = typer.Option("data/real/sources_e.txt", help="One URL per line"),
                  out_dir: str = typer.Option("data/real/raw", help="Where to save HTML")):
    """
    Download curated URLs to local HTML files.
    """
    pairs = fetch_all(Path(sources), Path(out_dir))
    typer.secho(f"Fetched {len(pairs)} pages into {out_dir}", fg=typer.colors.GREEN)

@app.command("ingest-web")
def cmd_ingest_web(raw_dir: str = typer.Option("data/real/raw"),
                   out_csv: str = typer.Option("data/real/docs_web.csv")):
    """
    Parse local HTML pages into docs CSV (doc_id, page_num=1, text).
    """
    n = parse_html_dir(Path(raw_dir), Path(out_csv))
    typer.secho(f"Wrote {n} rows to {out_csv}", fg=typer.colors.GREEN)

@app.command("build-faiss")
def cmd_build_faiss(
    chunks_csv: str = typer.Option(..., "--chunks-csv", help="CSV with columns: chunk_id, doc_id, page_num, text"),
    index_path: str = typer.Option(..., "--index-path", help="Output FAISS index path"),
    meta_csv: str = typer.Option(..., "--meta-csv", help="Output metadata CSV path"),
    embed_model: str = typer.Option(..., "--embed-model", help="Embedding model id (e.g., mistral-embed-2312)"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Embedding batch size"),
):
    """Build a FAISS index from chunks CSV, writing index + meta. Uses the embedding cache."""
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        typer.secho("Missing MISTRAL_API_KEY", fg=typer.colors.RED)
        raise typer.Exit(1)
    client = MistralClient(api_key=key)

    n = build_and_save(
        Path(chunks_csv),
        Path(index_path),
        Path(meta_csv),
        client,
        embed_model,
        batch_size=batch_size,
    )
    typer.secho(f"Built index with {n} chunks -> {index_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
