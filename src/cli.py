# src/cli.py
# Purpose: your CLI front door. We start with two commands:
#  - ping: verify your environment and FAISS work
#  - api-check: verify your Mistral API key and list available models

from __future__ import annotations
#from models.client_mistral import MistralClient
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


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
def cmd_answer(question: str = typer.Argument(...),
               index_path: str = typer.Option("data/faiss.index"),
               meta_csv: str = typer.Option("data/chunk_meta.csv"),
               embed_model: str = typer.Option(..., prompt=True),
               chat_model: str = typer.Option("mistral-medium-latest", help="Chat model to use"),
               k: int = typer.Option(5)):
    """
    Retrieve top-k chunks, then call Mistral chat to generate an answer that cites [doc_id pX].
    """
    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        typer.secho("Missing MISTRAL_API_KEY", fg=typer.colors.RED)
        raise typer.Exit(1)
    client = MistralClient(api_key=key)
    ctx = faiss_search(Path(index_path), Path(meta_csv), client, embed_model, question, k=k)
    if not ctx:
        typer.secho("No retrieval results; cannot answer.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    # Build a minimal, explicit prompt with citations.
    context_blocks = []
    for r in ctx:
        context_blocks.append(f"[{r['doc_id']} p{r['page_num']}] {r['text']}")
    context_str = "\n\n".join(context_blocks)

    system_msg = {
        "role": "system",
        "content": (
            "You are a precise assistant. Answer using ONLY the CONTEXT. "
            "Cite sources in square brackets like [doc_id pX]. If the answer is not in context, say you don't know."
        )
    }
    user_msg = {
        "role": "user",
        "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context_str}"
    }

    answer = client.chat(model=chat_model, messages=[system_msg, user_msg], temperature=0.0)
    typer.secho("\n=== ANSWER ===", fg=typer.colors.GREEN)
    typer.echo(answer.strip())

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
    End-to-end eval: retrieval + chat answer quality (EM, groundedness).
    """
    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        typer.secho("Missing MISTRAL_API_KEY", fg=typer.colors.RED)
        raise typer.Exit(1)
    client = MistralClient(api_key=key)

    out = run_end2end_eval(Path(labelset), Path(index_path), Path(meta_csv), client, embed_model, chat_model, k=k)

    typer.secho("Per-question results:", fg=typer.colors.GREEN)
    for r in out["rows"]:
        typer.echo(f"- Q: {r['question']}")
        typer.echo(f"  retrieved: {r['retrieved']}")
        typer.echo(f"  Recall@k={r['Recall@k']:.2f}  MRR={r['MRR']:.2f}")
        typer.echo(f"  EM={r['EM']:.2f}  Grounded={r['Grounded']:.2f}")
        typer.echo(f"  ANSWER: {r['answer']}\n")
    s = out["summary"]
    typer.secho(f"\nSummary: N={s['n']}  avg_EM={s['avg_EM']:.2f}  avg_Grounded={s['avg_Grounded']:.2f}", fg=typer.colors.CYAN)

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


if __name__ == "__main__":
    app()
