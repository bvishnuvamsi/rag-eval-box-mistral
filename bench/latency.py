# bench/latency.py
from __future__ import annotations

# --- make 'src' importable when running this file directly ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root (parent of 'bench')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------

import os, time, csv, math, json, argparse
from dotenv import load_dotenv

from src.models.client_mistral import MistralClient
from src.index.search_faiss import search as faiss_search


def est_tokens(s: str) -> int:
    return max(1, math.ceil(len((s or "").strip()) / 4))


def run_once(client: MistralClient, model: str, question: str, context: str,
             temperature: float = 0.0, max_tokens: int = 256):
    sys_msg = {
        "role": "system",
        "content": "Answer briefly using ONLY the context. If not present, reply exactly: I don't know."
    }
    usr = {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n"}
    t0 = time.perf_counter()
    text = client.chat(model=model, messages=[sys_msg, usr], temperature=temperature, max_tokens=max_tokens)
    t1 = time.perf_counter()
    dur = t1 - t0
    toks = est_tokens(text)
    return {"latency_s": dur, "tokens": toks, "tps": toks / dur if dur > 0 else float("inf"), "answer": text}


def main():
    ap = argparse.ArgumentParser()
    # Use safe defaults; can be overridden by --models or env RAG_BENCH_MODELS
    default_models = os.getenv("RAG_BENCH_MODELS", "mistral-medium-latest,mistral-large-latest,ministral-8b-latest")
    ap.add_argument("--models", default=default_models)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--index-path", default="data/real/faiss_web.index")
    ap.add_argument("--meta-csv", default="data/real/chunk_meta_web.csv")
    ap.add_argument("--embed-model", default="mistral-embed-2312")
    ap.add_argument("--labelset", default="src/evals/qa_labelset_dev.jsonl")
    ap.add_argument("--out-csv", default="bench/out/latency.csv")
    args = ap.parse_args()

    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise SystemExit("Missing MISTRAL_API_KEY")
    client = MistralClient(api_key=key)

    # Filter requested models by what your key can see
    available = set(client.list_models())
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    chosen = [m for m in requested if m in available]
    missing = [m for m in requested if m not in available]
    if missing:
        print(f"[bench] Skipping unavailable models: {', '.join(missing)}")
    if not chosen:
        sample = ", ".join(sorted(list(available))[:10])
        raise SystemExit(f"[bench] None of the requested models are available. Visible models include: {sample} ...")

    # tiny subset of questions (first 5) for quick bench
    qs = []
    with open(args.labelset, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5: break
            obj = json.loads(line)
            q = obj.get("question") or obj.get("q") or obj.get("prompt")
            if q: qs.append(q)

    # Build context via FAISS
    contexts = []
    for q in qs:
        ctx = faiss_search(Path(args.index_path), Path(args.meta_csv), client, args.embed_model, q, k=args.k)
        blocks, seen = [], set()
        for r in ctx:
            tok = f"[{r['doc_id']} p{r['page_num']}]"
            if tok not in seen:
                seen.add(tok)
                blocks.append(f"{tok} {r['text']}")
        contexts.append("\n\n".join(blocks))

    Path("bench/out").mkdir(parents=True, exist_ok=True)
    rows = []
    for model in chosen:
        for q, ctx in zip(qs, contexts):
            m = run_once(client, model, q, ctx, temperature=0.0, max_tokens=256)
            rows.append({
                "model": model, "k": args.k, "question": q,
                "latency_s": round(m["latency_s"], 3),
                "tokens": m["tokens"],
                "tokens_per_sec": round(m["tps"], 1)
            })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","k","question","latency_s","tokens","tokens_per_sec"])
        w.writeheader(); w.writerows(rows)

    print(f"Wrote {args.out_csv}")
    # Summary
    import statistics as st
    by_model = {}
    for r in rows:
        by_model.setdefault((r["model"], r["k"]), []).append(r["latency_s"])
    print("\nSummary (E2E latency):")
    for (m,k), arr in by_model.items():
        p50 = st.median(arr)
        p95 = sorted(arr)[max(0, int(0.95*(len(arr)-1)))]
        avg_tps = st.mean([x["tokens_per_sec"] for x in rows if x["model"]==m and x["k"]==k])
        print(f"- {m} k={k}: p50={p50:.2f}s p95={p95:.2f}s, avg tps={avg_tps:.1f}")


if __name__ == "__main__":
    main()
