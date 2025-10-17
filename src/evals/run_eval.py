# src/evals/run_eval.py
# Purpose: run retrieval and end-to-end QA evals against a JSONL labelset.

from __future__ import annotations
from pathlib import Path
import json
from typing import List, Dict, Any

from .metrics import recall_at_k, reciprocal_rank, exact_substring_match, groundedness_proxy

# We reuse your existing search + client
from ..index.search_faiss import search as faiss_search
from ..models.client_mistral import MistralClient


def load_labelset(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out

def run_retrieval_eval(labelset_path: Path, index_path: Path, meta_csv: Path,
                       client: MistralClient, embed_model: str, k: int = 5) -> Dict[str, Any]:
    labels = load_labelset(labelset_path)
    rows = []
    r_at_k = []
    mrr = []

    for item in labels:
        q = item["question"]
        gold = item["gold_chunk_ids"]
        # retrieve
        hits = faiss_search(index_path, meta_csv, client, embed_model, q, k=k)
        retrieved_ids = [h["chunk_id"] for h in hits]

        r = recall_at_k(gold, retrieved_ids)
        rr = reciprocal_rank(gold, retrieved_ids)
        r_at_k.append(r)
        mrr.append(rr)

        rows.append({
            "question": q,
            "gold": gold,
            "retrieved": retrieved_ids,
            "Recall@k": r,
            "MRR": rr,
        })

    summary = {
        "avg_Recall@k": sum(r_at_k) / len(r_at_k) if r_at_k else 0.0,
        "avg_MRR": sum(mrr) / len(mrr) if mrr else 0.0,
        "n": len(labels),
    }
    return {"rows": rows, "summary": summary}


def build_context_blocks(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for r in hits:
        blocks.append(f"[{r['doc_id']} p{r['page_num']}] {r['text']}")
    return "\n\n".join(blocks)


def run_end2end_eval(labelset_path: Path, index_path: Path, meta_csv: Path,
                     client: MistralClient, embed_model: str, chat_model: str, k: int = 5) -> Dict[str, Any]:
    labels = load_labelset(labelset_path)
    rows = []

    ems = []
    grounds = []

    for item in labels:
        q = item["question"]
        gold = item["gold_chunk_ids"]
        expect = item.get("expected_substrings", [])

        # Retrieve
        hits = faiss_search(index_path, meta_csv, client, embed_model, q, k=k)

        # Build prompt
        system = {
            "role": "system",
            "content": (
                "You are a precise assistant. Answer using ONLY the CONTEXT. "
                "Cite sources in square brackets like [doc_id pX]. If the answer is not in context, say you don't know."
            )
        }
        user = {
            "role": "user",
            "content": f"QUESTION:\n{q}\n\nCONTEXT:\n{build_context_blocks(hits)}"
        }

        # Chat
        answer = client.chat(model=chat_model, messages=[system, user], temperature=0.0)

        # Metrics
        # Retrieval metrics for reference
        retrieved_ids = [h["chunk_id"] for h in hits]
        r = recall_at_k(gold, retrieved_ids)
        rr = reciprocal_rank(gold, retrieved_ids)

        # QA metrics
        em = exact_substring_match(answer, expect)
        gr = groundedness_proxy(answer, hits, require_citation=True)

        ems.append(em)
        grounds.append(gr)

        rows.append({
            "question": q,
            "gold": gold,
            "retrieved": retrieved_ids,
            "Recall@k": r,
            "MRR": rr,
            "answer": answer.strip(),
            "EM": em,
            "Grounded": gr,
        })

    summary = {
        "avg_EM": sum(ems) / len(ems) if ems else 0.0,
        "avg_Grounded": sum(grounds) / len(grounds) if grounds else 0.0,
        "n": len(labels),
    }
    return {"rows": rows, "summary": summary}




