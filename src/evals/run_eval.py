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

import re

def _normalize_text(s: str) -> str:
    # lower-case, collapse punctuation/whitespace
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def em_from_candidates(pred: str, answers: list[str]) -> float:
    """Return 1.0 if ANY normalized gold answer is a substring of the normalized prediction."""
    p = _normalize_text(pred)
    for a in answers or []:
        if _normalize_text(a) in p:
            return 1.0
    return 0.0

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
    meta_rows = _load_meta_rows(meta_csv)
    rows = []
    r_at_k = []
    mrr = []

    for item in labels:
        q = item["question"]
        #gold = item["gold_chunk_ids"]
        gold = _resolve_gold_chunks(item, meta_rows)

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


def run_end2end_eval(
    labelset_path: Path,
    index_path: Path,
    meta_csv: Path,
    client: MistralClient,
    embed_model: str,
    chat_model: str,
    k: int = 5,
) -> Dict[str, Any]:
    import re

    def _normalize_text(s: str) -> str:
        # lowercase, remove non-alphanum to spaces, collapse spaces
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    def em_from_candidates(pred: str, answers: list[str]) -> float:
        """1.0 if ANY normalized gold answer is a substring of normalized prediction; else 0.0."""
        if not answers:
            return 0.0
        p = _normalize_text(pred or "")
        for a in answers:
            if _normalize_text(a) in p:
                return 1.0
        return 0.0

    labels = load_labelset(labelset_path)
    rows = []
    meta_rows = _load_meta_rows(meta_csv)
    ems: list[float] = []
    grounds: list[float] = []

    for item in labels:
        q = item["question"]

        # Resolve gold chunks from gold_doc_ids (or whatever your resolver supports)
        gold = _resolve_gold_chunks(item, meta_rows)
        # Debug: show what gold we resolved
        if item.get("gold_doc_ids"):
            print(f"[gold] {q} -> {item['gold_doc_ids']}")
        else:
            print(f"[gold] {q} -> NONE FOUND")

        # Acceptable answers (new field), fallback to older key if present
        expected_answers = item.get("answers") or item.get("expected_substrings") or []

        # Retrieve
        hits = faiss_search(index_path, meta_csv, client, embed_model, q, k=k)

        # Build prompt
        system = {
            "role": "system",
            "content": (
                "You are a precise assistant. Answer using ONLY the CONTEXT. "
                "Cite sources in square brackets like [doc_id pX]. If the answer is not in context, say you don't know."
            ),
        }
        user = {
            "role": "user",
            "content": f"QUESTION:\n{q}\n\nCONTEXT:\n{build_context_blocks(hits)}",
        }

        # Chat
        answer = client.chat(model=chat_model, messages=[system, user], temperature=0.0)
        answer_text = (answer or "").strip()

        # Retrieval metrics
        retrieved_ids = [h["chunk_id"] for h in hits]
        r = recall_at_k(gold, retrieved_ids)
        rr = reciprocal_rank(gold, retrieved_ids)

        # QA metrics
        em = em_from_candidates(answer_text, expected_answers)
        gr = groundedness_proxy(answer_text, hits, require_citation=True)

        ems.append(em)
        grounds.append(gr)

        rows.append({
            "question": q,
            "gold": gold,
            "retrieved": retrieved_ids,
            "Recall@k": r,
            "MRR": rr,
            "answer": answer_text,
            "EM": em,
            "Grounded": gr,
        })

    summary = {
        "avg_EM": sum(ems) / len(ems) if ems else 0.0,
        "avg_Grounded": sum(grounds) / len(grounds) if grounds else 0.0,
        "n": len(labels),
    }
    return {"rows": rows, "summary": summary}


def _load_meta_rows(meta_csv: Path) -> list[dict]:
    import csv
    rows = []
    with meta_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "page_num": int(row["page_num"]),
                "text": row["text"],
            })
    return rows

def _resolve_gold_chunks(item: dict, meta_rows: list[dict]) -> list[str]:
    # If explicit chunk ids exist, use them.
    if item.get("gold_chunk_ids"):
        return item["gold_chunk_ids"]

    # Expand doc-level ids if provided.
    out_ids = set()
    doc_patterns = [p.lower() for p in item.get("gold_doc_ids", []) + item.get("gold_doc_patterns", [])]
    text_snippets = [s.lower() for s in item.get("gold_text_snippets", [])]

    for r in meta_rows:
        ok = True
        if doc_patterns:
            ok = any(p in r["doc_id"].lower() for p in doc_patterns)
        if ok and text_snippets:
            ok = any(s in r["text"].lower() for s in text_snippets)
        if ok and (doc_patterns or text_snippets):
            out_ids.add(r["chunk_id"])

    return list(out_ids)



