# src/evals/metrics.py
# Purpose: small, testable metrics for retrieval and QA quality.

from __future__ import annotations
from typing import List, Sequence, Tuple, Set
import re


def recall_at_k(gold_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    """
    Binary Recall@k for 'at least one relevant item found in top-k'.
    Returns 1.0 if ANY gold id is in retrieved_ids, else 0.0.
    """
    gold = set(gold_ids)
    return 1.0 if any(r in gold for r in retrieved_ids) else 0.0


def reciprocal_rank(gold_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    """
    Mean Reciprocal Rank component for a single query.
    If the first relevant appears at 1-based rank R, return 1/R, else 0.0.
    """
    gold = set(gold_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def exact_substring_match(answer: str, expected_substrings: Sequence[str]) -> float:
    """
    Simple EM proxy: all expected substrings appear in the answer (case-insensitive).
    Returns 1.0 if all appear, else 0.0.
    """
    a = answer.lower()
    return 1.0 if all(s.lower() in a for s in expected_substrings) else 0.0


_CITATION_RE = re.compile(r"\[([^\]]+?)\s+p(\d+)\]")  # matches [doc_id p12]


def parse_citations(answer: str) -> Set[Tuple[str, int]]:
    """
    Extract citations of the form [doc_id pX] from the model answer.
    Returns a set of (doc_id, page_num).
    """
    out = set()
    for m in _CITATION_RE.finditer(answer):
        doc_id = m.group(1).strip()
        page = int(m.group(2))
        out.add((doc_id, page))
    return out


def groundedness_proxy(answer: str, retrieved_ctx_rows: Sequence[dict], require_citation: bool = True) -> float:
    """
    Cheap groundedness check:
    - If require_citation=True, fail (0.0) when there are NO [doc pX] citations.
    - Pass (1.0) if ANY cited (doc,page) appears among the retrieved context rows.
    - Else 0.0.
    """
    cites = parse_citations(answer)
    if require_citation and not cites:
        return 0.0

    ctx_pairs = {(r["doc_id"], int(r["page_num"])) for r in retrieved_ctx_rows}
    return 1.0 if any((d, p) in ctx_pairs for (d, p) in cites) else 0.0
