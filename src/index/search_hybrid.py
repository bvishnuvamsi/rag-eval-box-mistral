from __future__ import annotations
from pathlib import Path
import csv, os, re, time
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from .search_faiss import search as dense_search
from src.models.client_mistral import MistralClient

WORD = re.compile(r"\w+", re.UNICODE)

# Cache BM25 index per meta file (path + mtime)
_BM25_CACHE: Dict[Tuple[str, float], Tuple[BM25Okapi, List[List[str]], List[Dict]]] = {}

def _read_rows(meta_csv: Path) -> List[Dict]:
    with meta_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)

def _tokenize_rows(rows: List[Dict]) -> List[List[str]]:
    return [WORD.findall((r.get("text") or "").lower()) for r in rows]

def _get_bm25(meta_csv: Path) -> Tuple[BM25Okapi, List[List[str]], List[Dict]]:
    mtime = os.path.getmtime(meta_csv)
    key = (str(meta_csv), mtime)
    hit = _BM25_CACHE.get(key)
    if hit:  # cache hit
        return hit
    rows = _read_rows(meta_csv)
    corpus = _tokenize_rows(rows)
    bm25 = BM25Okapi(corpus)
    _BM25_CACHE.clear()            # keep only latest version
    _BM25_CACHE[key] = (bm25, corpus, rows)
    return bm25, corpus, rows

def _bm25_topk(bm25: BM25Okapi, corpus: List[List[str]], query: str, rows: List[Dict], k: int) -> List[Dict]:
    q = WORD.findall((query or "").lower())
    scores = bm25.get_scores(q)
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    out = []
    for rank, i in enumerate(idxs, 1):
        r = rows[i]
        out.append({
            "rank": rank,
            "score_bm25": float(scores[i]),
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "page_num": int(r["page_num"]),
            "text": r["text"],
        })
    return out

def _dedupe_by_chunk_id(lst: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in lst:
        cid = r.get("chunk_id")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(r)
    return out

def _rrf_fuse(dense: List[Dict], lexical: List[Dict], k: int, rrf_k: int = 60,
              w_dense: float = 0.5, w_bm25: float = 0.5) -> List[Dict]:
    # reciprocal rank fusion with optional weights
    scores = {}
    def add(lst, w):
        for i, item in enumerate(lst):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + w * (1.0 / (rrf_k + i + 1))
    add(dense, w_dense); add(lexical, w_bm25)

    fused = []
    rank = 1
    for cid, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]:
        item = next((x for x in dense if x["chunk_id"] == cid), None) or next((x for x in lexical if x["chunk_id"] == cid), None)
        o = dict(item)
        o["score_rrf"] = sc
        o["rank"] = rank
        rank += 1
        fused.append(o)
    return fused

def search(index_path: Path,
           meta_csv: Path,
           client: MistralClient,
           embed_model: str,
           query: str,
           k: int = 5,
           rrf_k: int = 60,
           bm25_top: int = 50,
           dense_weight: float = 0.5,
           bm25_weight: float = 0.5) -> List[Dict]:
    # lexical
    bm25, corpus, rows = _get_bm25(meta_csv)
    bm25_hits = _bm25_topk(bm25, corpus, query, rows, k=bm25_top)
    bm25_hits = _dedupe_by_chunk_id(bm25_hits)

    # dense
    dense_hits = dense_search(index_path, meta_csv, client, embed_model, query, k=bm25_top)
    dense_hits = _dedupe_by_chunk_id(dense_hits)

    # fuse
    fused = _rrf_fuse(dense_hits, bm25_hits, k=k, rrf_k=rrf_k, w_dense=dense_weight, w_bm25=bm25_weight)
    return fused
