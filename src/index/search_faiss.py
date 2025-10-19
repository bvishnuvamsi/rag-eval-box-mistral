# src/index/search_faiss.py
# Purpose: load FAISS + meta, embed a query (with cache), return top-k matches.

from __future__ import annotations
from pathlib import Path
import os, csv
import faiss
import numpy as np
from src.index.emb_cache import EmbeddingCache, get_or_embed

def load_meta(meta_csv: Path) -> list[dict]:
    rows = []
    with meta_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["vec_id"] = int(r["vec_id"])
            r["page_num"] = int(r["page_num"])
            rows.append(r)
    return rows

def _embed_query(client, model: str, query: str) -> np.ndarray:
    cache_path = Path(os.getenv("EMB_CACHE_PATH") or "data/emb_cache.sqlite")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return client.embed(model=model, inputs=texts)

    with EmbeddingCache(cache_path) as cache:
        embs, hits, misses = get_or_embed(cache, model, [query], embed_fn)
        if hits or misses:
            print(f"[cache] query hits={hits} misses={misses}  ({cache_path})")
    return np.asarray(embs[0], dtype="float32")[None, :]

def search(index_path: Path, meta_csv: Path, client, embed_model: str, query: str, k: int = 5) -> list[dict]:
    index = faiss.read_index(str(index_path))
    meta = load_meta(meta_csv)

    qvec = _embed_query(client, embed_model, query)

    distances, ids = index.search(qvec, k)
    out = []
    for dist, vid in zip(distances[0].tolist(), ids[0].tolist()):
        if vid == -1:
            continue
        row = meta[vid]
        out.append({
            "rank": len(out) + 1,
            "score_l2": dist,
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "page_num": row["page_num"],
            "text": row["text"],
        })
    return out
