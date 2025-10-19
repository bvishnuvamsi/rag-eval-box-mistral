# src/index/build_faiss.py
# Purpose: read chunk CSV -> embed text (with cache) -> build FAISS index -> save index + metadata CSV.

from __future__ import annotations

import csv
import os
from pathlib import Path

import faiss
import numpy as np

from src.index.emb_cache import EmbeddingCache, get_or_embed


def read_chunks(chunks_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with chunks_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "page_num": int(row["page_num"]),
                    "text": row["text"],
                }
            )
    return rows


def write_meta(meta_csv: Path, rows_with_ids: list[dict]) -> None:
    meta_csv.parent.mkdir(parents=True, exist_ok=True)
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["vec_id", "chunk_id", "doc_id", "page_num", "text"]
        )
        writer.writeheader()
        for r in rows_with_ids:
            writer.writerow(r)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a flat L2 index. Fine for small corpora.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)  # rows correspond 1:1 with vec_id indices
    return index


def _embed_all_with_cache(
    client,
    model: str,
    texts: list[str],
    cache_path: Path | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Embed all texts using the embedding cache. Returns float32 array [N, D].
    """
    if cache_path is None:
        cache_path = Path(os.getenv("EMB_CACHE_PATH", "data/emb_cache.sqlite"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def embed_fn(batch_texts: list[str]) -> list[list[float]]:
        # NOTE: this uses 'model' (bugfix from earlier 'embed_model')
        return client.embed(model=model, inputs=batch_texts)

    all_vecs: list[list[float]] = []
    total_hits = total_misses = 0
    with EmbeddingCache(cache_path) as cache:
        for i in range(0, len(texts), batch_size):
            sub = texts[i : i + batch_size]
            vecs, hits, misses = get_or_embed(cache, model, sub, embed_fn)
            total_hits += hits
            total_misses += misses
            all_vecs.extend(vecs)
    if total_hits or total_misses:
        print(f"[cache] build hits={total_hits} misses={total_misses}  ({cache_path})")

    return np.asarray(all_vecs, dtype="float32")


def build_and_save(
    chunks_csv: Path,
    index_path: Path,
    meta_csv: Path,
    client,
    embed_model: str,
    batch_size: int = 64,
) -> int:
    rows = read_chunks(chunks_csv)
    texts = [r["text"] for r in rows]

    # Use cache-aware embedding
    embs = _embed_all_with_cache(
        client, embed_model, texts, batch_size=batch_size
    )
    if embs.shape[0] != len(rows):
        raise RuntimeError("Embedding count mismatch")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path)) if (index := build_index(embs)) else None

    rows_with_ids = []
    for i, r in enumerate(rows):
        rows_with_ids.append(
            {
                "vec_id": i,
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "page_num": r["page_num"],
                "text": r["text"],
            }
        )
    write_meta(meta_csv, rows_with_ids)
    return len(rows)
