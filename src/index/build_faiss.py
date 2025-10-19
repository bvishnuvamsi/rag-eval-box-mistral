# src/index/build_faiss.py
# Purpose: read chunk CSV -> embed text -> build FAISS index -> save index + metadata CSV.
from __future__ import annotations
import csv
import math
import faiss
import numpy as np
from src.index.emb_cache import EmbeddingCache, get_or_embed
from pathlib import Path

def _embed_fn_factory(client, model: str):
    # Returns a function that embeds a list[str] -> list[list[float]]
    def _fn(texts):
        # client.embed returns list[list[float]]
        return client.embed(model=model, inputs=texts)
    return _fn


def read_chunks(chunks_csv: Path):
    rows = []
    with chunks_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "page_num": int(row["page_num"]),
                "text": row["text"],
            })
    return rows

def write_meta(meta_csv: Path, rows_with_ids: list[dict]) -> None:
    meta_csv.parent.mkdir(parents=True, exist_ok=True)
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["vec_id", "chunk_id", "doc_id", "page_num", "text"])
        writer.writeheader()
        for r in rows_with_ids:
            writer.writerow(r)

def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a flat L2 index. We keep it simple in v1 (no IVF/HNSW), which is fine for small corpora.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)  # rows correspond 1:1 with vec_id indices
    return index

def embed_in_batches(
    client,
    model: str,
    texts: list[str],
    batch_size: int = 64,
    cache_path: Path | None = None,
) -> np.ndarray:
    """
    Cache-aware batch embedding. Returns float32 numpy array [N, D] in the same order as `texts`.
    """
    vectors: list[list[float]] = []
    embed_fn = _embed_fn_factory(client, model)
    cache_path = cache_path or Path("data/emb_cache.sqlite")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with EmbeddingCache(cache_path) as cache:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            embs, hits, misses = get_or_embed(
                cache=cache,
                model=model,
                texts=chunk,
                embed_fn=embed_fn,
            )
            print(f"[cache] batch={i//batch_size} hits={hits} misses={misses}")
            vectors.extend(embs)

    return np.asarray(vectors, dtype="float32")


def build_and_save(chunks_csv: Path, index_path: Path, meta_csv: Path, client, embed_model: str, batch_size: int = 64) -> int:
    rows = read_chunks(chunks_csv)
    texts = [r["text"] for r in rows]
    embs = embed_in_batches(client, embed_model, texts, batch_size=batch_size)
    if embs.shape[0] != len(rows):
        raise RuntimeError("Embedding count mismatch")

    index = build_index(embs)
    faiss.write_index(index, str(index_path))

    rows_with_ids = []
    for i, r in enumerate(rows):
        rows_with_ids.append({
            "vec_id": i,
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "page_num": r["page_num"],
            "text": r["text"],
        })
    write_meta(meta_csv, rows_with_ids)
    return len(rows)
