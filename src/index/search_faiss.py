# src/index/search_faiss.py
# Purpose: load FAISS + meta, embed a query, return top-k matches with metadata.

from __future__ import annotations
from pathlib import Path
import csv
import faiss
import numpy as np

def load_meta(meta_csv: Path) -> list[dict]:
    rows = []
    with meta_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["vec_id"] = int(r["vec_id"])
            r["page_num"] = int(r["page_num"])
            rows.append(r)
    return rows

def search(index_path: Path, meta_csv: Path, client, embed_model: str, query: str, k: int = 5) -> list[dict]:
    index = faiss.read_index(str(index_path))
    meta = load_meta(meta_csv)

    # Embed the query; FAISS expects [N, D]; N=1 here.
    qvec = np.asarray(client.embed(model=embed_model, inputs=[query])[0], dtype="float32")[None, :]

    # L2: smaller distance is better; faiss returns distances + ids
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
