# src/index/emb_cache.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Callable
import sqlite3, hashlib, json

class EmbeddingCache:
    """
    SQLite cache: (model, text_hash) -> embedding JSON.
    Auto-creates & auto-migrates the table (drops old schema if needed).
    """
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "EmbeddingCache":
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._ensure_schema()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.conn:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def _ensure_schema(self) -> None:
        cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
        exists = cur.fetchone() is not None
        if not exists:
            self._create_schema()
            return
        # Ensure new column exists; if not, drop & recreate (safe: it's only a cache)
        cols = [row[1] for row in self.conn.execute("PRAGMA table_info(embeddings)").fetchall()]
        if "text_hash" not in cols:
            self.conn.execute("DROP TABLE embeddings")
            self._create_schema()

    def _create_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE embeddings (
              model     TEXT NOT NULL,
              text_hash TEXT NOT NULL,
              dim       INTEGER NOT NULL,
              vec       TEXT NOT NULL,
              PRIMARY KEY(model, text_hash)
            )
            """
        )

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: List[str]) -> List[Optional[List[float]]]:
        if not texts:
            return []
        hashes = [self._hash_text(t) for t in texts]
        placeholders = ",".join(["?"] * len(hashes))
        rows = self.conn.execute(
            f"""
            SELECT text_hash, dim, vec
            FROM embeddings
            WHERE model=? AND text_hash IN ({placeholders})
            """,
            [model, *hashes],
        ).fetchall()
        found = {h: (dim, json.loads(vec_json)) for (h, dim, vec_json) in rows}
        out: List[Optional[List[float]]] = []
        for h in hashes:
            tup = found.get(h)
            out.append(tup[1] if tup else None)
        return out

    def put_many(self, model: str, texts: List[str], vecs: List[List[float]]) -> None:
        if not texts:
            return
        assert len(texts) == len(vecs)
        dim = len(vecs[0]) if vecs else 0
        rows = [(model, self._hash_text(t), dim, json.dumps(v)) for t, v in zip(texts, vecs)]
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings(model, text_hash, dim, vec) VALUES (?, ?, ?, ?)",
            rows,
        )

def get_or_embed(
    cache: EmbeddingCache,
    model: str,
    texts: List[str],
    embed_fn: Callable[[List[str]], List[List[float]]],
) -> Tuple[List[List[float]], int, int]:
    """
    Returns (vectors, hits, misses)
    """
    cached = cache.get_many(model, texts)
    to_embed_idx = [i for i, v in enumerate(cached) if v is None]
    hits = len(texts) - len(to_embed_idx)
    misses = len(to_embed_idx)

    if to_embed_idx:
        need = [texts[i] for i in to_embed_idx]
        new_vecs = embed_fn(need)
        cache.put_many(model, need, new_vecs)
        # Merge, preserving order
        out: List[List[float]] = []
        j = 0
        for i in range(len(texts)):
            if cached[i] is None:
                out.append(new_vecs[j]); j += 1
            else:
                out.append(cached[i])  # type: ignore
    else:
        out = cached  # type: ignore

    return out, hits, misses
