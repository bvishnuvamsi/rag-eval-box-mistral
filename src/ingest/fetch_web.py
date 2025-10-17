# src/ingest/fetch_web.py
# Purpose: download a curated set of docs (HTML) and save them locally.
# We keep it explicit and respectful (timeouts, no aggressive crawling).

from __future__ import annotations
from pathlib import Path
import hashlib
import httpx

def _safe_name(url: str) -> str:
    # Stable, filesystem-safe name: host__path__sha8.html
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return url.replace("https://", "").replace("http://", "").replace("/", "__") + f"__{h}.html"

def fetch_all(sources_file: Path, out_dir: Path, timeout: float = 15.0) -> list[tuple[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, Path]] = []
    with sources_file.open("r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            name = _safe_name(url)
            dest = out_dir / name
            # GET with short timeout; no retries to be polite.
            with httpx.Client(timeout=timeout, headers={"User-Agent": "rag-eval-box/0.1"}) as client:
                r = client.get(url)
                r.raise_for_status()
                dest.write_text(r.text, encoding="utf-8")
            pairs.append((url, dest))
    return pairs
