# src/ingest/chunkers.py
# Purpose: split long page text into fixed-size chunks (v1: no overlap).
# Why no overlap? Keep v1 dumb and measurable; we add overlap later and compare recall.

from __future__ import annotations
from pathlib import Path
import csv

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 0):
    """
    Yield strings of length up to chunk_size from text.
    Overlap is 0 in v1; we will add sliding-window later.
    """
    if overlap != 0:
        raise ValueError("v1 chunker uses overlap=0; we'll add sliding windows later.")
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        yield text[start:end]
        start = end  # no overlap in v1

def make_chunks(docs_csv: Path, chunks_csv: Path, chunk_size: int = 800) -> int:
    """
    Read docs_csv (doc_id,page_num,text) and write chunks_csv with:
    chunk_id, doc_id, page_num, text
    """
    chunks_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with docs_csv.open("r", encoding="utf-8") as fin, chunks_csv.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["chunk_id", "doc_id", "page_num", "text"])
        writer.writeheader()

        for row in reader:
            doc_id = row["doc_id"]
            page_num = int(row["page_num"])
            text = row["text"] or ""
            # For each page, emit chunk(s)
            local_i = 0
            for chunk in chunk_text(text, chunk_size=chunk_size, overlap=0):
                chunk_id = f"{doc_id}__p{page_num}__c{local_i}"
                writer.writerow({"chunk_id": chunk_id, "doc_id": doc_id, "page_num": page_num, "text": chunk})
                local_i += 1
                count += 1
    return count
