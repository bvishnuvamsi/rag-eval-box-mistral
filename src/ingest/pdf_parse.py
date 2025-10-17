# src/ingest/pdf_parse.py
# Purpose: read PDFs, extract plain text per page, and save a CSV with metadata.

from __future__ import annotations
from pathlib import Path
import csv
import fitz  # PyMuPDF

def extract_pages(pdf_path: Path):
    """Yield dicts: {'doc_id', 'page_num', 'text'} for each page in a PDF."""
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):                # 1-based pages (friendlier)
            text = page.get_text("text") or ""                  # extract plain text
            yield {"doc_id": pdf_path.name, "page_num": i, "text": text}
    finally:
        doc.close()

def parse_dir(input_dir: Path, out_csv: Path) -> int:
    """
    Parse all .pdf files in input_dir and write rows to out_csv.
    Returns the number of pages written.
    """
    pdfs = sorted([p for p in input_dir.glob("*.pdf") if p.is_file()])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "page_num", "text"])
        writer.writeheader()
        for pdf in pdfs:
            for row in extract_pages(pdf):
                writer.writerow(row)
                count += 1
    return count
