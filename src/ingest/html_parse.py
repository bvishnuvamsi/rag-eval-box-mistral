# src/ingest/html_parse.py
# Purpose: turn raw HTML into clean text lines we can chunk/index.

from __future__ import annotations
from pathlib import Path
import csv
from bs4 import BeautifulSoup
import re


BLOCK_TAGS = {"p", "li", "pre", "code"}
HEADER_TAGS = {"h1", "h2", "h3"}

HASH_SUFFIX = re.compile(r"__(?P<h>[0-9a-f]{8})\.html$", re.IGNORECASE)

def _derive_doc_id(html_file: Path) -> str:
    name = html_file.name
    if name.endswith(".html"):
        base = name[:-5]  # drop ".html"
        m = HASH_SUFFIX.search(name)
        if m:
            # remove "__<hash>.html" from the end
            return name[:m.start()]
        return base
    return html_file.stem

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Strip obvious noise
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Heuristic: drop nav/aside/footer if present
    for tag in soup.find_all(["nav", "aside", "footer"]):
        tag.decompose()

    lines: list[str] = []

    # Keep headings to preserve structure
    for h in soup.find_all(list(HEADER_TAGS)):
        txt = h.get_text(" ", strip=True)
        if txt:
            lines.append(f"# {txt}")

    # Then add body blocks in order
    for tag in soup.find_all(True):
        name = tag.name.lower()
        if name in BLOCK_TAGS:
            txt = tag.get_text(" ", strip=True)
            if txt:
                lines.append(txt)

    text = "\n".join(lines)
    return text

def parse_dir(raw_dir: Path, out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "page_num", "text"])
        writer.writeheader()
        for html_file in sorted(raw_dir.glob("*.html")):
            # doc_id = original URL proxy: use filename without hash suffix
            # e.g., docs.stripe.com__api__<hash>.html -> docs.stripe.com__api
            stem = html_file.name[:-14] if html_file.name.endswith(".html") else html_file.stem
            #doc_id = stem  # stable id for citations
            doc_id = _derive_doc_id(html_file)
            html = html_file.read_text(encoding="utf-8", errors="ignore")
            text = html_to_text(html)
            if not text.strip():
                continue
            writer.writerow({"doc_id": doc_id, "page_num": 1, "text": text})
            rows += 1
    return rows
