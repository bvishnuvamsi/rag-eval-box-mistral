# src/ingest/make_seed_pdfs.py
# Purpose: generate two tiny PDFs locally so we can test the pipeline w/o downloading anything.
# Why PyMuPDF? It's already installed as part of pymupdf and lets us write simple PDFs quickly.

from __future__ import annotations
from pathlib import Path
import fitz  # PyMuPDF

def make_seed_pdfs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- PDF 1: "company_policy.pdf" ---
    policy = fitz.open()                     # create an empty PDF
    page = policy.new_page()                 # add 1 page
    text = (
        "ACME Corp Employee Handbook\n\n"
        "PTO Policy: Full-time employees accrue 15 days per year.\n"
        "Sick Leave: 10 days per year.\n"
        "WFH Policy: Up to 3 days per week with manager approval.\n"
        "Security: Use strong passwords and enable 2FA.\n"
    )
    page.insert_text((72, 72), text, fontsize=12)  # place text at (72,72) points
    policy.save(out_dir / "company_policy.pdf")
    policy.close()

    # --- PDF 2: "api_guide.pdf" ---
    guide = fitz.open()
    p1 = guide.new_page()
    t1 = (
        "ACME API Guide\n\n"
        "GET /v1/users: Returns a list of users.\n"
        "POST /v1/users: Creates a new user (fields: name, email).\n"
        "Auth: Use a Bearer token in the Authorization header.\n"
        "Rate Limits: 60 requests/minute.\n"
    )
    p1.insert_text((72, 72), t1, fontsize=12)
    guide.save(out_dir / "api_guide.pdf")
    guide.close()
