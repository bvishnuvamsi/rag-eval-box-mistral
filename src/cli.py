# src/cli.py
# Purpose: your CLI front door. We start with two commands:
#  - ping: verify your environment and FAISS work
#  - api-check: verify your Mistral API key and list available models

from __future__ import annotations
#from models.client_mistral import MistralClient


import os
import sys
import typer  # zero-boilerplate CLI
from dotenv import load_dotenv  # load .env into env vars
import httpx  # only for friendly error printing in api-check


app = typer.Typer(help="RAG Eval Box CLI")


@app.command()
def ping() -> None:
    """
    Sanity check: shows Python/FAISS versions and whether MISTRAL_API_KEY is set.
    Exits non-zero if FAISS is broken so you don't waste time later.
    """
    load_dotenv()  # read .env if present (safe no-op if missing)

    # Don't import FAISS at module import time; fail fast only when you run ping.
    try:
        import faiss  # noqa: F401
        faiss_version = getattr(faiss, "__version__", "unknown")
    except Exception as e:
        typer.secho(f"FAISS import failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    key_set = bool(os.getenv("MISTRAL_API_KEY"))
    typer.echo(f"Python: {sys.version.split()[0]}")
    typer.echo(f"FAISS:  {faiss_version}")
    typer.echo("MISTRAL_API_KEY set: " + ("yes" if key_set else "no"))


@app.command("api-check")
def api_check() -> None:
    """
    Calls Mistral /v1/models to verify your API key and connectivity.
    Prints up to 10 model IDs on success.
    """
    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        typer.secho("Missing MISTRAL_API_KEY in .env (or environment).", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Import here so unit tests can stub the client without importing the world.
    from .models.client_mistral import MistralClient

    client = MistralClient(api_key=key)

    try:
        models = client.list_models()
    except httpx.HTTPStatusError as e:
        # Show status code and a slice of server text so debugging is fast.
        status = e.response.status_code
        body = e.response.text[:300].replace("\n", " ")
        typer.secho(f"API error {status}: {body}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except httpx.RequestError as e:
        typer.secho(f"Network error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"OK. {len(models)} model(s) visible:", fg=typer.colors.GREEN)
    for m in models[:10]:
        typer.echo(f" - {m}")


if __name__ == "__main__":
    app()
