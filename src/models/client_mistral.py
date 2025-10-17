# src/models/client_mistral.py
# Purpose: a tiny, safe wrapper around Mistral's HTTP API.
# Design choices:
#  - No heavy SDKs: fewer surprises, easy to read.
#  - Short timeouts + explicit errors: fail fast if the key/URL is wrong.
#  - Base URL is overridable via env (MISTRAL_BASE_URL) for later (e.g., Azure).

from __future__ import annotations  # allow future typing features in 3.11

import os
from dataclasses import dataclass  # compact, immutable-like config object
from typing import List
import httpx  # we picked this over requests for better timeouts/async

# Default to Mistral's public API domain; allow override for other deployments.
DEFAULT_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai")


@dataclass
class MistralClient:
    api_key: str                         # required: we won't proceed without it
    base_url: str = DEFAULT_BASE_URL     # allows switching endpoints via env
    timeout: float = 15.0                # keeps hung calls from stalling your CLI

    def _headers(self) -> dict:
        # Standard bearer token auth and JSON content negotiation.
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def list_models(self) -> List[str]:
        """
        Smoke test: list models visible to your key.
        Returns a list of model IDs (strings). If unauthorized, we raise with details.
        """
        url = f"{self.base_url}/v1/models"  # minimal, safe endpoint for a health check
        # Use a context manager so the HTTP connection is closed cleanly.
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(url, headers=self._headers())
            # If 401/403/5xx, this throws httpx.HTTPStatusError with server text.
            r.raise_for_status()
            data = r.json()
        # Expected shape: {"data": [{"id": "mistral-large-latest"}, ...]}
        items = data.get("data", [])
        return [item.get("id", "unknown") for item in items]
