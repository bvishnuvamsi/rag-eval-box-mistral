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
import time
import random


# Default to Mistral's public API domain; allow override for other deployments.
DEFAULT_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai")


@dataclass
class MistralClient:
    api_key: str                         # required: we won't proceed without it
    base_url: str = DEFAULT_BASE_URL     # allows switching endpoints via env
    timeout: float = 60.0                # keeps hung calls from stalling your CLI

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
    
    def _post_json(self, path: str, payload: Dict[str, Any], max_retries: int = 6) -> Dict[str, Any]:
        """
        Robust POST with retries for free/experiment plans:
        - Retries on 429 and 5xx, honoring Retry-After when present
        - Retries on transient network errors (ReadTimeout, ConnectTimeout, etc.)
        - Exponential backoff with jitter, capped
        """
        url = f"{self.base_url}{path}"
        backoff = 1.0  # seconds
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=10)
        with httpx.Client(timeout=self.timeout, limits=limits) as client:
            for attempt in range(max_retries):
                try:
                    r = client.post(url, headers=self._headers(), json=payload)
                    if r.status_code in (429,) or 500 <= r.status_code < 600:
                        # Respect server hint if present
                        retry_after = r.headers.get("retry-after")
                        wait = float(retry_after) if retry_after else backoff
                        time.sleep(wait + random.uniform(0, 0.5))
                        backoff = min(backoff * 2, 16.0)
                        continue
                    r.raise_for_status()
                    return r.json()
                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError, httpx.RemoteProtocolError) as e:
                    # Transient network problem: back off and retry
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(backoff + random.uniform(0, 0.5))
                    backoff = min(backoff * 2, 16.0)
                except httpx.HTTPStatusError as e:
                    # Non-retriable 4xx errors should surface immediately
                    if 400 <= e.response.status_code < 500 and e.response.status_code not in (429,):
                        raise
                    # Otherwise treat as retriable (shouldn’t happen because we handle above)
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(backoff + random.uniform(0, 0.5))
                    backoff = min(backoff * 2, 16.0)
        raise httpx.HTTPError(f"Failed after {max_retries} attempts: {path}")

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        data = self._post_json("/v1/embeddings", {"model": model, "input": inputs})
        out = []
        for item in data.get("data", []):
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise ValueError("Unexpected embeddings response shape")
            out.append(emb)
        return out

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: Optional[int] = 256) -> str:
        # Cap tokens to keep responses fast/stable on free tier
        payload = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        data = self._post_json("/v1/chat/completions", payload)
        choices = data.get("choices", [])
        if not choices or "message" not in choices[0] or "content" not in choices[0]["message"]:
            raise ValueError("Unexpected chat response shape")
        return choices[0]["message"]["content"]
