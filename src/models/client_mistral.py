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
    
    def _post_json(self, path: str, payload: dict, max_retries: int = 8):
        import time, httpx, os
        from httpx import Limits

        url = self.base_url + path
        backoff = 1.0
        rate_limit_seconds = float(os.getenv("MISTRAL_RATE_LIMIT_SECONDS", "0"))  # e.g. 0.5 or 1.0
        limits = Limits(max_connections=10, max_keepalive_connections=10, keepalive_expiry=5.0)
        last_exc = None

        with httpx.Client(timeout=self.timeout, limits=limits) as client:
            for attempt in range(max_retries):
                try:
                    if rate_limit_seconds > 0:
                        time.sleep(rate_limit_seconds)

                    r = client.post(url, headers=self._headers(), json=payload)

                    if r.status_code < 400:
                        return r.json()

                    # Graceful handling for rate limits and transient errors
                    if r.status_code in (429, 503):
                        retry_after = r.headers.get("Retry-After")
                        if retry_after and retry_after.replace(".", "", 1).isdigit():
                            wait = float(retry_after)
                        else:
                            wait = backoff
                        time.sleep(wait)
                        backoff = min(backoff * 2, 32.0)
                        continue

                    # Other 4xx/5xx: raise with body snippet for debugging
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:200]}", request=r.request, response=r)

                except Exception as e:
                    last_exc = e
                    time.sleep(min(backoff, 8.0))
                    backoff = min(backoff * 2, 32.0)

        raise httpx.HTTPError(f"Failed after {max_retries} attempts: {path} ({last_exc})")


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
