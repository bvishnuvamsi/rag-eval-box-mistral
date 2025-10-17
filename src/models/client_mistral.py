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
    
    def _post_json(self, path: str, payload: dict, max_retries: int = 6) -> dict:
        """
        POST JSON with simple exponential backoff for 429/5xx.
        Respects Retry-After seconds if present. Jitter prevents thundering herd.
        """
        url = f"{self.base_url}{path}"
        backoff = 1.0  # seconds
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(max_retries):
                r = client.post(url, headers=self._headers(), json=payload)
                # Success: return parsed JSON
                if r.status_code < 400:
                    return r.json()

                # Rate limit or transient server error → backoff & retry
                if r.status_code in (429, 500, 502, 503, 504):
                    # Respect Retry-After header when present
                    retry_after = r.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        sleep_s = float(retry_after)
                    else:
                        # Exponential backoff with jitter, capped at ~20s
                        sleep_s = min(backoff, 20.0) + random.uniform(0, 0.5)
                        backoff *= 2

                    # Last attempt? raise
                    if attempt == max_retries - 1:
                        r.raise_for_status()
                    time.sleep(sleep_s)
                    continue

                # Other client errors: raise immediately (bad key, bad request, etc.)
                r.raise_for_status()

        # Should never hit here
        raise RuntimeError("Unexpected retry loop exit")

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        data = self._post_json("/v1/embeddings", {"model": model, "input": inputs})
        out = []
        for item in data.get("data", []):
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise ValueError("Unexpected embeddings response shape")
            out.append(emb)
        return out

    def chat(self, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int | None = None) -> str:
        payload = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        data = self._post_json("/v1/chat/completions", payload)
        choices = data.get("choices", [])
        if not choices or "message" not in choices[0] or "content" not in choices[0]["message"]:
            raise ValueError("Unexpected chat response shape")
        return choices[0]["message"]["content"]
