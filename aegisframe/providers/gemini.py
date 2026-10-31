"""Google Gemini via the REST API (works on the free tier of Google AI Studio)."""

from __future__ import annotations

import base64
import os

import httpx

from .base import ProviderError, parse_json_reply

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ProviderError("GeminiProvider needs GEMINI_API_KEY")
        self.model = model or os.environ.get("AEGIS_GEMINI_MODEL", "gemini-2.5-flash")
        self.name = f"gemini:{self.model}"
        self.client = client or httpx.Client(timeout=timeout)

    def generate_json(self, system: str, user: str, image: bytes | None = None) -> dict:
        parts: list[dict] = [{"text": user}]
        if image is not None:
            parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image).decode()}})
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        }
        try:
            r = self.client.post(
                f"{BASE_URL}/models/{self.model}:generateContent",
                json=body,
                headers={"x-goog-api-key": self.api_key},
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
        if r.status_code != 200:
            raise ProviderError(f"{self.name}: HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"{self.name}: no content in reply: {str(data)[:300]}") from exc
        return parse_json_reply(text)
