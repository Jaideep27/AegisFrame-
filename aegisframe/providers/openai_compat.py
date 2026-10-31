"""Any OpenAI-compatible chat endpoint: OpenAI, Azure OpenAI, Groq, NVIDIA NIM, vLLM, Ollama."""

from __future__ import annotations

import base64
import os

import httpx

from .base import ProviderError, parse_json_reply


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        json_mode: bool = True,
    ):
        self.base_url = (base_url or os.environ.get("AEGIS_OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("AEGIS_OPENAI_MODEL", "gpt-4.1-mini")
        self.name = f"openai-compat:{self.model}"
        self.json_mode = json_mode
        self.client = client or httpx.Client(timeout=timeout)

    def generate_json(self, system: str, user: str, image: bytes | None = None) -> dict:
        content: list[dict] = [{"type": "text", "text": user}]
        if image is not None:
            url = "data:image/png;base64," + base64.b64encode(image).decode()
            content.append({"type": "image_url", "image_url": {"url": url}})
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": 0,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = self.client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
        if r.status_code != 200:
            raise ProviderError(f"{self.name}: HTTP {r.status_code}: {r.text[:300]}")
        try:
            text = r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name}: no content in reply") from exc
        return parse_json_reply(text or "")
