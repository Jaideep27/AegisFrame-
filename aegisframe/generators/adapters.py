"""Generator adapters. None of them needs a vendor SDK: REST via httpx, or duck typing."""

from __future__ import annotations

import base64
import io
import os
from collections.abc import Callable

import httpx

from .base import GeneratorError, as_png


class CallableGenerator:
    """Wrap any `fn(prompt, image=None, **kw) -> bytes`."""

    def __init__(self, fn: Callable, name: str = "callable"):
        self.fn = fn
        self.name = name

    def generate(self, prompt: str, image: bytes | None, **kwargs) -> bytes:
        return as_png(self.fn(prompt, image, **kwargs))


def _post(client: httpx.Client, name: str, url: str, **kw) -> dict:
    try:
        r = client.post(url, **kw)
    except httpx.HTTPError as exc:
        raise GeneratorError(f"{name}: {exc}") from exc
    if r.status_code != 200:
        raise GeneratorError(f"{name}: HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


class OpenAIImagesGenerator:
    """OpenAI Images API (also Azure OpenAI and compatible servers): /images/edits and /images/generations."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-image-1", base_url: str | None = None,
                 client: httpx.Client | None = None, timeout: float = 180.0, **defaults):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self.defaults = defaults
        self.name = f"openai-images:{model}"

    def generate(self, prompt: str, image: bytes | None, **kwargs) -> bytes:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        params = {"model": self.model, **self.defaults, **kwargs}
        if image is not None:
            data = {"prompt": prompt, **{k: str(v) for k, v in params.items()}}
            body = _post(self.client, self.name, f"{self.base_url}/images/edits", headers=headers, data=data,
                         files={"image": ("image.png", image, "image/png")})
        else:
            body = _post(self.client, self.name, f"{self.base_url}/images/generations", headers=headers,
                         json={"prompt": prompt, "n": 1, **params})
        try:
            item = body["data"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise GeneratorError(f"{self.name}: no image in reply") from exc
        if item.get("b64_json"):
            return as_png(base64.b64decode(item["b64_json"]))
        if item.get("url"):
            return as_png(self.client.get(item["url"]).content)
        raise GeneratorError(f"{self.name}: no image in reply")


class GeminiImageGenerator:
    """Gemini image models via generateContent with responseModalities IMAGE."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash-image",
                 client: httpx.Client | None = None, timeout: float = 180.0):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        self.client = client or httpx.Client(timeout=timeout)
        self.name = f"gemini-image:{model}"

    def generate(self, prompt: str, image: bytes | None, **kwargs) -> bytes:
        parts: list[dict] = [{"text": prompt}]
        if image is not None:
            parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image).decode()}})
        body = {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], **kwargs}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        reply = _post(self.client, self.name, url, json=body, headers={"x-goog-api-key": self.api_key or ""})
        for cand in reply.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return as_png(base64.b64decode(inline["data"]))
        raise GeneratorError(f"{self.name}: the model returned no image")


class HTTPGenerator:
    """Any internal JSON endpoint: {prompt_field: str, image_field: base64} -> base64 at response_path."""

    def __init__(self, url: str, headers: dict | None = None, prompt_field: str = "prompt",
                 image_field: str = "image", response_path: str = "image", extra: dict | None = None,
                 client: httpx.Client | None = None, timeout: float = 180.0):
        self.url, self.headers = url, headers or {}
        self.prompt_field, self.image_field, self.response_path = prompt_field, image_field, response_path
        self.extra = extra or {}
        self.client = client or httpx.Client(timeout=timeout)
        self.name = f"http:{url}"

    def generate(self, prompt: str, image: bytes | None, **kwargs) -> bytes:
        body = {self.prompt_field: prompt, **self.extra, **kwargs}
        if image is not None:
            body[self.image_field] = base64.b64encode(image).decode()
        node = _post(self.client, self.name, self.url, json=body, headers=self.headers)
        for key in self.response_path.split("."):
            if not isinstance(node, dict) or key not in node:
                raise GeneratorError(f"{self.name}: no {self.response_path!r} in reply")
            node = node[key]
        return as_png(base64.b64decode(node))


class DiffusersGenerator:
    """A local open-source pipeline (diffusers or anything with the same call shape)."""

    def __init__(self, pipe, **call_kwargs):
        self.pipe = pipe
        self.call_kwargs = call_kwargs
        self.name = f"local:{type(pipe).__name__}"

    def generate(self, prompt: str, image: bytes | None, **kwargs) -> bytes:
        kw = {**self.call_kwargs, **kwargs}
        if image is not None:
            from PIL import Image

            kw["image"] = Image.open(io.BytesIO(image)).convert("RGB")
        out = self.pipe(prompt=prompt, **kw)
        images = getattr(out, "images", None)
        if not images:
            raise GeneratorError(f"{self.name}: pipeline returned no images")
        return as_png(images[0])
