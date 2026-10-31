"""Model provider interface. Every LLM/VLM call in AegisFrame goes through this."""

from __future__ import annotations

import json
import re
from typing import Protocol


class ProviderError(RuntimeError):
    """Any failure talking to a model. The guard fails closed on it."""


class Provider(Protocol):
    name: str

    def generate_json(self, system: str, user: str, image: bytes | None = None) -> dict: ...


def parse_json_reply(text: str) -> dict:
    """Models sometimes wrap JSON in prose or code fences; take the outermost object."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ProviderError(f"no JSON object in model reply: {text[:200]!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ProviderError(f"bad JSON in model reply: {exc}") from exc
