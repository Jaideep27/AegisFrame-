"""A provider that replays fixed replies. For tests and offline demos — no network."""

from __future__ import annotations

from collections.abc import Callable

from .base import ProviderError


class ScriptedProvider:
    def __init__(self, replies: list[dict] | Callable[[str, str, bytes | None], dict], name: str = "scripted"):
        self.name = name
        self._replies = replies
        self.calls: list[tuple[str, str, bytes | None]] = []

    def generate_json(self, system: str, user: str, image: bytes | None = None) -> dict:
        self.calls.append((system, user, image))
        if callable(self._replies):
            return self._replies(system, user, image)
        if not self._replies:
            raise ProviderError("scripted provider has no replies left")
        return self._replies.pop(0)
