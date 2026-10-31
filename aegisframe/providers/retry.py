"""Retry with exponential backoff and jitter, so a rate limit does not instantly fail closed."""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from .base import Provider, ProviderError


class RetryingProvider:
    def __init__(self, inner: Provider, attempts: int = 3, base_delay: float = 0.5,
                 sleep: Callable[[float], None] = time.sleep):
        self.inner = inner
        self.name = inner.name
        self.attempts = max(1, attempts)
        self.base_delay = base_delay
        self._sleep = sleep

    def generate_json(self, system: str, user: str, image: bytes | None = None) -> dict:
        last: ProviderError | None = None
        for attempt in range(self.attempts):
            try:
                return self.inner.generate_json(system, user, image)
            except ProviderError as exc:
                last = exc
                if attempt + 1 < self.attempts:
                    self._sleep(self.base_delay * (2 ** attempt) * (1 + random.random() * 0.25))
        assert last is not None
        raise last
