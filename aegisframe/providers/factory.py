"""Build named providers from config. Keys are read from the environment, never from config files."""

from __future__ import annotations

import os

from ..config import AegisConfig, ConfigError, ProviderConfig
from .base import Provider


def build_provider(name: str, pc: ProviderConfig) -> Provider:
    key = os.environ.get(pc.api_key_env) if pc.api_key_env else None
    if pc.kind == "gemini":
        from .gemini import GeminiProvider

        provider: Provider = GeminiProvider(api_key=key, model=pc.model)
    elif pc.kind == "openai_compat":
        from .openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider(base_url=pc.base_url, api_key=key, model=pc.model)
    elif pc.kind == "claude_cli":
        from .cli import ClaudeCLIProvider

        provider = ClaudeCLIProvider(model=pc.model)
    elif pc.kind == "codex_cli":
        from .cli import CodexCLIProvider

        provider = CodexCLIProvider(model=pc.model)
    else:  # pragma: no cover - guarded by the config schema
        raise ConfigError(f"unknown provider kind {pc.kind}")
    if pc.retries:
        from .retry import RetryingProvider

        provider = RetryingProvider(provider, attempts=pc.retries + 1)
    return provider


def build_providers(cfg: AegisConfig) -> dict[str, Provider]:
    return {name: build_provider(name, pc) for name, pc in cfg.providers.items()}
