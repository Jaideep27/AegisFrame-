"""AegisConfig: every behaviour of the guard as a typed, validated attribute.

Load order: defaults < YAML file < environment variables (AEGIS__SECTION__KEY=value).

    aegis.yaml
    ----------
    policy_file: policy.yaml
    providers:
      gemini: {kind: gemini, model: gemini-2.5-flash}
    separator: {mode: agreement, providers: [gemini]}
    council:
      agents:
        - {name: severity, weight: 1.0}
        - {name: fraud_patterns, weight: 1.2}
        - {name: plain_request, weight: 1.5, provider: gemini}
    review: {on_review: raise}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ConfigError(ValueError):
    pass


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderConfig(_Section):
    kind: Literal["gemini", "openai_compat", "claude_cli", "codex_cli"]
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None  # name of the env var holding the key; keys never live in config files
    retries: int = 2


class SeparatorConfig(_Section):
    mode: Literal["rules", "llm", "agreement"] = "rules"
    providers: list[str] = Field(default_factory=list)


class GroundingConfig(_Section):
    ocr: Literal["rapid", "none"] = "rapid"


class AgentConfig(_Section):
    name: str
    weight: float = 1.0
    provider: str | None = None
    options: dict = Field(default_factory=dict)


class DebateConfig(_Section):
    rounds: int = 1
    spread: float = 0.30  # run a round when beliefs disagree by more than this
    openness: float = 0.3  # how far an agent moves toward its peers in one round


class ThresholdConfig(_Section):
    block: float = 0.85
    review: float = 0.50


class PurposeConfig(_Section):
    enabled: bool = True
    weight: float = 1.0
    provider: str | None = None


class CouncilConfig(_Section):
    enabled: bool = True
    prior: float = 0.10
    agents: list[AgentConfig] = Field(
        default_factory=lambda: [
            AgentConfig(name="severity", weight=1.0),
            AgentConfig(name="fraud_patterns", weight=1.2),
            AgentConfig(name="history", weight=1.0),
        ]
    )
    scheduling: Literal["sequential", "all"] = "sequential"
    debate: DebateConfig = Field(default_factory=DebateConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    veto: float = 0.80  # any agent at or above this on a CRITICAL change forces BLOCK
    purpose: PurposeConfig = Field(default_factory=PurposeConfig)
    calibration_file: str | None = None


class DecisionConfig(_Section):
    fail_mode: Literal["closed", "open"] = "closed"
    claim_blind: bool = True  # research ablation only: False lets council agents read the story


class ReviewConfig(_Section):
    on_review: Literal["raise", "block", "allow"] = "raise"
    queue_path: str | None = None  # SQLite file; None = no persistent queue
    secret_env: str = "AEGIS_REVIEW_SECRET"
    webhook_url: str | None = None


class MemoryConfig(_Section):
    backend: Literal["memory", "sqlite"] = "memory"
    path: str | None = None


class AuditConfig(_Section):
    sink: Literal["memory", "jsonl", "sqlite"] = "memory"
    path: str | None = None


class PrivacyConfig(_Section):
    store_claims: Literal["hash", "text", "none"] = "hash"


class VerifyConfig(_Section):
    enabled: bool = True


class LocalModelConfig(_Section):
    enabled: Literal["auto", "on", "off"] = "auto"  # auto: use the local model if one is installed
    path: str | None = None  # default: $AEGIS_LOCAL_MODEL or ~/.cache/aegisframe/models/aegis-local
    quantized: bool = True
    threads: int | None = None
    trust_threshold: float = 0.85  # above this intent confidence the model's parse is used alone


class ServerConfig(_Section):
    upstream: str = "https://api.openai.com/v1"
    trust_identity_header: bool = False


class AegisConfig(_Section):
    policy_file: str | None = None
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    separator: SeparatorConfig = Field(default_factory=SeparatorConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    council: CouncilConfig = Field(default_factory=CouncilConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    local_model: LocalModelConfig = Field(default_factory=LocalModelConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, env: dict[str, str] | None = None) -> AegisConfig:
        data: dict = {}
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ConfigError(f"{path}: top level must be a mapping")
            data = {k: v for k, v in raw.items() if v is not None}  # an empty section means "defaults"
            base = Path(path).parent
            if data.get("policy_file") and not Path(data["policy_file"]).is_absolute():
                data["policy_file"] = str(base / data["policy_file"])
            lm = data.get("local_model") or {}
            if isinstance(lm, dict) and lm.get("path") and not Path(lm["path"]).is_absolute():
                lm["path"] = str((base / lm["path"]).resolve())
        _apply_env(data, os.environ if env is None else env)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc


def _coerce(value: str):
    low = value.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(value) if low.lstrip("-").isdigit() else float(value)
    except ValueError:
        return value


def _apply_env(data: dict, env) -> None:
    for key, value in env.items():
        if not key.startswith("AEGIS__"):
            continue
        parts = [p.lower() for p in key[len("AEGIS__"):].split("__") if p]
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if parts:
            node[parts[-1]] = _coerce(value)
