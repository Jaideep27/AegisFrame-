"""Agent registry: built-ins and custom agents are configured by name.

    from aegisframe.council import CouncilAgent, Belief, register_agent

    @register_agent("sanctions_list")
    class SanctionsAgent(CouncilAgent):
        def assess(self, ctx):
            ...
            return Belief(agent=self.name, p_harm=0.9, confidence=0.7, reason_code="sanctioned-entity")
"""

from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register_agent(name: str):
    def decorator(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def agent_class(name: str) -> type:
    try:
        return _REGISTRY[name]
    except KeyError:
        from ..config import ConfigError

        raise ConfigError(f"unknown council agent {name!r}; registered: {sorted(_REGISTRY)}") from None


def registered() -> list[str]:
    return sorted(_REGISTRY)
