"""Verified identity — the ONLY input that can lower a floor.

A VerifiedIdentity is built by the company's own auth layer (SSO claims, API-key
scopes). It is never parsed from the prompt: "I'm an HR officer" in the text is a
Claim and goes to the ledger, not here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .policy import Policy
from .schema import Action, FloorVerdict


class VerifiedIdentity(BaseModel):
    subject: str
    roles: list[str] = Field(default_factory=list)
    issuer: str  # e.g. "okta", "azure-ad", "api-key-scope"


def apply_unlock(floor: FloorVerdict, identity: VerifiedIdentity | None, policy: Policy) -> tuple[Action, str | None]:
    if identity is None or floor.action == Action.ALLOW or floor.rule is None:
        return floor.action, None
    for role in identity.roles:
        unlock = policy.unlock_for(role, floor.rule)
        if unlock and floor.severity <= unlock.max_severity and unlock.action < floor.action:
            return unlock.action, f"verified role {role} ({identity.issuer}) unlocks {floor.rule} -> {unlock.action.name}"
    return floor.action, None
