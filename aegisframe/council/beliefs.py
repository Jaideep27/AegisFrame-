"""What agents say to each other: probabilities with confidence and a reason code, never prose."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..schema import Action, Claim, EditSpec, GroundedDoc, ResolvedChange, Severity


class Belief(BaseModel):
    agent: str
    p_harm: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)  # 0 = "no information" (neutral)
    reason_code: str = ""
    evidence: list[str] = Field(default_factory=list)
    round: int = 0


class CouncilVerdict(BaseModel):
    action: Action
    prior: float
    deliberation_posterior: float  # claim-blind
    posterior: float  # after the one-sided purpose term
    thresholds: dict[str, float]
    vetoed: bool = False
    stopped_early: bool = False
    agents_used: int = 0
    agents_total: int = 0
    rounds: int = 0
    beliefs: list[Belief] = Field(default_factory=list)
    purpose: Belief | None = None


@dataclass
class AgentContext:
    """Everything an agent may look at. There is deliberately no claims field:
    the story reaches only the purpose agent, through Council.deliberate(claims=...)."""

    edit: EditSpec
    doc: GroundedDoc
    attrs: dict[str, bool]
    resolved: list[ResolvedChange]
    severity: Severity
    canonical: str
    prior: float = 0.1
    image: bytes | None = None
    halted_before: bool = False
    prior_turns: int = 0
    claims_for_ablation: tuple[Claim, ...] = field(default_factory=tuple)  # empty unless claim_blind=False
