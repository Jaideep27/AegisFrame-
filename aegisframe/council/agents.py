"""Council agents. Each one reads the claim-blind context and returns a Belief.

Built-in offline agents: severity, fraud_patterns, history (deliberation) and purpose
(the one-sided claim audit applied after deliberation). LLM agents live in llm_agents.py.
"""

from __future__ import annotations

from ..schema import Claim, Severity
from .beliefs import AgentContext, Belief
from .kb import load_patterns, match
from .registry import register_agent


class CouncilAgent:
    """Base class. Subclasses implement assess(); register with @register_agent("name").

    bounds / max_confidence describe the most extreme belief the agent can ever give.
    The sequential scheduler uses them to know when the remaining agents can no longer
    change the decision.
    """

    name = "agent"
    bounds: tuple[float, float] = (0.02, 0.98)
    max_confidence: float = 1.0

    def __init__(self, weight: float = 1.0, provider=None, **options):
        self.weight = weight
        self.provider = provider
        self.options = options

    def assess(self, ctx: AgentContext) -> Belief:  # pragma: no cover - abstract
        raise NotImplementedError

    def update(self, ctx: AgentContext, own: Belief, peers: list[Belief], openness: float) -> Belief | None:
        """Return an updated belief after seeing peers, or None for the default DeGroot update."""
        return None

    def neutral(self, ctx: AgentContext, reason: str) -> Belief:
        return Belief(agent=self.name, p_harm=ctx.prior, confidence=0.0, reason_code=reason)


def _genuine_factual(ctx: AgentContext) -> bool:
    return ctx.attrs["factual_change"] and ctx.attrs["passes_as_genuine"] and ctx.attrs["real_source"]


SEVERITY_P = {Severity.NONE: 0.05, Severity.LOW: 0.30, Severity.MODERATE: 0.40, Severity.HIGH: 0.90,
              Severity.CRITICAL: 0.97}


@register_agent("severity")
class SeverityAgent(CouncilAgent):
    """How serious is the kind of fact being changed, if the result passes as genuine?"""

    bounds = (0.05, 0.97)
    max_confidence = 0.9

    def assess(self, ctx: AgentContext) -> Belief:
        if ctx.edit.real_person_false_act and ctx.attrs["passes_as_genuine"]:
            return Belief(agent=self.name, p_harm=0.9, confidence=0.9, reason_code="real-person-fabrication")
        if not _genuine_factual(ctx):
            reason = "marked-or-fictional" if ctx.attrs["factual_change"] else "no-factual-change"
            return Belief(agent=self.name, p_harm=0.05, confidence=0.9, reason_code=reason)
        confidence = 0.9 if ctx.doc.is_evidence else 0.3
        return Belief(agent=self.name, p_harm=SEVERITY_P[ctx.severity], confidence=confidence,
                      reason_code=f"severity-{ctx.severity.name.lower()}",
                      evidence=sorted({r.field_class for r in ctx.resolved}))


@register_agent("fraud_patterns")
class FraudPatternAgent(CouncilAgent):
    """Does the change match a known scam pattern? Cites the pattern."""

    bounds = (0.30, 0.98)  # every pattern is evidence of harm: this agent only ever raises
    max_confidence = 0.8

    def __init__(self, weight: float = 1.0, provider=None, kb: str | None = None, **options):
        super().__init__(weight, provider, **options)
        self.patterns = load_patterns(kb)

    def assess(self, ctx: AgentContext) -> Belief:
        if not _genuine_factual(ctx):
            return self.neutral(ctx, "no-pattern")
        m = match(self.patterns, ctx.resolved)
        if m is None:
            return self.neutral(ctx, "no-pattern")
        return Belief(agent=self.name, p_harm=m.p, confidence=0.8 if ctx.doc.is_evidence else 0.4,
                      reason_code=m.pattern.id, evidence=m.evidence + [m.pattern.note])


@register_agent("history")
class HistoryAgent(CouncilAgent):
    """Is this a retry after a halt, or a forgery being built up in steps?"""

    bounds = (0.50, 0.85)  # history signals only ever raise
    max_confidence = 0.8

    def assess(self, ctx: AgentContext) -> Belief:
        if not _genuine_factual(ctx):  # a genuinely different output (marked, fictional) is not a retry
            return self.neutral(ctx, "no-history-signal")
        if ctx.halted_before:
            return Belief(agent=self.name, p_harm=0.8, confidence=0.8, reason_code="retry-after-halt")
        if ctx.prior_turns >= 2:
            return Belief(agent=self.name, p_harm=0.7, confidence=0.6, reason_code="stepwise-edits")
        return self.neutral(ctx, "no-history-signal")


PURPOSE_REASON = {
    "purpose": "purpose-achievable-without-forgery",
    "context": "purpose-achievable-without-forgery",
    "authority": "unverified-authority",
    "role": "unverified-authority",
    "urgency": "pressure-tactic",
    "emotional": "pressure-tactic",
}


class PurposeAgent(CouncilAgent):
    """The ONLY agent that reads the story, and it can only raise risk.

    Question: does the stated purpose actually require a realistic altered record?
    Research, teaching, demos and disputes are served by fictional or marked samples or by
    the original record, so an unverified purpose attached to a forgery is a reframing signal.
    """

    name = "purpose"

    def assess(self, ctx: AgentContext) -> Belief:
        return self.neutral(ctx, "no-claims")

    def assess_claims(self, ctx: AgentContext, claims: list[Claim]) -> Belief:
        stories = [c for c in claims if c.kind != "other"]
        if not stories or not _genuine_factual(ctx):
            return self.neutral(ctx, "no-claims" if not stories else "nothing-to-justify")
        reasons = sorted({PURPOSE_REASON.get(c.kind, "unverified-claim") for c in stories})
        return Belief(agent=self.name, p_harm=0.85, confidence=0.8, reason_code=reasons[0],
                      evidence=[f"{c.kind}: claim not verified, output not necessary" for c in stories])
