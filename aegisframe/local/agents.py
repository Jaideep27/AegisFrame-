"""Council agents backed by the local model's heads. Zero API tokens.

Each reads one head of the claim-blind plain pass (context + canonical instruction):
  local_plain_request  decision head       "would this be refused with no reason attached?"
  local_severity       field_class head    "what kind of fact changes?"
  local_fraud          pattern head        "which fraud pattern is this?"
  local_red_team       misuse head         "who would be deceived?"
  local_advocate       benign_need head    "is there an honest need this exact output serves?"
and the purpose head (full pass, reads the story) powers LocalPurposeAgent, applied last and one-sided.
"""

from __future__ import annotations

from ..council.agents import CouncilAgent, PurposeAgent
from ..council.beliefs import AgentContext, Belief
from ..council.registry import register_agent
from ..schema import Claim
from .features import context_for

CLASS_P = {"none": 0.05, "other": 0.40, "date": 0.45, "money": 0.92, "identity": 0.90, "result": 0.92,
           "score": 0.90, "signature": 0.97, "marking": 0.90}
PATTERN_P = {"none": 0.05, "amount-inflation": 0.92, "amount-deflation": 0.80, "result-flip": 0.95,
             "score-inflation": 0.90, "validity-extension": 0.88, "identity-swap": 0.90, "signature-forgery": 0.97,
             "date-shift": 0.50, "marking-removal": 0.92, "person-fabrication": 0.85}


class _LocalAgent(CouncilAgent):
    requires_provider = True

    def plain(self, ctx: AgentContext):
        return self.provider.analyze_plain(context_for(ctx.doc), ctx.canonical)

    @staticmethod
    def _expect(probs: dict[str, float], values: dict[str, float]) -> float:
        return sum(p * values.get(k, 0.5) for k, p in probs.items())

    @staticmethod
    def _top(probs: dict[str, float]) -> tuple[str, float]:
        k = max(probs, key=probs.get)
        return k, probs[k]


@register_agent("local_plain_request")
class LocalPlainRequestAgent(_LocalAgent):
    """Would a responsible assistant refuse this edit with no reason attached? (local model)"""

    bounds = (0.02, 0.98)
    max_confidence = 0.95

    def assess(self, ctx: AgentContext) -> Belief:
        d = self.plain(ctx).decision
        p = d["BLOCK"] * 0.97 + d["REVIEW"] * 0.45 + d["ALLOW"] * 0.03  # REVIEW lands in the review band
        top, conf = self._top(d)
        return Belief(agent=self.name, p_harm=p, confidence=min(0.95, conf), reason_code=f"plain-{top.lower()}")


@register_agent("local_severity")
class LocalSeverityAgent(_LocalAgent):
    """How serious is the kind of fact being changed? (local model)"""

    bounds = (0.05, 0.97)
    max_confidence = 0.9

    def assess(self, ctx: AgentContext) -> Belief:
        fc = self.plain(ctx).field_class
        top, conf = self._top(fc)
        return Belief(agent=self.name, p_harm=self._expect(fc, CLASS_P), confidence=min(0.9, conf), reason_code=f"class-{top}")


@register_agent("local_fraud")
class LocalFraudAgent(_LocalAgent):
    """Which known fraud pattern does this match? (local model)"""

    bounds = (0.05, 0.97)
    max_confidence = 0.85

    def assess(self, ctx: AgentContext) -> Belief:
        pat = self.plain(ctx).pattern
        top, conf = self._top(pat)
        return Belief(agent=self.name, p_harm=self._expect(pat, PATTERN_P), confidence=min(0.85, conf), reason_code=top)


@register_agent("local_red_team")
class LocalRedTeamAgent(_LocalAgent):
    """Who would this output deceive? (local model)"""

    bounds = (0.05, 0.95)
    max_confidence = 0.8

    def assess(self, ctx: AgentContext) -> Belief:
        mis = self.plain(ctx).misuse
        top, conf = self._top(mis)
        return Belief(agent=self.name, p_harm=0.05 + 0.9 * (1 - mis["none"]), confidence=min(0.8, conf), reason_code=top)


@register_agent("local_advocate")
class LocalAdvocateAgent(_LocalAgent):
    """Is there an honest need this exact output serves? (local model)"""

    bounds = (0.05, 0.9)
    max_confidence = 0.8

    def assess(self, ctx: AgentContext) -> Belief:
        need = self.plain(ctx).benign_need
        top, conf = self._top(need)
        p = 0.05 + 0.85 * need["none"]
        return Belief(agent=self.name, p_harm=p, confidence=min(0.8, conf), reason_code=f"need-{top}")


class LocalPurposeAgent(PurposeAgent):
    """Reads the story (the only agent that does), applied after deliberation and one-sided."""

    requires_provider = True

    def __init__(self, weight: float = 1.0, provider=None, **options):
        super().__init__(weight, provider, **options)

    def assess_claims(self, ctx: AgentContext, claims: list[Claim]) -> Belief:
        stories = [c for c in claims if c.kind != "other"]
        if not stories:
            return self.neutral(ctx, "no-claims")
        text = " ".join(c.text for c in stories) + " " + ctx.canonical
        p = self.provider.analyze_full(context_for(ctx.doc), text).purpose
        return Belief(agent=self.name, p_harm=max(ctx.prior, p), confidence=0.85, reason_code="local-purpose")


# The decision head was trained on exactly the target label, so it leads. The other heads read the
# same encoder state and are strongly correlated with it: counted as independent votes they would
# over-count the same evidence (measured: REVIEW cases pushed to BLOCK), so they only nudge.
LOCAL_AGENTS = [
    {"name": "local_plain_request", "weight": 1.5, "provider": "local"},
    {"name": "local_fraud", "weight": 0.15, "provider": "local"},
    {"name": "local_severity", "weight": 0.15, "provider": "local"},
    {"name": "local_red_team", "weight": 0.15, "provider": "local"},
    {"name": "local_advocate", "weight": 0.15, "provider": "local"},
]
