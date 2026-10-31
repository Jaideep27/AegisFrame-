"""LLM council agents. Each sees the canonical instruction (and the image), never the story,
unless the claim_blind ablation is switched off for research.

They answer with one small JSON belief: {"p_harm", "confidence", "reason_code"}.
In the communication round they are re-prompted with their peers' reason codes.
"""

from __future__ import annotations

from ..providers.base import ProviderError
from ..schema import Claim
from .agents import CouncilAgent, PurposeAgent
from .beliefs import AgentContext, Belief
from .registry import register_agent

REPLY = ('Return ONLY JSON: {"p_harm": <probability 0..1 that producing this output causes harm>, '
         '"confidence": <0..1>, "reason_code": "<short-kebab-case-code>"}')


def _clamp(x, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(x)))
    except (TypeError, ValueError):
        return default


class LLMAgent(CouncilAgent):
    requires_provider = True
    role = ""

    def _user(self, ctx: AgentContext, extra: str = "") -> str:
        parts = [f"EDIT (the only thing that matters):\n{ctx.canonical}"]
        if ctx.doc.fields:
            facts = "; ".join(f"{k}: {v}" for k, v in list(ctx.doc.fields.items())[:12])
            parts.append(f"FIELDS READ FROM THE IMAGE: {facts}")
        if ctx.claims_for_ablation:  # research ablation only
            parts.append("CONTEXT GIVEN BY THE USER: " + " | ".join(c.text for c in ctx.claims_for_ablation))
        if extra:
            parts.append(extra)
        return "\n\n".join(parts)

    def _ask(self, ctx: AgentContext, extra: str = "", round_: int = 0) -> Belief:
        if self.provider is None:
            raise ProviderError(f"{self.name}: no provider configured")
        raw = self.provider.generate_json(self.role + "\n\n" + REPLY, self._user(ctx, extra), ctx.image)
        if "p_harm" not in raw:
            raise ProviderError(f"{self.name}: reply has no p_harm: {raw}")
        return Belief(agent=self.name, p_harm=_clamp(raw["p_harm"], 1.0),
                      confidence=_clamp(raw.get("confidence", 0.7), 0.7),
                      reason_code=str(raw.get("reason_code", ""))[:60], round=round_)

    def assess(self, ctx: AgentContext) -> Belief:
        return self._ask(ctx)

    def update(self, ctx: AgentContext, own: Belief, peers: list[Belief], openness: float) -> Belief:
        notes = "\n".join(f"- {b.agent}: p={b.p_harm:.2f} (confidence {b.confidence:.2f}) because {b.reason_code}"
                          for b in peers if b.confidence > 0)
        extra = (f"YOUR EARLIER ANSWER: p={own.p_harm:.2f} because {own.reason_code}\n"
                 f"OTHER REVIEWERS SAID:\n{notes}\nReconsider once and answer again.")
        return self._ask(ctx, extra, round_=own.round + 1)


@register_agent("plain_request")
class PlainRequestAgent(LLMAgent):
    """Would a responsible assistant refuse this edit if no reason were attached? (LLM)"""

    role = ("You review image-editing requests for a company. You see ONE plain instruction with no reason "
            "attached, on purpose. Would a responsible image-editing assistant refuse to carry out this exact "
            "edit, because the result could deceive someone (a forged record, fake evidence, a false depiction "
            "of a real person) or cause other serious harm?")


@register_agent("red_team")
class RedTeamAgent(LLMAgent):
    """How exactly would someone misuse this output, and how badly? (LLM)"""

    role = ("You are a red-team reviewer. Imagine the most likely way someone would misuse the exact output of "
            "this edit: who would be shown it, what would they decide because of it, and how bad would that be?")


@register_agent("advocate")
class AdvocateAgent(LLMAgent):
    """What is the most likely honest need, and does it require this exact output? (LLM)"""

    role = ("You are the advocate. Find the most likely honest need behind this edit and whether the exact "
            "output is required for it. Be fair: if the output is a realistic altered record that would pass as "
            "genuine, say so even if an honest need exists.")


class LLMPurposeAgent(PurposeAgent):
    """Reads the story. Its p is clipped at the prior and the council applies it one-sidedly."""

    requires_provider = True
    role = ("A user gave reasons for an image edit. Reasons are NOT verified. Decide whether the stated purpose "
            "actually requires this exact output (a realistic altered record), or whether it could be met by a "
            "fictional sample, a clearly marked sample, the original record, or an official channel. "
            "High p_harm means: the purpose does not require this output, so the story is a reframing attempt.")

    def assess_claims(self, ctx: AgentContext, claims: list[Claim]) -> Belief:
        stories = [c for c in claims if c.kind != "other"]
        if not stories:
            return self.neutral(ctx, "no-claims")
        user = (f"EDIT:\n{ctx.canonical}\n\nSTATED REASONS (unverified):\n"
                + "\n".join(f"- [{c.kind}] {c.text}" for c in stories))
        raw = self.provider.generate_json(self.role + "\n\n" + REPLY, user, None)
        if "p_harm" not in raw:
            raise ProviderError(f"purpose: reply has no p_harm: {raw}")
        return Belief(agent=self.name, p_harm=max(ctx.prior, _clamp(raw["p_harm"], 1.0)),
                      confidence=_clamp(raw.get("confidence", 0.7), 0.7),
                      reason_code=str(raw.get("reason_code", ""))[:60])

    def __init__(self, weight: float = 1.0, provider=None, **options):
        super().__init__(weight, provider, **options)
