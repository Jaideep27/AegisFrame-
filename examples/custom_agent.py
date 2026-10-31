"""Add your own council agent: e.g. a company list of document types you never edit.

    uv run python examples/custom_agent.py
"""

from aegisframe import AegisConfig, Guard
from aegisframe.council import Belief, CouncilAgent, register_agent
from reframebench.docs import CATALOG, render, to_png


@register_agent("no_edit_issuers")
class NoEditIssuers(CouncilAgent):
    """Raises risk for any factual edit on records from issuers the company never touches."""

    bounds = (0.5, 0.97)  # this agent only ever raises risk
    max_confidence = 0.9

    def assess(self, ctx):
        issuers = [i.lower() for i in self.options.get("issuers", [])]
        hit = any(i in ctx.doc.doc_type.lower() for i in issuers)
        if not (hit and ctx.attrs["factual_change"]):
            return self.neutral(ctx, "not-listed")
        return Belief(agent=self.name, p_harm=0.95, confidence=0.9, reason_code="protected-issuer",
                      evidence=[ctx.doc.doc_type])


config = AegisConfig.model_validate({"council": {"agents": [
    {"name": "severity"}, {"name": "fraud_patterns", "weight": 1.2}, {"name": "history"},
    {"name": "no_edit_issuers", "weight": 1.5, "options": {"issuers": ["Skylark Airways"]}},
]}})
guard = Guard(config=config)

d = guard.check_request("Change the seat to 2A", image=to_png(render(CATALOG["boarding_pass"])))
print(d.action.name, "| council:", [(b.agent, round(b.p_harm, 2), b.reason_code) for b in d.council.beliefs])
