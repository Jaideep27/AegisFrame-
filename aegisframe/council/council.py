"""The council: dynamic, sequential, probabilistic deliberation.

1. Agents are asked in order of weight. After each answer the pooled posterior is updated.
2. Sequential stopping: if the most extreme beliefs the remaining agents could still give can
   no longer move the posterior across a threshold, stop. Clear cases cost one or two agents.
3. Veto: on a CRITICAL change, one agent at or above `veto` is enough to block.
4. Communication: when a full deliberation ends with agents disagreeing by more than
   `debate.spread`, each agent sees its peers' (p, confidence, reason_code) and updates once.
5. Claim audit: only now does the purpose agent read the story. Its term is one-sided
   (max(0, ...)), so a story can raise the posterior but never lower it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import CouncilConfig
from ..schema import Action, Claim, Severity
from .agents import CouncilAgent, PurposeAgent
from .beliefs import AgentContext, Belief, CouncilVerdict
from .pooling import contribution, degroot, logit, sigmoid
from .registry import agent_class


class Council:
    def __init__(
        self,
        agents: list[CouncilAgent],
        purpose: PurposeAgent | None,
        prior: float = 0.1,
        thresholds: dict[str, float] | None = None,
        veto: float = 0.8,
        scheduling: str = "sequential",
        debate_rounds: int = 1,
        debate_spread: float = 0.3,
        openness: float = 0.3,
    ):
        self.agents = sorted(agents, key=lambda a: -a.weight)
        self.purpose = purpose
        self.prior = prior
        self.thresholds = thresholds or {"block": 0.85, "review": 0.50}
        self.veto = veto
        self.scheduling = scheduling
        self.debate_rounds = debate_rounds
        self.debate_spread = debate_spread
        self.openness = openness

    def _swing(self, rest: list[CouncilAgent]) -> tuple[float, float]:
        lp = logit(self.prior)
        up = sum(a.weight * a.max_confidence * max(0.0, logit(a.bounds[1]) - lp) for a in rest)
        down = sum(a.weight * a.max_confidence * min(0.0, logit(max(a.bounds[0], 1e-4)) - lp) for a in rest)
        return up, down

    def _action(self, posterior: float) -> Action:
        if posterior >= self.thresholds["block"]:
            return Action.BLOCK
        if posterior >= self.thresholds["review"]:
            return Action.REVIEW
        return Action.ALLOW

    def deliberate(self, ctx: AgentContext, claims: list[Claim] | tuple[Claim, ...] = ()) -> CouncilVerdict:
        weights = {a.name: a.weight for a in self.agents}
        z_block, z_review = logit(self.thresholds["block"]), logit(self.thresholds["review"])
        z = logit(self.prior)
        beliefs: list[Belief] = []
        vetoed = False

        for i, agent in enumerate(self.agents):
            belief = agent.assess(ctx)
            beliefs.append(belief)
            z += contribution(belief, agent.weight, self.prior)
            if ctx.severity == Severity.CRITICAL and belief.confidence > 0 and belief.p_harm >= self.veto:
                vetoed = True
                break
            if self.scheduling == "sequential":
                up, down = self._swing(self.agents[i + 1:])
                if z + down >= z_block or z + up < z_review or (z + down >= z_review and z + up < z_block):
                    break

        stopped_early = len(beliefs) < len(self.agents)
        rounds = 0
        informative = [b for b in beliefs if b.confidence > 0]
        if (not stopped_early and not vetoed and self.debate_rounds > 0 and len(informative) > 1
                and max(b.p_harm for b in informative) - min(b.p_harm for b in informative) > self.debate_spread):
            by_name = {a.name: a for a in self.agents}
            for _ in range(self.debate_rounds):
                defaults = degroot(beliefs, weights, self.openness)
                new = []
                for own, default in zip(beliefs, defaults):
                    peers = [b for b in beliefs if b.agent != own.agent]
                    custom = by_name[own.agent].update(ctx, own, peers, self.openness)
                    new.append(custom or default)
                beliefs = new
                rounds += 1
            z = logit(self.prior) + sum(contribution(b, weights[b.agent], self.prior) for b in beliefs)

        deliberation = sigmoid(z)
        purpose_belief = None
        # the story is read only when the plain request is already risky: a reason attached to an edit
        # that would be refused in plain form is a warning sign; a reason for a harmless edit is not
        if self.purpose is not None and claims and deliberation >= self.thresholds["review"]:
            purpose_belief = self.purpose.assess_claims(ctx, list(claims))
            # one-sided: a story can only add evidence of harm
            z += self.purpose.weight * purpose_belief.confidence * max(0.0, logit(purpose_belief.p_harm) - logit(self.prior))
        posterior = sigmoid(z)

        return CouncilVerdict(
            action=Action.BLOCK if vetoed else self._action(posterior),
            prior=self.prior, deliberation_posterior=deliberation, posterior=posterior,
            thresholds=dict(self.thresholds), vetoed=vetoed, stopped_early=stopped_early,
            agents_used=len(beliefs), agents_total=len(self.agents), rounds=rounds,
            beliefs=beliefs, purpose=purpose_belief,
        )


def build_council(cfg: CouncilConfig, providers: dict) -> Council:
    from . import agents as _builtin  # noqa: F401  (registers the built-in agents)
    from . import llm_agents as _llm  # noqa: F401  (registers the LLM agents)
    from ..local import agents as _local  # noqa: F401  (registers the local-model agents)

    agents = []
    for ac in cfg.agents:
        cls = agent_class(ac.name)
        provider = providers[ac.provider] if ac.provider else None
        if getattr(cls, "requires_provider", False) and provider is None:
            from ..config import ConfigError

            raise ConfigError(f"council agent {ac.name!r} needs a provider")
        agents.append(cls(weight=ac.weight, provider=provider, **ac.options))

    purpose = None
    if cfg.purpose.enabled:
        if cfg.purpose.provider == "local" and "local" in providers:
            from ..local.agents import LocalPurposeAgent

            purpose = LocalPurposeAgent(weight=cfg.purpose.weight, provider=providers["local"])
        elif cfg.purpose.provider:
            from .llm_agents import LLMPurposeAgent

            purpose = LLMPurposeAgent(weight=cfg.purpose.weight, provider=providers[cfg.purpose.provider])
        else:
            purpose = PurposeAgent(weight=cfg.purpose.weight)

    thresholds = {"block": cfg.thresholds.block, "review": cfg.thresholds.review}
    if cfg.calibration_file:
        data = json.loads(Path(cfg.calibration_file).read_text(encoding="utf-8"))
        thresholds.update({k: float(data[k]) for k in ("block", "review") if k in data})

    return Council(agents, purpose, prior=cfg.prior, thresholds=thresholds, veto=cfg.veto,
                   scheduling=cfg.scheduling, debate_rounds=cfg.debate.rounds,
                   debate_spread=cfg.debate.spread, openness=cfg.debate.openness)
