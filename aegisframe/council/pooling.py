"""Log-odds pooling and the DeGroot communication round.

pool():   logit(post) = logit(prior) + sum_i w_i * c_i * (logit(p_i) - logit(prior))
          Each agent is a likelihood-ratio update relative to the prior, scaled by its weight and
          confidence. Monotone non-decreasing in every p_i. Unlike an average, one confident
          objection is not diluted by several mild votes.

degroot(): each informative agent moves a fraction `openness` toward the confidence-weighted
          mean of its peers. One round, so beliefs converge without collapsing.
"""

from __future__ import annotations

import math

from .beliefs import Belief

EPS = 1e-4


def logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def contribution(belief: Belief, weight: float, prior: float) -> float:
    return weight * belief.confidence * (logit(belief.p_harm) - logit(prior))


def pool(prior: float, items: list[tuple[Belief, float]]) -> float:
    return sigmoid(logit(prior) + sum(contribution(b, w, prior) for b, w in items))


def degroot(beliefs: list[Belief], weights: dict[str, float], openness: float) -> list[Belief]:
    updated = []
    for me in beliefs:
        peers = [b for b in beliefs if b.agent != me.agent and b.confidence > 0]
        mass = sum(weights.get(b.agent, 1.0) * b.confidence for b in peers)
        if me.confidence == 0 or mass == 0:
            updated.append(me.model_copy(update={"round": me.round + 1}))
            continue
        peer_mean = sum(weights.get(b.agent, 1.0) * b.confidence * b.p_harm for b in peers) / mass
        p = (1 - openness) * me.p_harm + openness * peer_mean
        updated.append(me.model_copy(update={"p_harm": p, "round": me.round + 1}))
    return updated
