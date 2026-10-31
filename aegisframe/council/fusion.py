"""Log-likelihood-ratio messages and a log-linear fusion pool with the two guarantees from the plan.

Each agent writes a calibrated LLR message relative to ALLOW, prior-corrected:

    m_c = log P(y=c | view) / P(y=A | view)  -  log pi_c / pi_A        c in {REVIEW, BLOCK}

The pool adds them in log space (a product of experts when weights are 1), turns the result into a
posterior, and decides by minimum Bayes risk. Two properties are proven in tests/test_fusion.py:

  Theorem 1 (story monotonicity): a one-sided message enters only through max(0, m) with weight >= 0,
    so adding one can only raise P(BLOCK)+P(REVIEW) and lower P(ALLOW). A story never helps.
  Theorem 2 (f-fault-tolerant ALLOW): ALLOW is returned only if it survives removing the f strongest
    allow-ward contributions, so no f agents can by themselves talk the pool into ALLOW.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DECISION = ["ALLOW", "REVIEW", "BLOCK"]  # ALLOW is the reference class (index 0)
_NONREF = DECISION[1:]  # REVIEW, BLOCK - the two components of a message

#: Bayes cost C[action][true class]. Allowing a true BLOCK is by far the worst; a REVIEW always costs
#: one unit of human effort. Rows/cols ordered ALLOW, REVIEW, BLOCK.
DEFAULT_COSTS = np.array([
    [0.0, 1.5, 40.0],    # action ALLOW: letting a true BLOCK through is by far the worst outcome
    [1.0, 1.0, 1.0],     # action REVIEW: a human look always costs one unit
    [3.0, 1.0, 0.0],     # action BLOCK: refusing a benign request is bad but recoverable
])

_EPS = 1e-9


@dataclass
class Message:
    """An agent's LLR vote. `m` is (m_REVIEW, m_BLOCK) relative to ALLOW; `u` is its uncertainty."""

    agent: str
    m: np.ndarray
    one_sided: bool = False
    u: float = 0.0

    def __post_init__(self) -> None:
        self.m = np.asarray(self.m, dtype=float).reshape(2)


def message_from_probs(agent: str, probs: dict[str, float], prior: dict[str, float],
                       one_sided: bool = False, u: float = 0.0) -> Message:
    """Build a message from an agent's class probabilities and the shared prior."""
    p = np.array([max(_EPS, probs[c]) for c in DECISION])
    pi = np.array([max(_EPS, prior[c]) for c in DECISION])
    llr = np.log(p[1:] / p[0]) - np.log(pi[1:] / pi[0])
    return Message(agent=agent, m=llr, one_sided=one_sided, u=u)


@dataclass
class Fused:
    decision: str
    p: dict[str, float]
    eta: np.ndarray
    robust_decision: str
    contributions: dict[str, np.ndarray] = field(default_factory=dict)


def _posterior(eta: np.ndarray) -> np.ndarray:
    z = np.concatenate([[0.0], eta])  # ALLOW is the reference, log-odds 0
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _bayes_action(p: np.ndarray, costs: np.ndarray) -> str:
    return DECISION[int((costs @ p).argmin())]


def fuse(messages: list[Message], prior: dict[str, float], weights: dict[str, float] | None = None,
         costs: np.ndarray | None = None, f: int = 1) -> Fused:
    """Fuse LLR messages into a decision. `f` is the fault tolerance for the ALLOW guard (Theorem 2)."""
    weights = weights or {}
    costs = DEFAULT_COSTS if costs is None else costs
    pi = np.array([max(_EPS, prior[c]) for c in DECISION])
    base = np.log(pi[1:] / pi[0])

    contributions: dict[str, np.ndarray] = {}
    eta = base.copy()
    for msg in messages:
        w = weights.get(msg.agent, 1.0)
        if msg.one_sided and w < 0:
            raise ValueError(f"one-sided agent {msg.agent!r} must have weight >= 0, got {w}")
        contrib = w * (np.maximum(0.0, msg.m) if msg.one_sided else msg.m)
        contributions[msg.agent] = contrib
        eta = eta + contrib

    p = _posterior(eta)
    decision = _bayes_action(p, costs)

    # Theorem 2: remove the f agents whose contribution pushed hardest toward ALLOW (most negative sum),
    # and only keep an ALLOW that survives.
    allow_ward = sorted(contributions.items(), key=lambda kv: kv[1].sum())
    robust_eta = eta.copy()
    for _agent, contrib in allow_ward[:max(0, f)]:
        robust_eta = robust_eta - np.minimum(0.0, contrib)  # add back the allow-ward (negative) part
    robust_decision = _bayes_action(_posterior(robust_eta), costs)

    final = decision if decision != "ALLOW" or robust_decision == "ALLOW" else robust_decision
    return Fused(decision=final, p=dict(zip(DECISION, p.tolist())), eta=eta,
                 robust_decision=robust_decision, contributions=contributions)
