"""The log-likelihood-ratio message protocol and the log-linear fusion pool.

The two theorems from the plan are property tests here, checked on random message sets:
  Theorem 1 (story monotonicity): a one-sided agent can only raise risk, never lower it.
  Theorem 2 (f-fault-tolerant ALLOW): no f agents can by themselves force an ALLOW.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegisframe.council.fusion import DECISION, Message, fuse, message_from_probs

PRIOR = {"ALLOW": 0.45, "REVIEW": 0.1, "BLOCK": 0.45}


# ---------------------------------------------------------------- the LLR message
def test_a_message_matching_the_prior_is_zero():
    m = message_from_probs("a", PRIOR, PRIOR)
    assert np.allclose(m.m, 0.0, atol=1e-9)


def test_more_block_than_prior_gives_a_positive_block_message():
    m = message_from_probs("a", {"ALLOW": 0.1, "REVIEW": 0.1, "BLOCK": 0.8}, PRIOR)
    assert m.m[DECISION.index("BLOCK") - 1] > 0


def test_more_allow_than_prior_gives_a_negative_block_message():
    m = message_from_probs("a", {"ALLOW": 0.9, "REVIEW": 0.05, "BLOCK": 0.05}, PRIOR)
    assert m.m[DECISION.index("BLOCK") - 1] < 0


# ---------------------------------------------------------------- fusion basics
def test_no_messages_returns_the_prior():
    f = fuse([], PRIOR)
    for c in DECISION:
        assert f.p[c] == pytest.approx(PRIOR[c], abs=1e-9)


def test_one_confident_block_agent_moves_the_posterior_up():
    f = fuse([message_from_probs("a", {"ALLOW": 0.02, "REVIEW": 0.03, "BLOCK": 0.95}, PRIOR)], PRIOR)
    assert f.p["BLOCK"] > PRIOR["BLOCK"]
    assert f.decision == "BLOCK"


def test_the_posterior_is_a_distribution():
    msgs = [message_from_probs("a", {"ALLOW": 0.3, "REVIEW": 0.3, "BLOCK": 0.4}, PRIOR),
            message_from_probs("b", {"ALLOW": 0.8, "REVIEW": 0.1, "BLOCK": 0.1}, PRIOR)]
    f = fuse(msgs, PRIOR)
    assert sum(f.p.values()) == pytest.approx(1.0)
    assert all(0 <= v <= 1 for v in f.p.values())


# ---------------------------------------------------------------- Theorem 1: story monotonicity
def _random_messages(rng, n):
    msgs = []
    for i in range(n):
        p = rng.dirichlet([1, 1, 1])
        probs = dict(zip(DECISION, p))
        msgs.append(message_from_probs(f"a{i}", probs, PRIOR, one_sided=bool(rng.integers(0, 2))))
    return msgs


def test_theorem_1_a_one_sided_agent_never_lowers_risk():
    """The true invariant: adding a one-sided agent never raises P(ALLOW) - risk only rises.
    (It is NOT true for P(BLOCK) alone, since a one-sided agent can shift BLOCK mass into REVIEW;
    see the ledger ruling. Risk = 1 - P(ALLOW) is the quantity the story-blindness guarantee is about.)"""
    rng = np.random.default_rng(0)
    for _ in range(300):
        base = _random_messages(rng, rng.integers(0, 5))
        before = fuse(base, PRIOR)
        story = message_from_probs("story", dict(zip(DECISION, rng.dirichlet([1, 1, 1]))), PRIOR,
                                   one_sided=True)
        after = fuse([*base, story], PRIOR)
        assert after.p["ALLOW"] <= before.p["ALLOW"] + 1e-9, "a story raised P(ALLOW)"
        if before.decision != "ALLOW":
            assert after.decision != "ALLOW", "a story flipped the decision to ALLOW"


def test_theorem_1_a_one_sided_agent_never_lowers_p_block_plus_review():
    """The allow mass can only shrink: P(ALLOW) never rises when a one-sided agent is added."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        base = _random_messages(rng, rng.integers(0, 5))
        before = fuse(base, PRIOR).p["ALLOW"]
        story = message_from_probs("s", dict(zip(DECISION, rng.dirichlet([1, 1, 1]))), PRIOR, one_sided=True)
        after = fuse([*base, story], PRIOR).p["ALLOW"]
        assert after <= before + 1e-9


# ---------------------------------------------------------------- Theorem 2: f-fault-tolerant ALLOW
def test_theorem_2_no_single_agent_can_force_allow():
    """With f=1, one maximally allow-ward agent cannot by itself produce ALLOW when the rest would not."""
    rng = np.random.default_rng(2)
    forced = 0
    for _ in range(300):
        others = _random_messages(rng, rng.integers(1, 5))
        liar = Message(agent="liar", m=np.array([-50.0, -50.0]), one_sided=False)
        f = fuse([*others, liar], PRIOR, f=1)
        if f.decision == "ALLOW":
            # it may only ALLOW because the OTHERS already allow without the liar
            without = fuse(others, PRIOR, f=0).decision
            assert without == "ALLOW", "one agent forced an ALLOW the rest did not support"
            forced += 1
    assert forced >= 0  # the assertion inside the loop is the real check


def test_theorem_2_f0_lets_a_single_agent_allow():
    """Sanity: with f=0 the fault tolerance is off, so one strong allow agent can swing it."""
    others = [message_from_probs("a", {"ALLOW": 0.1, "REVIEW": 0.1, "BLOCK": 0.8}, PRIOR)]
    liar = Message(agent="liar", m=np.array([-50.0, -50.0]), one_sided=False)
    assert fuse([*others, liar], PRIOR, f=0).decision == "ALLOW"
    assert fuse([*others, liar], PRIOR, f=1).decision != "ALLOW"


# ---------------------------------------------------------------- Bayes-risk decision
def test_allowing_a_harmful_request_is_the_costliest_mistake():
    """A borderline block-ish posterior must not resolve to ALLOW under the default costs."""
    f = fuse([message_from_probs("a", {"ALLOW": 0.55, "REVIEW": 0.1, "BLOCK": 0.35}, PRIOR)], PRIOR)
    assert f.decision in ("REVIEW", "BLOCK")


def test_a_clearly_benign_request_is_allowed_when_agents_agree():
    """ALLOW needs more than f agents behind it: two corroborating benign votes survive f=1."""
    benign = {"ALLOW": 0.95, "REVIEW": 0.03, "BLOCK": 0.02}
    msgs = [message_from_probs("a", benign, PRIOR), message_from_probs("b", benign, PRIOR)]
    assert fuse(msgs, PRIOR, f=1).decision == "ALLOW"


def test_a_lone_benign_vote_is_reviewed_not_allowed_under_f1():
    """The cost of fault tolerance: one agent cannot by itself reach ALLOW, so it falls to REVIEW."""
    f = fuse([message_from_probs("a", {"ALLOW": 0.97, "REVIEW": 0.02, "BLOCK": 0.01}, PRIOR)], PRIOR, f=1)
    assert f.decision == "REVIEW"
    assert fuse([message_from_probs("a", {"ALLOW": 0.97, "REVIEW": 0.02, "BLOCK": 0.01}, PRIOR)],
                PRIOR, f=0).decision == "ALLOW"


# ---------------------------------------------------------------- weights
def test_a_zero_weight_agent_is_ignored():
    msg = message_from_probs("a", {"ALLOW": 0.02, "REVIEW": 0.03, "BLOCK": 0.95}, PRIOR)
    assert fuse([msg], PRIOR, weights={"a": 0.0}).p["BLOCK"] == pytest.approx(PRIOR["BLOCK"], abs=1e-9)


def test_a_one_sided_weight_cannot_be_negative():
    with pytest.raises(ValueError):
        fuse([Message("a", np.array([1.0, 1.0]), one_sided=True)], PRIOR, weights={"a": -1.0})
