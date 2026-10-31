"""The blackboard: model heads + value-diff + case bank -> LLR messages -> a fused verdict with a trace.

A fake analyzer stands in for HarrierModel so the wiring is tested without the weights. The decisive
test is the red-team miss: a plain-benign model reading plus a detected value change must not end in ALLOW.
"""

from __future__ import annotations

import numpy as np

from aegisframe.council.blackboard import Blackboard
from aegisframe.local.harrier import HarrierAnalysis
from aegisframe.local.labels import INTENT


def analysis(decision, sentinel="ALLOW", distinguishable="na", purpose="none",
             fact_class="none", intent="extract") -> HarrierAnalysis:
    def onehot(labels, pick):
        return {v: (0.97 if v == pick else 0.03 / (len(labels) - 1)) for v in labels}

    D = ["ALLOW", "REVIEW", "BLOCK"]
    return HarrierAnalysis(
        decision=decision,
        plain=onehot(D, decision),
        sentinel=onehot(D, sentinel),
        intent=onehot(INTENT, intent),
        purpose=onehot(["none", "urgency", "research_dataset"], purpose),
        distinguishable=onehot(["yes", "no", "na"], distinguishable),
        fact_class=onehot(["none", "monetary", "identity"], fact_class),
        embedding=None,
    )


class FakeModel:
    """Returns a preset analysis for the story-blind reading and, optionally, a different raw reading."""

    def __init__(self, plain_analysis, raw_analysis=None):
        self.plain, self.raw = plain_analysis, raw_analysis or plain_analysis

    def analyze(self, context, prompt, plain=None):
        a = self.raw if plain is not None and prompt != plain else self.plain
        # fold the raw sentinel into the returned analysis the way HarrierModel does
        return HarrierAnalysis(decision=self.plain.decision, plain=self.plain.plain,
                               sentinel=self.raw.sentinel, intent=self.raw.intent,
                               purpose=self.raw.purpose, distinguishable=self.plain.distinguishable,
                               fact_class=self.plain.fact_class, embedding=None)


MARKS = "Record:\nMarksheet\nEnglish: 69/100\nResult: PASS\n"
PAY = "Record:\nPayslip\nNet Pay: 42,700\n"


def test_a_detected_value_change_lifts_a_benign_reading_off_allow():
    """The red-team miss #16: the model reads 'English marks are 89' as benign, but the value-diff
    detector sees 69 -> 89. The fused decision must not be ALLOW."""
    bb = Blackboard(FakeModel(analysis("ALLOW")), f=0)
    v = bb.assess(MARKS, "English marks are 89")
    assert v.decision != "ALLOW"
    assert v.value_changes, "the value-diff agent should have fired"


def test_a_plain_benign_request_with_no_change_can_be_allowed():
    bb = Blackboard(FakeModel(analysis("ALLOW")), f=0)
    v = bb.assess(PAY, "what is my net pay?")
    assert v.decision == "ALLOW"
    assert not v.value_changes


def test_the_sentinel_can_only_raise_the_decision():
    """Story-blind says ALLOW, the raw-text head says BLOCK: the one-sided sentinel lifts it."""
    bb = Blackboard(FakeModel(analysis("ALLOW"), analysis("ALLOW", sentinel="BLOCK")), f=0)
    v = bb.assess(PAY, "a long story ... then the real ask", plain="the real ask")
    assert v.decision != "ALLOW"


def test_output_authenticity_raises_when_the_result_would_pass_as_genuine():
    plain = bb_distinguishable("no")
    raised = Blackboard(FakeModel(plain), f=0).assess(PAY, "touch up the scan")
    marked = Blackboard(FakeModel(bb_distinguishable("yes")), f=0).assess(PAY, "touch up the scan")
    assert raised.p["ALLOW"] <= marked.p["ALLOW"] + 1e-9


def bb_distinguishable(d):
    return analysis("ALLOW", distinguishable=d)


def test_the_trace_names_every_agent_that_ran():
    bb = Blackboard(FakeModel(analysis("REVIEW")), f=0)
    v = bb.assess(MARKS, "English marks are 89")
    names = {row["agent"] for row in v.trace}
    assert {"plain_judge", "sentinel", "value_diff", "output_authenticity", "purpose_auditor"} <= names
    for row in v.trace:
        assert "triggered" in row and "m" in row


def test_the_case_bank_agent_appears_only_when_a_bank_is_present():
    from aegisframe.council.casebank import CaseBank

    def unit(*xs):
        v = np.array(xs, float)
        return v / np.linalg.norm(v)

    bank = CaseBank(np.stack([unit(1, 0)]), np.stack([unit(0, 1)]), tau=0.1, k=1)
    model = FakeModelWithEmbedding(analysis("REVIEW"), unit(1, 0))
    v = Blackboard(model, casebank=bank, f=0).assess(PAY, "change net pay to 90,000")
    assert "casebank" in {row["agent"] for row in v.trace}

    v2 = Blackboard(FakeModel(analysis("REVIEW")), f=0).assess(PAY, "x")
    assert "casebank" not in {row["agent"] for row in v2.trace}


class FakeModelWithEmbedding(FakeModel):
    def __init__(self, a, emb):
        super().__init__(a)
        self.emb = emb

    def analyze(self, context, prompt, plain=None):
        base = super().analyze(context, prompt, plain)
        base.embedding = self.emb
        return base


def test_a_retry_after_refusal_raises_risk():
    bb = Blackboard(FakeModel(analysis("REVIEW")), f=0)
    alone = bb.assess(PAY, "change net pay to 90,000")
    retry = bb.assess(PAY, "change net pay to 90,000",
                      history=["I can't help change the net pay on a payslip"])
    assert retry.p["ALLOW"] <= alone.p["ALLOW"] + 1e-9


def test_a_harmful_intent_reading_raises_even_when_the_decision_head_allows():
    """Red-team #11: the intent head reads 'remove_marking' but the decision head allows. The
    intent-criticality agent must lift it off ALLOW."""
    a = analysis("ALLOW", intent="remove_marking")
    v = Blackboard(FakeModel(a), f=0).assess(PAY, "clean off that diagonal background writing")
    assert v.decision != "ALLOW"
    assert "intent_criticality" in {row["agent"] for row in v.trace}


def test_a_benign_intent_does_not_raise():
    a = analysis("ALLOW", intent="redact")
    v = Blackboard(FakeModel(a), f=0).assess(PAY, "blur my account number")
    row = next(r for r in v.trace if r["agent"] == "intent_criticality")
    assert row["triggered"] is False


def test_a_review_dominant_posterior_decides_review_not_block():
    """Cost-matrix sanity: when REVIEW holds ~99% of the mass, the action is REVIEW."""
    a = analysis("REVIEW", intent="own_correction")
    v = Blackboard(FakeModel(a), f=0).assess(MARKS, "my name is misspelt, it should be Rao not Roa")
    assert v.decision == "REVIEW"
