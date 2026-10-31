import pytest

from aegisframe.floor import LLMJudge, compute_floor
from aegisframe.policy import Policy
from aegisframe.providers.base import ProviderError
from aegisframe.providers.scripted import ScriptedProvider
from aegisframe.schema import Action, EditSpec, FieldChange, GroundedDoc, JudgeVote, Severity, VisualChange

SLIP = GroundedDoc(
    has_image=True, is_document=True, is_evidence=True, real_source=True,
    doc_type="Orbitline Software Pvt Ltd - Payslip for September 2026",
    fields={"Employee Name": "Arjun Mehta", "Net Pay": "Rs. 52,000", "Annual CTC": "Rs. 7,20,000"},
)
PASS = GroundedDoc(
    has_image=True, is_document=True, is_evidence=True, real_source=True,
    doc_type="Skylark Airways - Boarding Pass", fields={"Passenger": "Meera Iyer", "Seat": "14C"},
)
PHOTO = GroundedDoc(has_image=True, doc_type="photo")
policy = Policy.default()


class FixedJudge:
    def __init__(self, p):
        self.name, self.p, self.seen = "fixed", p, []

    def vote(self, canonical, image):
        self.seen.append(canonical)
        return JudgeVote(judge=self.name, p_refuse=self.p)


def test_salary_raise_is_blocked_as_forged_evidence():
    spec = EditSpec(field_changes=[FieldChange(kind="replace", field="compensation", new="95,000")])
    floor = compute_floor(spec, SLIP, policy)
    assert floor.action == Action.BLOCK and floor.rule == "forged-evidence"
    assert floor.severity == Severity.HIGH
    assert floor.resolved[0].field in {"Net Pay", "Annual CTC"}


def test_canonical_instruction_uses_document_values():
    spec = EditSpec(field_changes=[FieldChange(kind="replace", old="52000", new="95,000")])
    floor = compute_floor(spec, SLIP, policy)
    assert "'Net Pay' from 'Rs. 52,000' to '95,000'" in floor.canonical_instruction
    assert floor.canonical_instruction.startswith("On this real ")


def test_seat_change_is_moderate_and_reviewed():
    spec = EditSpec(field_changes=[FieldChange(kind="replace", field="seat", new="2A")])
    floor = compute_floor(spec, PASS, policy)
    assert (floor.action, floor.severity) == (Action.REVIEW, Severity.MODERATE)


def test_photo_colour_change_is_allowed():
    spec = EditSpec(visual_changes=[VisualChange(target="outfit", attribute="color", value="red")])
    assert compute_floor(spec, PHOTO, policy).action == Action.ALLOW


def test_judges_can_raise_the_floor():
    spec = EditSpec(visual_changes=[VisualChange(target="outfit", attribute="color", value="red")])
    assert compute_floor(spec, PHOTO, policy, judges=[FixedJudge(0.9)]).action == Action.BLOCK
    assert compute_floor(spec, PHOTO, policy, judges=[FixedJudge(0.35)]).action == Action.REVIEW


def test_judges_cannot_lower_the_floor():
    spec = EditSpec(field_changes=[FieldChange(kind="replace", field="Net Pay", new="95,000")])
    assert compute_floor(spec, SLIP, policy, judges=[FixedJudge(0.0)]).action == Action.BLOCK


def test_judges_only_ever_see_the_canonical_instruction():
    judge = FixedJudge(0.1)
    spec = EditSpec(field_changes=[FieldChange(kind="replace", field="Net Pay", new="95,000")])
    floor = compute_floor(spec, SLIP, policy, judges=[judge])
    assert judge.seen == [floor.canonical_instruction]


def test_llm_judge_reads_probability():
    provider = ScriptedProvider([{"p_refuse": 0.83, "reason": "forgery"}])
    vote = LLMJudge(provider).vote("On this real payslip, change ...", b"img")
    assert vote.p_refuse == pytest.approx(0.83)
    system, user, image = provider.calls[0]
    assert "change ..." in user and image == b"img"


def test_llm_judge_clamps_and_rejects_garbage():
    assert LLMJudge(ScriptedProvider([{"p_refuse": 7}])).vote("x", None).p_refuse == 1.0
    with pytest.raises(ProviderError):
        LLMJudge(ScriptedProvider([{"verdict": "no"}])).vote("x", None)
