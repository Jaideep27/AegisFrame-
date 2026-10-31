import pytest

from aegisframe.policy import Policy, PolicyError, attributes
from aegisframe.schema import Action, EditSpec, FieldChange, GroundedDoc, Marking, Severity

EVIDENCE = GroundedDoc(has_image=True, is_document=True, is_evidence=True, real_source=True, doc_type="salary slip")
PHOTO = GroundedDoc(has_image=True, is_document=False, is_evidence=False, real_source=True, doc_type="photo")
CHANGE = EditSpec(field_changes=[FieldChange(kind="replace", field="Net Pay", new="95,000")])
MARKED = EditSpec(
    field_changes=CHANGE.field_changes,
    marking=Marking(kind="watermark", text="SAMPLE", covers_changed_fields=True),
)

policy = Policy.default()


def test_forged_evidence_high_severity_blocks():
    action, rule, _ = policy.evaluate(attributes(EVIDENCE, CHANGE), Severity.HIGH, EVIDENCE.doc_type)
    assert (action, rule) == (Action.BLOCK, "forged-evidence")


def test_forged_evidence_moderate_severity_goes_to_review():
    action, rule, _ = policy.evaluate(attributes(EVIDENCE, CHANGE), Severity.MODERATE, EVIDENCE.doc_type)
    assert (action, rule) == (Action.REVIEW, "forged-evidence")


def test_enforced_marking_means_no_forgery():
    action, rule, _ = policy.evaluate(attributes(EVIDENCE, MARKED), Severity.HIGH, EVIDENCE.doc_type)
    assert (action, rule) == (Action.ALLOW, None)


def test_photo_edit_is_allowed():
    action, _, _ = policy.evaluate(attributes(PHOTO, EditSpec()), Severity.NONE, "photo")
    assert action == Action.ALLOW


def test_real_person_fabrication_blocks():
    spec = EditSpec(real_person_false_act=True)
    action, rule, _ = policy.evaluate(attributes(PHOTO, spec), Severity.NONE, "photo")
    assert (action, rule) == (Action.BLOCK, "real-person-fabrication")


def test_override_can_only_tighten():
    p = Policy.from_dict(
        {
            "rules": Policy.default().raw["rules"],
            "overrides": [
                {"match": {"doc_type_contains": "hall ticket"}, "action": "block"},
                {"match": {"doc_type_contains": "salary"}, "action": "allow"},
            ],
        }
    )
    hall = EVIDENCE.model_copy(update={"doc_type": "Exam Hall Ticket"})
    assert p.evaluate(attributes(hall, CHANGE), Severity.MODERATE, hall.doc_type)[0] == Action.BLOCK
    # an override that tries to loosen is ignored: the rule still blocks
    assert p.evaluate(attributes(EVIDENCE, CHANGE), Severity.HIGH, "salary slip")[0] == Action.BLOCK


def test_unknown_attribute_in_rule_is_rejected():
    with pytest.raises(PolicyError):
        Policy.from_dict({"rules": [{"id": "x", "when": {"is_purple": True}, "action": "block"}]})


def test_policy_has_verified_role_unlocks():
    unlock = policy.unlock_for("hr_verification_officer", "forged-evidence")
    assert unlock is not None and unlock.max_severity == Severity.HIGH
    assert policy.unlock_for("random_role", "forged-evidence") is None


def test_unparsed_edit_on_a_record_goes_to_review():
    action, rule, _ = policy.evaluate(attributes(EVIDENCE, EditSpec()), Severity.NONE, EVIDENCE.doc_type)
    assert (action, rule) == (Action.REVIEW, "unparsed-record-edit")


def test_unparsed_edit_on_a_photo_is_allowed():
    action, _, _ = policy.evaluate(attributes(PHOTO, EditSpec()), Severity.NONE, "photo")
    assert action == Action.ALLOW


def test_reframing_escalation_is_a_per_rule_flag():
    assert policy.escalates("forged-evidence") is True
    assert policy.escalates("unparsed-record-edit") is False
    assert policy.escalates(None) is True
