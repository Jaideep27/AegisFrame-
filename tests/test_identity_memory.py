import io

from PIL import Image

from aegisframe.identity import VerifiedIdentity, apply_unlock
from aegisframe.memory import SessionMemory, image_key
from aegisframe.policy import Policy
from aegisframe.schema import Action, EditSpec, FieldChange, FloorVerdict, Severity
from reframebench.docs import CATALOG, render, render_photo, to_png

policy = Policy.default()


def floor(action=Action.BLOCK, severity=Severity.HIGH, rule="forged-evidence"):
    return FloorVerdict(action=action, severity=severity, rule=rule, canonical_instruction="x")


def test_no_identity_no_unlock():
    assert apply_unlock(floor(), None, policy) == (Action.BLOCK, None)


def test_verified_role_unlocks_up_to_its_ceiling():
    who = VerifiedIdentity(subject="u1", roles=["hr_verification_officer"], issuer="okta")
    action, reason = apply_unlock(floor(), who, policy)
    assert action == Action.REVIEW and "hr_verification_officer" in reason


def test_verified_role_does_not_unlock_above_its_ceiling():
    who = VerifiedIdentity(subject="u1", roles=["hr_verification_officer"], issuer="okta")
    assert apply_unlock(floor(severity=Severity.CRITICAL), who, policy)[0] == Action.BLOCK


def test_unrelated_role_does_not_unlock():
    who = VerifiedIdentity(subject="u1", roles=["designer"], issuer="okta")
    assert apply_unlock(floor(), who, policy)[0] == Action.BLOCK


def test_unlock_never_raises_an_allowed_floor():
    who = VerifiedIdentity(subject="u1", roles=["hr_verification_officer"], issuer="okta")
    assert apply_unlock(floor(action=Action.ALLOW, rule=None), who, policy)[0] == Action.ALLOW


def reencode_jpeg(png: bytes) -> bytes:
    buf = io.BytesIO()
    Image.open(io.BytesIO(png)).convert("RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def test_image_key_survives_reencoding():
    png = to_png(render(CATALOG["upi_payment"]))
    assert SessionMemory.same_image(image_key(png), image_key(reencode_jpeg(png)))


def test_image_key_separates_different_images():
    a = image_key(to_png(render(CATALOG["upi_payment"])))
    b = image_key(to_png(render_photo()))
    assert not SessionMemory.same_image(a, b)


CHANGE = EditSpec(field_changes=[FieldChange(kind="replace", field="Amount", new="5000")])


def test_halted_before_on_the_same_image():
    mem = SessionMemory()
    key = image_key(to_png(render(CATALOG["upi_payment"])))
    assert not mem.halted_before(key)
    mem.record(key, CHANGE, Action.BLOCK)
    assert mem.halted_before(key)


def test_retry_detection_ignores_other_images():
    mem = SessionMemory()
    mem.record(image_key(to_png(render(CATALOG["upi_payment"]))), CHANGE, Action.BLOCK)
    assert not mem.halted_before(image_key(to_png(render_photo())))


def test_cumulative_spec_composes_changes_across_turns():
    mem = SessionMemory()
    key = image_key(to_png(render(CATALOG["upi_payment"])))
    step1 = EditSpec(field_changes=[FieldChange(kind="redact", field="Amount")])
    mem.record(key, step1, Action.ALLOW)
    step2 = EditSpec(field_changes=[FieldChange(kind="add", new="5000")])
    total = mem.cumulative(key, step2)
    assert [c.kind for c in total.field_changes] == ["redact", "add"]
    assert total.factual_changes  # the composed edit is a factual change


def test_cumulative_ignores_blocked_steps():
    mem = SessionMemory()
    key = image_key(to_png(render(CATALOG["upi_payment"])))
    mem.record(key, CHANGE, Action.BLOCK)
    assert mem.cumulative(key, EditSpec()).field_changes == []


def test_a_review_also_counts_as_a_halt():
    mem = SessionMemory()
    key = image_key(to_png(render(CATALOG["boarding_pass"])))
    mem.record(key, CHANGE, Action.REVIEW)
    assert mem.halted_before(key)
