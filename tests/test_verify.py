import pytest

from aegisframe.ground import ground
from aegisframe.schema import (
    Action, Decision, EditSpec, FieldChange, FloorVerdict, GroundedDoc, Marking, OcrLine,
    ResolvedChange, Severity, VisualChange,
)
from aegisframe.verify import verify_docs


def row(label, value, y, extra=None):
    lines = [OcrLine(text=label, box=(60, y, 200, y + 26)), OcrLine(text=value, box=(430, y, 600, y + 26))]
    if extra:
        lines.append(OcrLine(text=extra, box=(720, y - 2, 850, y + 28)))
    return lines


def doc(*rows):
    lines = [OcrLine(text="PayNimbus", box=(60, 20, 200, 50)), OcrLine(text="Payment Successful", box=(60, 95, 300, 125))]
    for r in rows:
        lines += r
    from aegisframe.ground import pair_fields

    return GroundedDoc(has_image=True, is_document=True, is_evidence=True, real_source=True,
                       doc_type="PayNimbus - Payment Successful", lines=lines, fields=pair_fields(lines))


ORIGINAL = doc(row("Amount", "Rs. 500", 170), row("Date", "12 Oct 2026", 220), row("Status", "Completed", 270))

MARKED_EDIT = EditSpec(
    field_changes=[FieldChange(kind="replace", field="Amount", old="Rs. 500", new="Rs. 5,000")],
    marking=Marking(kind="stamp", text="SAMPLE", covers_changed_fields=True),
)


def approved(edit, resolved_fields=("Amount",)):
    resolved = [
        ResolvedChange(change=c.model_copy(update={"field": f}), field=f, field_class="money", severity=Severity.HIGH)
        for c, f in zip(edit.factual_changes, resolved_fields)
    ]
    floor = FloorVerdict(action=Action.ALLOW, severity=Severity.HIGH, canonical_instruction="x", resolved=resolved)
    return Decision(action=Action.ALLOW, edit=edit, floor=floor)


def test_output_matching_the_approved_marked_edit_passes():
    out = doc(row("Amount", "Rs. 5,000", 170, extra="SAMPLE"), row("Date", "12 Oct 2026", 220), row("Status", "Completed", 270))
    result = verify_docs(ORIGINAL, out, approved(MARKED_EDIT))
    assert result.action == Action.ALLOW, result.reasons


def test_missing_marking_blocks():
    out = doc(row("Amount", "Rs. 5,000", 170), row("Date", "12 Oct 2026", 220), row("Status", "Completed", 270))
    result = verify_docs(ORIGINAL, out, approved(MARKED_EDIT))
    assert result.action == Action.BLOCK
    assert any("SAMPLE" in r for r in result.reasons)


def test_marking_away_from_the_changed_value_blocks():
    out = doc(row("Amount", "Rs. 5,000", 170), row("Date", "12 Oct 2026", 220), row("Status", "Completed", 270, extra="SAMPLE"))
    assert verify_docs(ORIGINAL, out, approved(MARKED_EDIT)).action == Action.BLOCK


def test_unapproved_extra_change_blocks():
    out = doc(row("Amount", "Rs. 5,000", 170, extra="SAMPLE"), row("Date", "15 Oct 2026", 220), row("Status", "Completed", 270))
    result = verify_docs(ORIGINAL, out, approved(MARKED_EDIT))
    assert result.action == Action.BLOCK
    assert any("Date" in r for r in result.reasons)


def test_value_other_than_approved_blocks():
    out = doc(row("Amount", "Rs. 50,000", 170, extra="SAMPLE"), row("Date", "12 Oct 2026", 220), row("Status", "Completed", 270))
    assert verify_docs(ORIGINAL, out, approved(MARKED_EDIT)).action == Action.BLOCK


def test_a_non_allowed_decision_never_verifies():
    d = approved(MARKED_EDIT).model_copy(update={"action": Action.BLOCK})
    assert verify_docs(ORIGINAL, ORIGINAL, d).action == Action.BLOCK


def test_photo_edit_without_text_passes():
    photo = GroundedDoc(has_image=True, doc_type="photo")
    edit = EditSpec(visual_changes=[VisualChange(target="outfit", attribute="color", value="red")])
    assert verify_docs(photo, photo, approved(edit, ())).action == Action.ALLOW


def test_photo_turned_into_a_record_goes_to_review():
    photo = GroundedDoc(has_image=True, doc_type="photo")
    edit = EditSpec(visual_changes=[VisualChange(target="outfit", attribute="color", value="red")])
    assert verify_docs(photo, ORIGINAL, approved(edit, ())).action == Action.REVIEW


@pytest.mark.ocr
def test_verify_on_real_renders():
    from aegisframe.ground import RapidOcrEngine
    from reframebench.docs import CATALOG, render, to_png

    ocr = RapidOcrEngine()
    tpl = CATALOG["upi_payment"]
    orig = ground(to_png(render(tpl)), ocr)
    good = ground(to_png(render(tpl, overrides={"Amount": "Rs. 5,000"}, marking="SAMPLE", marking_over=["Amount"])), ocr)
    bad = ground(to_png(render(tpl, overrides={"Amount": "Rs. 5,000"})), ocr)
    assert verify_docs(orig, good, approved(MARKED_EDIT)).action == Action.ALLOW
    assert verify_docs(orig, bad, approved(MARKED_EDIT)).action == Action.BLOCK


def test_approved_crop_may_remove_fields_but_not_change_them():
    edit = EditSpec(visual_changes=[VisualChange(target="image", attribute="crop")])
    cropped = doc(row("Amount", "Rs. 500", 170))
    assert verify_docs(ORIGINAL, cropped, approved(edit, ())).action == Action.ALLOW
    cropped_and_forged = doc(row("Amount", "Rs. 5,000", 170))
    assert verify_docs(ORIGINAL, cropped_and_forged, approved(edit, ())).action == Action.BLOCK


def test_approved_redaction_may_hide_that_field_only():
    edit = EditSpec(field_changes=[FieldChange(kind="redact", field="date")])
    redacted = doc(row("Amount", "Rs. 500", 170), [OcrLine(text="Date", box=(60, 220, 200, 246))],
                   row("Status", "Completed", 270))
    assert verify_docs(ORIGINAL, redacted, approved(edit, ())).action == Action.ALLOW
    wrong_field = doc(row("Amount", "Rs. 500", 170), row("Date", "12 Oct 2026", 220),
                      [OcrLine(text="Status", box=(60, 270, 200, 296))])
    assert verify_docs(ORIGINAL, wrong_field, approved(edit, ())).action == Action.BLOCK
