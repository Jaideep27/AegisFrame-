"""Stage 5 — check what was actually produced.

The generated image is compared with the ORIGINAL upload, not with the previous
turn, so an edit built up in small steps is still seen as one change. The diff
must stay inside what was approved, and any required marking must be present
where it was promised.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .ground import OcrEngine, _norm_value, ground, resolve_field
from .schema import Action, Decision, GroundedDoc, OcrLine

ROW_SLACK = 30  # px: a marking counts as "over" a value if it sits on the same row


class VerifyResult(BaseModel):
    action: Action
    reasons: list[str] = Field(default_factory=list)
    changed_fields: dict[str, tuple[str, str]] = Field(default_factory=dict)

    @property
    def message_for_user(self) -> str:
        return "OK" if self.action == Action.ALLOW else "The generated image did not match the approved edit."


def _strip_marking(value: str, marking: str | None) -> str:
    if not marking:
        return value
    return re.sub(re.escape(marking), "", value, flags=re.I).strip()


def _row_of(doc: GroundedDoc, label: str) -> OcrLine | None:
    for ln in doc.lines:
        if ln.text.strip() == label:
            return ln
    return None


def verify_docs(original: GroundedDoc, output: GroundedDoc, decision: Decision) -> VerifyResult:
    if decision.action != Action.ALLOW:
        return VerifyResult(action=Action.BLOCK, reasons=[f"request decision was {decision.action.name}"])

    edit = decision.edit
    mark_text = edit.marking.text if edit.marking else None
    reasons: list[str] = []
    action = Action.ALLOW

    if not original.is_document:
        if output.is_evidence and not edit.from_scratch:
            return VerifyResult(action=Action.REVIEW, reasons=["output looks like a record but the input was a photo"])
        return VerifyResult(action=Action.ALLOW)

    approved = {}
    for r in decision.floor.resolved if decision.floor else []:
        if r.field:
            approved[r.field] = _norm_value(r.change.new or "")

    # fields that an approved edit may make disappear (but never change)
    if any(v.attribute == "crop" for v in edit.visual_changes):
        may_vanish = set(original.fields)
    else:
        may_vanish = {
            label for c in edit.field_changes if c.kind == "redact"
            if (label := resolve_field(c, original.fields)) is not None
        }

    changed: dict[str, tuple[str, str]] = {}
    for label, before in original.fields.items():
        after = output.fields.get(label)
        after_clean = _strip_marking(after, mark_text) if after is not None else None
        if after_clean is None:
            if label not in approved and label not in may_vanish:
                changed[label] = (before, "")
                reasons.append(f"field {label!r} disappeared from the output")
                action = Action.BLOCK
            continue
        if _norm_value(after_clean) == _norm_value(before):
            continue
        changed[label] = (before, after_clean)
        if label not in approved:
            reasons.append(f"field {label!r} changed from {before!r} to {after_clean!r} but was not approved")
            action = Action.BLOCK
        elif approved[label] and _norm_value(after_clean) != approved[label]:
            reasons.append(f"field {label!r} became {after_clean!r}, not the approved value")
            action = Action.BLOCK

    for label in output.fields.keys() - original.fields.keys():
        if label.strip().upper() == (mark_text or "").upper():
            continue
        reasons.append(f"new field {label!r} appeared in the output")
        action = Action.BLOCK

    if edit.marking:
        marks = [ln for ln in output.lines if edit.marking.text.lower() in ln.text.lower()]
        if not marks:
            reasons.append(f"required marking {edit.marking.text!r} is missing")
            action = Action.BLOCK
        elif edit.marking.covers_changed_fields:
            for label in approved:
                anchor = _row_of(output, label) or _row_of(original, label)
                if anchor is None:
                    continue
                if not any(abs(m.box[1] - anchor.box[1]) <= ROW_SLACK for m in marks):
                    reasons.append(f"marking {edit.marking.text!r} is not over the changed field {label!r}")
                    action = Action.BLOCK

    return VerifyResult(action=action, reasons=reasons, changed_fields=changed)


def verify_output(original: bytes | None, output: bytes, decision: Decision, ocr: OcrEngine) -> VerifyResult:
    out_doc = ground(output, ocr)
    if original is None:
        # generated from scratch: there is no record to compare against, but a requested
        # marking must still be there
        if decision.action != Action.ALLOW:
            return VerifyResult(action=Action.BLOCK, reasons=[f"request decision was {decision.action.name}"])
        if decision.edit.marking and not any(decision.edit.marking.text.lower() in ln.text.lower() for ln in out_doc.lines):
            return VerifyResult(action=Action.BLOCK, reasons=[f"required marking {decision.edit.marking.text!r} is missing"])
        return VerifyResult(action=Action.ALLOW)
    return verify_docs(ground(original, ocr), out_doc, decision)
