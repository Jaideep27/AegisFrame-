"""Stage 3 — the floor: would this edit be refused in plain form?

compute_floor() takes no claims argument. That is the point: the justification has
no path into the floor. Reference judges see only the canonical instruction rebuilt
from the EditSpec, and they can only raise the floor, never lower it.
"""

from __future__ import annotations

from typing import Protocol

from .ground import resolve_field
from .policy import Policy, attributes
from .providers.base import Provider, ProviderError
from .schema import (
    Action,
    EditSpec,
    FieldChange,
    FloorVerdict,
    GroundedDoc,
    JudgeVote,
    ResolvedChange,
    Severity,
    strictest,
)
from .semantics import classify_change
from .separate import canonical_instruction


class Judge(Protocol):
    name: str

    def vote(self, canonical: str, image: bytes | None) -> JudgeVote: ...


JUDGE_SYSTEM = """You are a reference reviewer for an image-editing service.
You are shown ONE plain editing instruction (and the image it applies to). No reason or
context is attached, on purpose: decide on the edit itself.

Should a responsible image-editing assistant REFUSE to carry out this exact edit, because the
result could be used to deceive someone (a forged record, fake evidence, a false depiction of a
real person) or cause other serious harm?

Return ONLY JSON: {"p_refuse": <probability 0..1>, "reason": "<one short sentence>"}"""


class LLMJudge:
    def __init__(self, provider: Provider):
        self.provider = provider
        self.name = f"judge:{provider.name}"

    def vote(self, canonical: str, image: bytes | None) -> JudgeVote:
        raw = self.provider.generate_json(JUDGE_SYSTEM, "INSTRUCTION:\n" + canonical, image)
        try:
            p = float(raw["p_refuse"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"{self.name}: reply has no p_refuse: {raw}") from exc
        return JudgeVote(judge=self.name, p_refuse=min(1.0, max(0.0, p)), reason=str(raw.get("reason", ""))[:300])


def resolve_changes(edit: EditSpec, doc: GroundedDoc) -> list[ResolvedChange]:
    resolved = []
    for c in edit.factual_changes:
        label = resolve_field(c, doc.fields)
        # quote what the record actually says, not the user's paraphrase of it
        old = doc.fields.get(label) if label else c.old
        cls, sev = classify_change(label or c.field, old, c.new)
        resolved.append(
            ResolvedChange(change=c.model_copy(update={"field": label or c.field, "old": old}), field=label,
                           field_class=cls, severity=sev)
        )
    return resolved


def compute_floor(
    edit: EditSpec,
    doc: GroundedDoc,
    policy: Policy,
    judges: list[Judge] | tuple[Judge, ...] = (),
    image: bytes | None = None,
) -> FloorVerdict:
    resolved = resolve_changes(edit, doc)
    severity = max((r.severity for r in resolved), default=Severity.NONE)
    if edit.real_person_false_act:
        severity = max(severity, Severity.HIGH)

    attrs = attributes(doc, edit)
    action, rule, reasons = policy.evaluate(attrs, severity, doc.doc_type)

    resolved_changes: list[FieldChange] = [r.change for r in resolved]
    redactions = [c for c in edit.field_changes if c.kind == "redact"]
    canonical_edit = edit.model_copy(update={"field_changes": resolved_changes + redactions})
    where = doc.doc_type if doc.is_document else "image"
    canonical = canonical_instruction(canonical_edit, where, real=attrs["real_source"])

    votes = []
    for judge in judges:
        vote = judge.vote(canonical, image)
        votes.append(vote)
        if vote.p_refuse >= policy.refuse_threshold:
            judged = Action.BLOCK
        elif vote.p_refuse >= policy.review_threshold:
            judged = Action.REVIEW
        else:
            judged = Action.ALLOW
        if judged > action:
            reasons.append(f"{vote.judge} p_refuse={vote.p_refuse:.2f} raised the floor to {judged.name}")
            action = strictest(action, judged)
            rule = rule or "reference-judge"

    return FloorVerdict(
        action=action, severity=severity, rule=rule if action > Action.ALLOW else None,
        canonical_instruction=canonical, resolved=resolved, votes=votes, reasons=reasons,
    )
