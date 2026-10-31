"""The separator, done by the local model: tags + intent -> the same closed EditSpec + claims."""

from __future__ import annotations

from ..schema import Claim, EditSpec, FieldChange, Marking, Separation, VisualChange
from .runtime import FullAnalysis, LocalModel

TWIST_TO_CLAIM = {"purpose": "purpose", "dispute": "purpose", "legal": "authority", "authority": "authority",
                  "urgency": "urgency", "emotional": "emotional", "hypothetical": "context", "roleplay": "context",
                  "correction_claim": "context", "ownership_claim": "role"}


def _first(spans, tag, prompt):
    for a, b, t, _ in spans:
        if t == tag:
            return prompt[a:b].strip(" '\".,")
    return None


def edit_from_analysis(prompt: str, full: FullAnalysis) -> tuple[EditSpec, list[Claim]]:
    intent, _ = full.top_intent
    sp = full.spans
    field, old, new = _first(sp, "FIELD", prompt), _first(sp, "OLD", prompt), _first(sp, "NEW", prompt)
    mark, obj = _first(sp, "MARK", prompt), _first(sp, "OBJ", prompt)

    stories = [prompt[a:b].strip() for a, b, t, _ in sp if t == "STORY"]
    story_kinds = sorted((k for k, p in full.twist.items() if p >= 0.5 and k in TWIST_TO_CLAIM), key=lambda k: -full.twist[k])
    kind = TWIST_TO_CLAIM[story_kinds[0]] if story_kinds else "context"
    claims = [Claim(text=s, kind=kind) for s in stories if s]

    without_story = prompt
    for s in stories:
        without_story = without_story.replace(s, " ")
    action = " ".join(without_story.split())[:80]

    if intent == "forge_replace":
        return EditSpec(field_changes=[FieldChange(kind="replace", field=field, old=old, new=new)]), claims
    if intent == "forge_remove":
        return EditSpec(field_changes=[FieldChange(kind="remove", field=field)]), claims
    if intent == "forge_signature":
        return EditSpec(field_changes=[FieldChange(kind="add", field=field or "Signature", new=new)]), claims
    if intent == "forge_marking_removal":
        return EditSpec(field_changes=[FieldChange(kind="remove", field=f"{mark or 'validity'} watermark")]), claims
    if intent == "marked_sample":
        change = FieldChange(kind="replace", field=field, old=old, new=new)
        return EditSpec(field_changes=[change], marking=Marking(kind="stamp", text=(mark or "SAMPLE").upper(),
                                                               covers_changed_fields=True)), claims
    if intent == "fictional":
        return EditSpec(from_scratch=True), claims
    if intent == "cosmetic":
        return EditSpec(visual_changes=[VisualChange(target="image", attribute="style")]), claims
    if intent == "redaction":
        return EditSpec(field_changes=[FieldChange(kind="redact", field=field)]), claims
    if intent == "annotate":
        return EditSpec(visual_changes=[VisualChange(target=field or "image", attribute="object_add", value="annotation")]), claims
    if intent == "question":
        return EditSpec(answer_only=True), claims
    if intent == "photo_visual":
        return EditSpec(visual_changes=[VisualChange(target=obj or "image", attribute="other", value=action)]), claims
    if intent.startswith("person_"):
        return EditSpec(visual_changes=[VisualChange(target=obj or "the person", attribute="other", value=action)],
                        real_person_false_act=intent in ("person_demeaning", "person_false_act")), claims
    return EditSpec(), claims  # unclear: the policy's unparsed-record rule decides


class LocalSeparator:
    """Separator backed by the local model. Use it inside AgreementSeparator with the rules
    so that any disagreement resolves to the stricter reading."""

    def __init__(self, model: LocalModel):
        self.model = model
        self.name = f"local:{model.version}"
        self.last: FullAnalysis | None = None
        self.confidence: float | None = None

    def separate(self, prompt: str, history: list[str] | None = None, context: str | None = None) -> Separation:
        text = " [TURN] ".join([*(history or [])[-2:], prompt]) if history else prompt
        full = self.model.analyze_full(context or "[NO IMAGE]", text)
        self.last = full
        self.confidence = full.top_intent[1]
        offset = len(text) - len(prompt)
        if offset:
            full = FullAnalysis(spans=[(a - offset, b - offset, t, c) for a, b, t, c in full.spans if a >= offset],
                                intent=full.intent, twist=full.twist, purpose=full.purpose)
        edit, claims = edit_from_analysis(prompt, full)
        return Separation(edit=edit, claims=claims)


class ConfidentSeparator:
    """Trust the local model's reading when it is confident; otherwise merge it with the rules
    (AgreementSeparator: any disagreement resolves to the stricter reading).

    Merging always would let the rules' misreadings of unfamiliar phrasings add phantom factual
    changes, and drop real SAMPLE markings the rules do not recognise."""

    def __init__(self, primary, fallback, threshold: float = 0.85):
        self.primary, self.fallback, self.threshold = primary, fallback, threshold
        self.name = f"confident({getattr(primary, 'name', '?')}|{getattr(fallback, 'name', '?')})"

    def separate(self, prompt: str, history: list[str] | None = None, context: str | None = None) -> Separation:
        from ..separate import AgreementSeparator, _call

        first = _call(self.primary, prompt, history, context)
        if (getattr(self.primary, "confidence", None) or 0.0) >= self.threshold:
            # a confident model can still be confidently wrong on wording it never saw: any forged
            # fact the rules find (a change carrying a new value) is added, never dropped
            ruled = _call(self.fallback, prompt, history, context)
            extra = [c for c in ruled.edit.field_changes if c.kind in ("replace", "add") and c.new]
            seen = {(c.new or "").lower() for c in first.edit.field_changes}
            extra = [c for c in extra if (c.new or "").lower() not in seen]
            if extra:
                edit = first.edit.model_copy(update={"field_changes": [*first.edit.field_changes, *extra]})
                return first.model_copy(update={"edit": edit, "agreed": False})
            return first

        class _Done:  # reuse the primary's result inside the agreement merge
            name = getattr(self.primary, "name", "primary")

            def separate(self, p, h=None, context=None):
                return first

        return AgreementSeparator(_Done(), self.fallback).separate(prompt, history, context=context)
