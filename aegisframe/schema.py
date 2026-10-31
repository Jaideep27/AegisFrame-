"""Core data types.

The separation between WHAT (EditSpec) and WHY (Claim) is the heart of the design:
EditSpec is a closed schema with no field where a justification could live, so the
decision path that consumes it cannot be persuaded by a story.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Action(IntEnum):
    """Decision outcomes, ordered by strictness so max() picks the strictest."""

    ALLOW = 0
    REVIEW = 1
    BLOCK = 2


class Severity(IntEnum):
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


def strictest(*actions: Action) -> Action:
    return max(actions, default=Action.ALLOW)


# --- WHAT: the edit itself -------------------------------------------------------

FieldChangeKind = Literal["replace", "add", "remove", "redact"]


class FieldChange(BaseModel):
    """A change to a factual detail shown on the image (a number, name, date, result...).

    `redact` is a visible black-out box: the record visibly shows something was hidden,
    so it does not pass as an unaltered record.
    """

    kind: FieldChangeKind
    field: str | None = None  # label as the user or the document names it
    old: str | None = None
    new: str | None = None


VisualAttribute = Literal[
    "color", "style", "background", "lighting", "crop", "object_add", "object_remove", "other"
]


class VisualChange(BaseModel):
    """A non-factual change (an outfit colour, a background, lighting)."""

    target: str
    attribute: VisualAttribute
    value: str | None = None


class Marking(BaseModel):
    """A visible marking the output must carry. Enforced and checked after generation."""

    kind: Literal["watermark", "label", "stamp"]
    text: str
    covers_changed_fields: bool = False


class EditSpec(BaseModel):
    field_changes: list[FieldChange] = Field(default_factory=list)
    visual_changes: list[VisualChange] = Field(default_factory=list)
    marking: Marking | None = None
    from_scratch: bool = False
    real_person_false_act: bool = False
    answer_only: bool = False  # a question about the image: no image needs to be generated

    @property
    def factual_changes(self) -> list[FieldChange]:
        return [c for c in self.field_changes if c.kind != "redact"]

    @property
    def passes_as_genuine(self) -> bool:
        """Decided by the output spec, never by stated intent.

        A factually altered record passes as genuine unless an enforced marking covers
        the changed values.
        """
        if not self.factual_changes and not self.real_person_false_act:
            return False
        return not (self.marking and self.marking.covers_changed_fields)


# --- WHY: everything else the user said ------------------------------------------

# "context" = narrative without a recognised keyword (still a story);
# "other" = leftover text that is not a justification (e.g. an edit the rules could not parse)
ClaimKind = Literal["purpose", "role", "authority", "urgency", "emotional", "context", "other"]


class Claim(BaseModel):
    text: str
    kind: ClaimKind = "other"


class Separation(BaseModel):
    edit: EditSpec
    claims: list[Claim] = Field(default_factory=list)
    agreed: bool = True  # False when independent separators disagreed (strict path taken)


# --- grounding, floor, decision ----------------------------------------------------


class OcrLine(BaseModel):
    text: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1
    score: float = 1.0
    angle: float = 0.0  # tilt of the text baseline in degrees (photos are rarely straight)


class GroundedDoc(BaseModel):
    """Attributes computed from the IMAGE (and the EditSpec), never from claims."""

    has_image: bool
    is_document: bool = False
    is_evidence: bool = False
    real_source: bool = False
    doc_type: str = "unknown"
    lines: list[OcrLine] = Field(default_factory=list)
    fields: dict[str, str] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


class ResolvedChange(BaseModel):
    change: FieldChange
    field: str | None  # label on the document, when it could be matched
    field_class: str
    severity: Severity


class JudgeVote(BaseModel):
    judge: str
    p_refuse: float
    reason: str = ""


class FloorVerdict(BaseModel):
    """'Would this be refused in plain form?' — computed with every justification removed."""

    action: Action
    severity: Severity
    rule: str | None = None
    canonical_instruction: str
    resolved: list[ResolvedChange] = Field(default_factory=list)
    votes: list[JudgeVote] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    action: Action
    rule: str | None = None
    severity: Severity = Severity.NONE
    reasons: list[str] = Field(default_factory=list)
    canonical_instruction: str = ""
    generator_prompt: str = ""  # what the image model receives on ALLOW (no story on the record path)
    edit: EditSpec = Field(default_factory=EditSpec)
    claims: list[Claim] = Field(default_factory=list)
    floor: FloorVerdict | None = None
    council: Any = None  # council.CouncilVerdict (typed loosely to avoid an import cycle)
    signals: dict[str, bool] = Field(default_factory=dict)
    image_key: str | None = None
    request_id: str = ""
    review_id: str | None = None  # set when a REVIEW decision was queued for a human

    @property
    def message_for_user(self) -> str:
        if self.action == Action.ALLOW:
            return "OK"
        if self.action == Action.REVIEW:
            return "This edit needs a human review before it can be generated."
        return (
            "This edit would change a factual detail on a real record so that it passes as "
            "genuine. It can't be generated, whatever the reason given. A version with a "
            "visible SAMPLE marking over the changed values, or a fictional document made from "
            "scratch, can be."
        )
