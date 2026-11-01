"""HarrierGuard - the v0.7 prompt-level engine the package exposes.

It judges a request to edit a document at the prompt level: the fine-tuned model reads the request and the
OCR-style record text, the council's agents turn its heads plus a deterministic value-diff into LLR
messages, and the pool fuses them into ALLOW / REVIEW / BLOCK with a full trace. The v0.6 Guard stays as
the image-pipeline engine; this is the one the v0.7 report measured and the one the MCP server calls.

    guard = HarrierGuard.load("models/aegis-v07-r1-onnx")
    v = guard.check("change the net pay to 90,000", context="Record:\nPayslip\nNet Pay: 42,700\n")
    v.decision        # "BLOCK"
    v.trace           # one row per agent: name, message, whether it fired, a note

The story-stripped "plain" text is built from the request by removing a leading justification clause, so
the decision is story-blind even without the v0.6 separator. When fusion artifacts (case bank,
calibration) are present beside the model they are loaded and the fault tolerance rises to f=1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .council.blackboard import Blackboard, Verdict
from .council.casebank import CaseBank
from .errors import GuardUnavailable
from .local.harrier import HarrierModel

#: lead-in clauses people put before the real ask; stripped to approximate the story-blind instruction
_STORY_LEAD = re.compile(
    r"^(?:(?:hi|hey|hello|bro|bhai|sir|madam|please|pls|so|ok|okay|look|listen|"
    r"i'?m\s+\w+|i am\s+\w+|for my\b|it'?s for\b|just\b|quick\b|urgent\w*)\b[^.?!]*[.?!,:]\s*)+",
    re.I)


@dataclass
class HarrierVerdict:
    decision: str
    p: dict[str, float]
    trace: list[dict]
    value_changes: list
    p_block: float
    intent: str
    reason: str

    @classmethod
    def from_blackboard(cls, v: Verdict) -> HarrierVerdict:
        fired = [r["note"] for r in v.trace if r["triggered"] and r["note"]]
        return cls(decision=v.decision, p=v.p, trace=v.trace, value_changes=v.value_changes,
                   p_block=v.analysis.p_block, intent=v.analysis.top_intent[0],
                   reason="; ".join(fired[:4]) or "no risk signals fired")


def canonical_instruction(prompt: str) -> str:
    """A rough story-blind rendering: drop a leading justification clause, keep the operative ask."""
    stripped = _STORY_LEAD.sub("", prompt, count=1).strip()
    return stripped or prompt


class HarrierGuard:
    def __init__(self, model, casebank: CaseBank | None = None, fusion: dict | None = None,
                 f: int | None = None, ocr=None):
        self.model = model
        self.casebank = casebank
        self.ocr = ocr
        self.fusion = fusion or {}
        prior = self.fusion.get("prior")
        weights = self.fusion.get("weights") or {}
        if f is None:
            f = self.fusion.get("f_default", 1 if casebank is not None else 0)
        self.blackboard = Blackboard(model, casebank=casebank, prior=prior, weights=weights, f=f)

    @classmethod
    def load(cls, path: str | Path | None = None, f: int | None = None, **model_kw) -> HarrierGuard:
        model = HarrierModel.load(path, **model_kw) if path is None else HarrierModel(path, **model_kw)
        base = Path(model.path)
        fusion_dir = base / "fusion"
        casebank = fusion = None
        if (fusion_dir / "casebank.npz").exists():
            casebank = CaseBank.load(fusion_dir / "casebank.npz")
        if (fusion_dir / "fusion.json").exists():
            fusion = json.loads((fusion_dir / "fusion.json").read_text(encoding="utf-8"))
        return cls(model, casebank=casebank, fusion=fusion, f=f)

    def check(self, prompt: str, context: str = "", history: list[str] | None = None) -> HarrierVerdict:
        if not context.strip():
            context = "Record: [no image attached]\n"
        plain = canonical_instruction(prompt)
        v = self.blackboard.assess(context, prompt, plain=plain, history=history)
        return HarrierVerdict.from_blackboard(v)

    def check_image(self, image: bytes, prompt: str, history: list[str] | None = None,
                    ocr=None) -> HarrierVerdict:
        """OCR the image into the table-aware context the model was trained on, then check.

        The record is read as a layout (header, key-values, table rows) and rendered in the same
        `[DOC] ... [TABLE] ...` form the training data used, so a forged table cell is seen in context.
        """
        from .ocr.layout import from_lines, render

        engine = ocr or self.ocr
        if engine is None:
            from .ground import RapidOcrEngine

            engine = self.ocr = RapidOcrEngine()
        context = render(from_lines(engine.read(image)))
        return self.check(prompt, context=context, history=history)
