"""The text the local model reads. Shared by the training-data generator and the SDK so that
training and inference see exactly the same format.

Two inputs per request (two passes of one encoder):
  full  = context + the user's whole message      -> separator tags, intent, twists, purpose
  plain = context + the canonical instruction     -> claim-blind heads (decision, severity, pattern, misuse...)
"""

from __future__ import annotations

from ..schema import GroundedDoc

MAX_FIELDS = 8
LABELS = {
    "tags": ["O", "B-FIELD", "I-FIELD", "B-OLD", "I-OLD", "B-NEW", "I-NEW", "B-MARK", "I-MARK", "B-OBJ", "I-OBJ",
             "B-STORY", "I-STORY"],
    "intent": ["forge_replace", "forge_remove", "forge_signature", "forge_marking_removal", "marked_sample",
               "fictional", "cosmetic", "redaction", "annotate", "question", "unclear", "photo_visual",
               "person_neutral", "person_demeaning", "person_false_act", "person_satire"],
    "decision": ["ALLOW", "REVIEW", "BLOCK"],
    "field_class": ["none", "money", "identity", "result", "score", "date", "signature", "marking", "other"],
    "pattern": ["none", "amount-inflation", "amount-deflation", "result-flip", "score-inflation", "validity-extension",
                "identity-swap", "signature-forgery", "date-shift", "marking-removal", "person-fabrication"],
    "twist": ["purpose", "dispute", "legal", "authority", "urgency", "emotional", "hypothetical", "roleplay",
              "correction_claim", "ownership_claim", "indirect", "obfuscated", "hinglish", "multi_turn"],
    "misuse": ["none", "deceive_lender", "deceive_employer", "deceive_merchant", "deceive_institution",
               "deceive_authority", "deceive_insurer", "deceive_public"],
    "benign_need": ["none", "personal_edit", "information", "sample", "privacy", "creative"],
}
FULL_HEADS = ("tags", "intent", "twist", "purpose")
PLAIN_HEADS = ("decision", "field_class", "pattern", "misuse", "benign_need")


def format_context(kind: str, header: list[str] | None = None, fields: dict[str, str] | None = None,
                   words: list[str] | None = None) -> str:
    """kind: record | photo | none."""
    if kind == "none":
        return "[NO IMAGE]"
    if kind == "photo":
        return "[PHOTO] " + " ".join(words or [])[:120]
    parts = [h for h in (header or []) if h][:2]
    parts += [f"{k}: {v}" for k, v in list((fields or {}).items())[:MAX_FIELDS]]
    return "[DOC] " + " | ".join(parts)


def context_for(doc: GroundedDoc) -> str:
    if not doc.has_image:
        return format_context("none")
    if not doc.is_document:
        # OCR on photos returns fragments (a face read as "O"); keep only real words
        words = [ln.text for ln in doc.lines if sum(ch.isalpha() for ch in ln.text) >= 3]
        return format_context("photo", words=words[:6])
    header = doc.doc_type.split(" - ")
    return format_context("record", header, doc.fields)
