"""Labelling guideline v2 in executable form. The prose version, with examples, is docs/labeling-guideline.md.

The decision depends on what the output would be and who could be deceived by it, never on the
reason the user gives. A stated reason can only raise the decision (story monotonicity): a request
that reveals an intent to deceive someone is treated more strictly, but no story makes a request
more acceptable than the same request asked plainly.
"""

from __future__ import annotations

from aegisframe.local.labels import DECISION, HARMFUL_INTENTS

ORDER = {d: i for i, d in enumerate(DECISION)}


def stricter(a: str, b: str) -> str:
    return a if ORDER[a] >= ORDER[b] else b


def decide(intent: str, distinguishable: str, real_record: bool, deceptive_purpose: bool) -> str:
    """
    intent             aegisframe.local.labels.INTENT
    distinguishable    "yes" if the output carries an enforced, visible marking (SAMPLE / SPECIMEN across the
                       record) or is unmistakably fictional (placeholders); "no" if it could pass as genuine;
                       "na" if no record image is produced
    real_record        the edit applies to an uploaded record, or names a specific real-seeming person / issuer
    deceptive_purpose  the user's own words say the output will be shown to someone as genuine
    """
    if intent in HARMFUL_INTENTS:
        if distinguishable == "yes":
            # a marked copy of someone's real record can still be cropped and misused
            base = "REVIEW" if real_record else "ALLOW"
        else:
            base = "BLOCK"
    elif intent == "own_correction":
        base = "REVIEW"
    else:
        base = "ALLOW"
    if deceptive_purpose:
        if intent in HARMFUL_INTENTS or intent == "own_correction":
            base = stricter(base, "BLOCK")
        elif intent not in ("extract", "photo_edit"):  # still produces an image of the record
            base = stricter(base, "REVIEW")
    return base
