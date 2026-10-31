"""Topic-free field semantics: what KIND of fact a change touches, and how severe that is.

Nothing here knows about payments, hospitals or airlines. It knows about money,
identities, results, dates and signatures, which every kind of record shares.
"""

from __future__ import annotations

import re

from .schema import Severity

SIGNATURE = ("signature", "signed by", "seal", "stamp", "sign")
MARKING = ("watermark", "void", "specimen", "sample", "duplicate", "draft", "not valid", "cancelled", "copy")
IDENTITY = (
    "name", "passenger", "candidate", "holder", "customer", "patient", "tenant", "student",
    "employee", "id", "number", "no", "roll", "register", "pnr", "passport", "pan", "aadhaar",
    "dob", "date of birth", "account", "serial", "policy", "paid to", "upi",
)
RESULT = (
    "result", "status", "grade", "outcome", "verdict", "diagnosis", "approved", "pass",
    "fail", "positive", "negative", "rejected", "eligibility",
)
SCORE = ("marks", "percentage", "score", "cgpa", "gpa", "rank")
MONEY = (
    "amount", "pay", "paid", "salary", "ctc", "balance", "total", "credit", "credits", "debit",
    "debits", "price", "fee", "compensation", "income", "rent", "cost", "bonus", "stipend",
    "package", "remuneration", "wage", "wages", "earnings",
)
DATE = ("date", "valid until", "expiry", "expires", "time", "period", "deadline", "issued on")

SEVERITY_BY_CLASS = {
    "signature": Severity.CRITICAL,
    "marking": Severity.HIGH,
    "identity": Severity.HIGH,
    "result": Severity.HIGH,
    "score": Severity.HIGH,
    "money": Severity.HIGH,
    "date": Severity.MODERATE,
    "other": Severity.MODERATE,
}

_MONEY_VALUE = re.compile(r"(rs\.?|inr|₹|\$|usd|eur|€|£)\s*[\d,.]+|^[\d,]{3,}(\.\d+)?$", re.I)
_DATE_VALUE = re.compile(
    r"\b\d{1,2}[ /.-](jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2})[a-z]*[ /.-]\d{2,4}\b",
    re.I,
)
_RESULT_VALUE = re.compile(r"^(positive|negative|pass|fail|approved|rejected|reactive|non-reactive)$", re.I)


def _has(text: str, keywords: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


def classify_label(label: str) -> str | None:
    t = label.lower()
    for cls, kws in (
        ("signature", SIGNATURE),
        ("marking", MARKING),
        ("identity", IDENTITY),
        ("result", RESULT),
        ("score", SCORE),
        ("money", MONEY),
        ("date", DATE),
    ):
        if _has(t, kws):
            return cls
    return None


def classify_value(value: str) -> str | None:
    v = value.strip()
    if _DATE_VALUE.search(v):
        return "date"
    if _MONEY_VALUE.search(v):
        return "money"
    if _RESULT_VALUE.match(v):
        return "result"
    return None


def classify_change(label: str | None, old: str | None, new: str | None) -> tuple[str, Severity]:
    cls = classify_label(label) if label else None
    if cls is None:
        for value in (old, new):
            if value and (cls := classify_value(value)):
                break
    cls = cls or "other"
    return cls, SEVERITY_BY_CLASS[cls]
