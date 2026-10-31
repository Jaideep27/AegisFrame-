"""The fraud-pattern knowledge base and the value comparisons it needs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path

import yaml

from ..schema import ResolvedChange


@dataclass(frozen=True)
class Pattern:
    id: str
    field_class: tuple[str, ...]
    p: float
    direction: str | None = None
    flip_pairs: tuple[tuple[str, str], ...] = ()
    label_any: tuple[str, ...] = ()
    note: str = ""


@dataclass
class Match:
    pattern: Pattern
    p: float
    evidence: list[str] = field(default_factory=list)


def load_patterns(path: str | Path | None = None) -> list[Pattern]:
    if path is None:
        text = resources.files("aegisframe.council").joinpath("fraud_patterns.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    return [
        Pattern(
            id=p["id"], field_class=tuple(p["field_class"]), p=float(p["p"]), direction=p.get("direction"),
            flip_pairs=tuple(tuple(x.lower() for x in pair) for pair in p.get("flip_pairs", [])),
            label_any=tuple(w.lower() for w in p.get("label_any", [])), note=p.get("note", ""),
        )
        for p in raw.get("patterns", [])
    ]


_NUM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(k|lakh|lakhs|l|crore|cr|m|mn)?\b", re.I)
_MULT = {"k": 1e3, "lakh": 1e5, "lakhs": 1e5, "l": 1e5, "crore": 1e7, "cr": 1e7, "m": 1e6, "mn": 1e6}
_DATE_FORMATS = ("%d %b %Y", "%d %B %Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%b %Y", "%B %Y", "%d %b %Y, %I:%M %p")


def as_number(text: str | None) -> float | None:
    if not text:
        return None
    m = _NUM.search(text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    return value * _MULT.get((m.group(2) or "").lower(), 1)


def as_date(text: str | None) -> datetime | None:
    if not text:
        return None
    t = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    if re.fullmatch(r"(19|20)\d\d", t):
        return datetime(int(t), 12, 31)
    m = re.search(r"\b(\d{1,2} [A-Za-z]{3,9} \d{4})\b", t)
    return as_date(m.group(1)) if m and m.group(1) != t else None


def _direction(cls: str, old: str | None, new: str | None) -> str | None:
    if cls == "date":
        a, b = as_date(old), as_date(new)
    else:
        a, b = as_number(old), as_number(new)
    if a is None or b is None or a == b:
        return None
    return "up" if b > a else "down"


def match(patterns: list[Pattern], resolved: list[ResolvedChange]) -> Match | None:
    best: Match | None = None
    for r in resolved:
        label = (r.field or r.change.field or "").lower()
        old, new = r.change.old, r.change.new
        for pat in patterns:
            if r.field_class not in pat.field_class:
                continue
            if pat.label_any and not any(w in label for w in pat.label_any):
                continue
            p = pat.p
            if pat.direction:
                d = _direction(r.field_class, old, new)
                if d is None:
                    p *= 0.8  # direction unknown: weaker evidence
                elif d != pat.direction:
                    continue
            if pat.flip_pairs:
                o, n = (old or "").strip().lower(), (new or "").strip().lower()
                flipped = any({o, n} == set(pair) for pair in pat.flip_pairs)
                one_side = any(n in pair for pair in pat.flip_pairs)
                if not flipped and not (not o and one_side):
                    continue
                if not flipped:
                    p *= 0.85
            evidence = [f"{r.field or r.change.field}: {old!r} -> {new!r}"]
            if best is None or p > best.p:
                best = Match(pat, p, evidence)
    return best
