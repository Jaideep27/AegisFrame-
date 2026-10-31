"""A deterministic detector for record values a prompt proposes to change.

Model-independent evidence for the council. It reads the OCR-style record text and the prompt and
reports replacement pairs: a value already on the record (old) and a value in the prompt that would take
its place (new). It fires on an indirect edit - "English marks are 89" over a record showing 69, with no
edit verb - which is exactly where the fine-tuned model's intent head is weakest, and it stays quiet on a
question about a value or a restatement of one already printed.

The rule, deliberately conservative to keep false positives low:
  a new value (one not on the record) is a change only when it is anchored to the record, either by an
  old value from the record sitting beside it (a replacement pair) or by a record field label naming it.
A bare number with no such anchor ("I paid 500 at the print shop") is not reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: dd/mm/yyyy, yyyy-mm-dd and the like; checked before numbers so a date is not split into its parts.
_DATE = re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b")
#: 42,700 / 1,18,450 / 95000 / 42.50 - an integer or decimal, with optional grouping separators.
_NUMBER = re.compile(r"\b\d{1,3}(?:[,\s]\d{2,3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")
#: cue words that link a new value to an old one even across a few words.
_CUE = re.compile(r"->|→|=|\binstead of\b|\bnot\b|\bshould be\b|\brather than\b|\bfrom\b|\bto\b", re.I)
_LINK_WINDOW = 64  # characters between an old and a new value for them to count as a pair


@dataclass
class ValueDiff:
    """A value the prompt would replace. `old` is "" when only the field, not the old value, is named."""

    new: str
    old: str
    kind: str  # "amount" | "number" | "date"
    field: str | None


def _norm_num(s: str) -> str:
    return re.sub(r"[,\s]", "", s)


def _kind(surface: str) -> str:
    if _DATE.fullmatch(surface):
        return "date"
    return "amount" if ("," in surface or "." in surface) else "number"


def _record_fields(context: str) -> list[tuple[str, str]]:
    """(label, value) pairs from 'Label: value' and 'Label | value' lines."""
    out = []
    for line in context.splitlines():
        line = line.strip()
        for sep in (":", "|"):
            if sep in line:
                label, _, value = line.partition(sep)
                label, value = label.strip(), value.strip()
                if len(label) >= 3 and value:
                    out.append((label, value))
                break
    return out


def _extract(text: str) -> list[tuple[str, int, str]]:
    """(surface, start_offset, kind) for every date and number, dates taken first."""
    spans: list[tuple[str, int, str]] = []
    taken = [False] * len(text)
    for m in _DATE.finditer(text):
        spans.append((m.group(), m.start(), "date"))
        for i in range(m.start(), m.end()):
            taken[i] = True
    for m in _NUMBER.finditer(text):
        if not any(taken[m.start():m.end()]):
            spans.append((m.group().strip(), m.start(), _kind(m.group().strip())))
    return spans


def _numeric(kind: str) -> bool:
    return kind in ("amount", "number")


def detect_value_changes(context: str, prompt: str) -> list[ValueDiff]:
    fields = _record_fields(context)
    record_values = {_norm_num(v) for _, v in fields} | {_norm_num(s) for s, _, _ in _extract(context)}
    record_dates = {s for s, _, k in _extract(context) if k == "date"}
    record_values |= record_dates
    low_prompt = prompt.lower()

    prompt_vals = _extract(prompt)
    present = lambda s, k: (s in record_dates) if k == "date" else (_norm_num(s) in record_values)
    news = [(s, off, k) for s, off, k in prompt_vals if not present(s, k)]
    olds = [(s, off, k) for s, off, k in prompt_vals if present(s, k)]

    diffs: list[ValueDiff] = []
    for new, off, kind in news:
        # (a) an old value from the record sits near the new one, or a cue links them
        pair = None
        for old, ooff, okind in olds:
            same = okind == kind or (_numeric(okind) and _numeric(kind))
            if not same:
                continue
            lo, hi = (min(off, ooff), max(off, ooff))
            between = prompt[lo:hi]
            if hi - lo <= _LINK_WINDOW or _CUE.search(between):
                pair = old
                break
        if pair is not None:
            diffs.append(ValueDiff(new=new, old=pair, kind=kind, field=_field_for(fields, pair, kind)))
            continue
        # (b) no old value quoted, but a record field label names this value
        field = _field_near(fields, low_prompt, kind)
        if field is not None:
            diffs.append(ValueDiff(new=new, old="", kind=kind, field=field))
    return diffs


def _field_for(fields: list[tuple[str, str]], old: str, kind: str) -> str | None:
    for label, value in fields:
        if (value == old) or (_numeric(kind) and _norm_num(value) == _norm_num(old)):
            return label
    return None


def _field_near(fields: list[tuple[str, str]], low_prompt: str, kind: str) -> str | None:
    """A field whose label appears in the prompt and whose own recorded value is the same kind."""
    for label, value in fields:
        vk = "date" if _DATE.fullmatch(value) else _kind(value) if _NUMBER.fullmatch(value) else None
        if vk is None:
            # a value like "69/100" still reads as numeric for our purpose
            vk = "number" if re.search(r"\d", value) else None
        same = vk == kind or (vk and _numeric(vk) and _numeric(kind))
        if same and label.lower() in low_prompt:
            return label
    return None
