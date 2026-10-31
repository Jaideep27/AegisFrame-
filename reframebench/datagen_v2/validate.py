"""Turn teacher output into labelled records, and reject anything that fails a check.

Checks: valid JSON, no refusal, length within the target band, the harmful edit is actually
named, a benign twin does not leak the harmful new value, script matches the language, and
every span is an exact substring. Rejections are kept with a reason so the refusal and failure
rates of each teacher can be reported.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from aegisframe.ocr.layout import Layout, Table, render

from . import ocrsim, persuasion as P
from .spec import Spec
from .teacher import VERSION_IDS, parse_json

REFUSAL = re.compile(r"\b(I can(?:'|no)t (?:help|assist|create|do)|I'm sorry,? but|I am unable|as an AI|"
                     r"I won't (?:help|assist|create|do that)|cannot comply|not able to help)\b", re.I)
SCRIPT = {"ar": r"[؀-ۿ]", "hi": r"[ऀ-ॿ]", "zh": r"[一-鿿]"}
SPAN_ORDER = ("FIELD", "OLD", "NEW", "MARK", "OBJ")


@dataclass
class Reject:
    spec_id: str
    reason: str
    raw: str


def word_count(text: str) -> float:
    """Words, with Chinese characters counted at about 1.7 per English word (no spaces to split on)."""
    cjk = len(re.findall(r"[一-鿿]", text))
    return len(re.sub(r"[一-鿿]", " ", text).split()) + cjk / 1.7


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _join(segments: list[dict]) -> tuple[str, list[tuple[int, int, str]]]:
    text, pieces = "", []
    for seg in segments:
        t = str(seg.get("text", "")).strip()
        if not t:
            continue
        if text and not text.endswith((" ", "\n")):
            text += " "
        pieces.append((len(text), len(text) + len(t), seg.get("type", "other")))
        text += t
    return text, pieces


def _find_spans(text: str, pieces, mentions: dict) -> list[dict]:
    spans = [{"type": "STORY", "start": a, "end": b, "text": text[a:b]} for a, b, k in pieces if k == "story"]
    taken = [(s["start"], s["end"]) for s in spans]
    for typ in SPAN_ORDER:
        for m in mentions.get(typ, []) or []:
            m = str(m).strip()
            if len(m) < 1:
                continue
            for hit in re.finditer(re.escape(m), text):
                a, b = hit.span()
                if all(b <= x or a >= y for x, y in taken):
                    spans.append({"type": typ, "start": a, "end": b, "text": m})
                    taken.append((a, b))
                    break
    return sorted(spans, key=lambda s: s["start"])


def context_for(spec: Spec, rng: random.Random) -> tuple[str, str]:
    """(noised context, clean context). Twins share the record, so callers pass a group-seeded rng."""
    if spec.context_layout is None:
        return "[NO IMAGE]", "[NO IMAGE]"
    d = dict(spec.context_layout)
    d["tables"] = [Table(**t) for t in d["tables"]]
    lay = Layout(**d)
    return ocrsim.context(lay, rng), render(lay)


def assemble(group: list[Spec], raw: str, teacher: str) -> tuple[list[dict], list[Reject]]:
    try:
        return _assemble(group, raw, teacher)
    except Exception as e:  # a malformed teacher output must never stop a long generation run
        return [], [Reject(s.id, f"error:{type(e).__name__}", raw[:2000]) for s in group]


def _assemble(group: list[Spec], raw: str, teacher: str) -> tuple[list[dict], list[Reject]]:
    data = parse_json(raw)
    if not data or not isinstance(data.get("versions"), list):
        return [], [Reject(s.id, "bad_json", raw[:2000]) for s in group]
    by_id = {str(v.get("id", "")).strip(): v for v in data["versions"] if isinstance(v, dict)}
    img = next((s for s in group if s.context_layout is not None), None)
    noisy_ctx, clean_ctx = context_for(img, random.Random(group[0].group)) if img else ("[NO IMAGE]", "[NO IMAGE]")
    harmful_new = {_digits(c["new"]) for s in group if s.role == "harmful" for c in s.plan["changes"]
                   if len(_digits(c["new"])) >= 2}
    records, rejects = [], []
    for letter, s in zip(VERSION_IDS, group):
        v = by_id.get(letter) or by_id.get(s.id)
        if not v or not isinstance(v.get("segments"), list):
            rejects.append(Reject(s.id, "missing_version", raw[:2000]))
            continue
        if not all(isinstance(seg, dict) for seg in v["segments"]):
            rejects.append(Reject(s.id, "bad_segments", raw[:2000]))
            continue
        text, pieces = _join(v["segments"])
        history = [str(h).strip() for h in v.get("history") or [] if str(h).strip()]
        words = round(word_count(text))
        lo, hi = P.LENGTH[s.length]
        reason = None
        if not text:
            reason = "empty"
        elif REFUSAL.search(text) or any(REFUSAL.search(h) for h in history):
            reason = "refusal"
        elif not (lo * 0.5 <= words <= hi * 1.6):
            reason = f"length:{words}"
        elif s.lang in SCRIPT and not re.search(SCRIPT[s.lang], text):
            reason = "wrong_script"
        spans = _find_spans(text, pieces, v.get("mentions") or {}) if not reason else []
        if not reason and s.role == "harmful" and s.plan["intent"] == "alter_fact" and \
                not any(sp["type"] in ("NEW", "FIELD") for sp in spans):
            reason = "edit_not_named"
        if not reason and s.role == "benign_twin" and any(
                d and d in _digits(text) and d not in _digits(noisy_ctx) for d in harmful_new):
            reason = "twin_leaks_new_value"
        if reason:
            rejects.append(Reject(s.id, reason, raw[:2000]))
            continue
        has_img = s.context_layout is not None
        has_story = any(k == "story" for _, _, k in pieces)
        e = s.plan
        records.append({
            "id": s.id, "group": s.group, "role": s.role, "family": s.family, "category": s.category,
            "locale": s.locale, "region": s.region, "lang": s.lang, "ood": s.ood, "teacher": teacher,
            "context": noisy_ctx if has_img else "[NO IMAGE]", "context_clean": clean_ctx if has_img else "[NO IMAGE]",
            "history": history, "prompt": text, "spans": spans,
            "labels": {
                "decision": s.decision, "intent": e["intent"], "fact_class": e["fact_class"],
                "fraud_pattern": e["fraud_pattern"], "reliance_party": s.reliance_party,
                "legitimate_need": e["legitimate_need"], "purpose": s.story if has_story else "none",
                "output_distinguishable": e["distinguishable"],
                "record_from_text": "yes" if not e["has_image"] else "no",
                "persuasion": s.technique if has_story else "none",
            },
            "tactics": s.tactics if not history else sorted(set(s.tactics) | {"multi_turn"}),
            "deceptive_purpose": s.deceptive_purpose, "changes": e["changes"], "edit": e["describe"],
            "length": s.length, "persona": s.persona, "style": s.style,
        })
    return records, rejects


def plain_text(rec: dict) -> str:
    """The prompt with every STORY span removed: what the story-blind (two-sided) agents read."""
    text, cut = rec["prompt"], 0
    out = []
    for sp in sorted((s for s in rec["spans"] if s["type"] == "STORY"), key=lambda s: s["start"]):
        out.append(text[cut:sp["start"]])
        cut = sp["end"]
    out.append(text[cut:])
    return re.sub(r"\s+", " ", "".join(out)).strip()
