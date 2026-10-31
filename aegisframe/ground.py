"""Stage 2 — ground the request in the IMAGE.

Every attribute here comes from the uploaded pixels (and, later, the EditSpec).
Nothing the user says about why they want the edit is read here.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import re
from typing import Protocol

from .schema import FieldChange, GroundedDoc, OcrLine
from .semantics import classify_change, classify_label, classify_value

ROW_TOLERANCE = 14  # px between line centres to count as the same row
_ID_LIKE = re.compile(r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z0-9]{6})[A-Z0-9-]{8,}\b")


class OcrEngine(Protocol):
    def read(self, image: bytes) -> list[OcrLine]: ...


class RapidOcrEngine:
    """Local CPU OCR (PP-OCR via ONNX). Loaded lazily; the models ship with the package."""

    def __init__(self, cache_size: int = 256) -> None:
        self._engine = None
        self._cache: dict[str, list[OcrLine]] = {}
        self._cache_size = cache_size

    def read(self, image: bytes) -> list[OcrLine]:
        digest = hashlib.sha256(image).hexdigest()
        if digest not in self._cache:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[digest] = self._read(image)
        return self._cache[digest]

    def _read(self, image: bytes) -> list[OcrLine]:
        if self._engine is None:
            from rapidocr import RapidOCR

            logging.getLogger("RapidOCR").setLevel(logging.WARNING)
            # RapidOCR logs model paths at INFO on start-up unless told otherwise
            self._engine = RapidOCR(params={"Global.log_level": "warning"})
        import numpy as np
        from PIL import Image

        arr = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
        result = self._engine(arr)
        if result.txts is None:
            return []
        lines = []
        for text, box, score in zip(result.txts, result.boxes, result.scores):
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            # box points run clockwise from top-left: the top edge gives the baseline tilt
            angle = math.degrees(math.atan2(ys[1] - ys[0], xs[1] - xs[0])) if xs[1] != xs[0] else 0.0
            lines.append(OcrLine(text=text, box=(min(xs), min(ys), max(xs), max(ys)), score=float(score),
                                 angle=angle))
        return lines


def _centre_y(line: OcrLine) -> float:
    return (line.box[1] + line.box[3]) / 2


def pair_fields(lines: list[OcrLine]) -> dict[str, str]:
    """Key-value pairs: the leftmost text on a row is the label, the rest is the value."""
    rows: list[list[OcrLine]] = []
    for ln in sorted(lines, key=_centre_y):
        if rows and abs(_centre_y(rows[-1][0]) - _centre_y(ln)) <= ROW_TOLERANCE:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    fields: dict[str, str] = {}
    for row in rows:
        if len(row) < 2:
            continue
        row.sort(key=lambda ln: ln.box[0])
        fields[row[0].text.strip()] = " ".join(ln.text.strip() for ln in row[1:])
    return fields


def ground(image: bytes | None, ocr: OcrEngine) -> GroundedDoc:
    if image is None:
        return GroundedDoc(has_image=False)
    lines = ocr.read(image)
    fields = pair_fields(lines)
    # many real records (payment apps, tickets) show facts without a label beside them
    fact_lines = sum(1 for ln in lines if classify_value(ln.text) or _ID_LIKE.search(ln.text))
    is_document = len(lines) >= 4 and (len(fields) >= 2 or fact_lines >= 2)
    holds_facts = fact_lines >= 1 or any(
        classify_change(label, value, None)[0] != "other" for label, value in fields.items()
    )
    header = [ln.text for ln in sorted(lines, key=_centre_y)[:2]]
    return GroundedDoc(
        has_image=True,
        real_source=True,  # an uploaded capture is treated as a real record by default
        is_document=is_document,
        is_evidence=is_document and holds_facts,
        doc_type=" - ".join(header) if is_document else "photo",
        lines=lines,
        fields=fields,
    )


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _norm_value(text: str) -> str:
    v = _norm(text)
    v = re.sub(r"^(rs|inr|usd)", "", v)
    digits = re.sub(r"\D", "", v)
    return digits if digits and len(digits) >= len(v) - 1 else v


def resolve_field(change: FieldChange, fields: dict[str, str]) -> str | None:
    """Which field on the document does this change touch?"""
    if change.old:
        target = _norm_value(change.old)
        for label, value in fields.items():
            if target and _norm_value(value) == target:
                return label
    if change.field:
        want = _norm(change.field)
        for label in fields:
            if _norm(label) == want:
                return label
        for label in fields:
            if want and (want in _norm(label) or _norm(label) in want):
                return label
        want_cls = classify_label(change.field)
        if want_cls:
            for label in fields:
                if classify_label(label) == want_cls:
                    return label
    return None
