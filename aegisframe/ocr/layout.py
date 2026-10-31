"""The document context the local model reads: one text format shared by training data and runtime.

A document is a header (title, issuer), key-value fields and optional tables. OCR output (runtime)
and the data engine's simulated OCR (training) both become a Layout and are rendered with
render(), so the model never sees a format at runtime that it did not see in training.

    [DOC] Statement of Grades - Semester 6
    Deccan Technical University
    Student Name: Karthik Sundaram
    GPA: 9.7
    [TABLE] Subject | Grade | Points
    Mathematics | O | 10
    [/TABLE]
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

NO_IMAGE = "[NO IMAGE]"
MAX_FIELDS = 24
MAX_ROWS = 20


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]


@dataclass
class Layout:
    kind: str = "record"  # record | photo | none
    title: str = ""
    issuer: str = ""
    fields: list[tuple[str, str]] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    text: list[str] = field(default_factory=list)  # loose lines that are neither fields nor table cells

    def get(self, label: str) -> str | None:
        return next((v for k, v in self.fields if k == label), None)


def _cell(s: str) -> str:
    return " ".join(str(s).replace("|", "/").split())


def render(layout: Layout | None) -> str:
    if layout is None or layout.kind == "none":
        return NO_IMAGE
    if layout.kind == "photo":
        return ("[PHOTO] " + " ".join(layout.text)).strip()
    out = ["[DOC] " + _cell(layout.title or "document")]
    if layout.issuer:
        out.append(_cell(layout.issuer))
    out += [f"{_cell(k)}: {_cell(v)}" for k, v in layout.fields[:MAX_FIELDS]]
    for t in layout.tables:
        out.append("[TABLE] " + " | ".join(_cell(h) for h in t.header))
        out += [" | ".join(_cell(c) for c in row) for row in t.rows[:MAX_ROWS]]
        out.append("[/TABLE]")
    out += [_cell(x) for x in layout.text if x.strip()]
    return "\n".join(out)


def from_lines(lines) -> Layout:
    """Rebuild a Layout from OCR lines (.text, .box = (x0, y0, x1, y1), optional .angle) by geometry alone.

    1. Undo the page tilt (median baseline angle) and group lines into rows by vertical centre.
    2. A row of three or more text-only cells followed by rows that share its columns is a table:
       the header fixes the column positions and every later box is assigned to the nearest column.
    3. A row of two cells, or one cell reading "label: value", is a field. Other rows are loose text;
       the tallest of the first rows is the title and the line next to it the issuer.
    """
    lines = [ln for ln in lines if str(ln.text).strip()]
    if not lines:
        return Layout(kind="photo")
    heights = sorted(ln.box[3] - ln.box[1] for ln in lines)
    angles = sorted(getattr(ln, "angle", 0.0) for ln in lines)
    tilt = math.radians(angles[len(angles) // 2]) if abs(angles[len(angles) // 2]) < 15 else 0.0
    # a tilted box is taller than its text; the shortest boxes are the closest to the real line height
    h = max(heights[len(heights) // 4], 1.0)
    cos, sin = math.cos(tilt), math.sin(tilt)
    items = []
    for ln in lines:
        cx, cy = (ln.box[0] + ln.box[2]) / 2, (ln.box[1] + ln.box[3]) / 2
        w = ln.box[2] - ln.box[0]
        x, y = cx * cos + cy * sin, -cx * sin + cy * cos  # rotate the centre back to a level page
        items.append((y, x - w / 2, x + w / 2, ln.text.strip(), ln.box[3] - ln.box[1]))
    rows: list[list] = []
    for it in sorted(items):
        if rows and abs(it[0] - rows[-1][0][0]) <= 0.55 * h:
            rows[-1].append(it)
        else:
            rows.append([it])
    for row in rows:
        row.sort(key=lambda it: it[1])

    def cells_of(row) -> list[list]:
        cells: list[list] = []
        for _, x0, x1, text, _ in row:
            if cells and x0 - cells[-1][2] < 0.3 * h:
                cells[-1][0] += " " + text  # one phrase split into two boxes
                cells[-1][2] = x1
            else:
                cells.append([text, x0, x1])
        return cells

    lay = Layout()
    loose: list[tuple[int, str, float]] = []
    i = 0
    while i < len(rows):
        cells = cells_of(rows[i])
        if len(cells) >= 3 and not any(ch.isdigit() for ch in "".join(c[0] for c in cells)):
            anchors = [c[1] for c in cells]
            body = []
            j = i + 1
            while j < len(rows):
                cols = [""] * len(anchors)
                for _, x0, x1, text, _ in rows[j]:
                    k = max((n for n, a in enumerate(anchors) if a <= x0 + 0.5 * h), default=0)
                    cols[k] = (cols[k] + " " + text).strip()
                if sum(bool(c) for c in cols) < 2:
                    break
                body.append(cols)
                j += 1
            if body:
                lay.tables.append(Table([c[0] for c in cells], body))
                i = j
                continue
        if len(cells) == 2:
            lay.fields.append((cells[0][0].rstrip(":").strip(), cells[1][0]))
        elif len(cells) == 1 and re.match(r"^[^:]{2,40}:\s*\S", cells[0][0]):
            k, v = cells[0][0].split(":", 1)
            lay.fields.append((k.strip(), v.strip()))
        else:
            loose.append((i, " ".join(c[0] for c in cells), max(it[4] for it in rows[i])))
        i += 1
    head = [x for x in loose if x[0] < 3]
    if head:
        t = max(head, key=lambda x: x[2])
        lay.title = t[1]
        rest = [x for x in head if x is not t]
        if rest:
            lay.issuer = rest[0][1]
        loose = [x for x in loose if x is not t and (not rest or x is not rest[0])]
    lay.text = [x[1] for x in loose]
    return lay


def parse(text: str) -> Layout:
    """Inverse of render(), so stored contexts can be re-noised or inspected."""
    text = text.strip()
    if text == NO_IMAGE:
        return Layout(kind="none")
    if text.startswith("[PHOTO]"):
        return Layout(kind="photo", text=text[len("[PHOTO]"):].split())
    lines = text.splitlines()
    lay = Layout(title=lines[0].removeprefix("[DOC]").strip())
    table: Table | None = None
    for i, ln in enumerate(lines[1:]):
        if ln.startswith("[TABLE]"):
            table = Table([c.strip() for c in ln[len("[TABLE]"):].split("|")], [])
        elif ln == "[/TABLE]" and table is not None:
            lay.tables.append(table)
            table = None
        elif table is not None:
            table.rows.append([c.strip() for c in ln.split("|")])
        elif ": " in ln:
            k, v = ln.split(": ", 1)
            lay.fields.append((k, v))
        elif i == 0:
            lay.issuer = ln
        else:
            lay.text.append(ln)
    return lay
