"""Simulated OCR: turn a clean record layout into what a real OCR pass would plausibly return.

Real OCR on phone photos and scans confuses characters (0/O, 1/l, ₹ read as 2), drops or merges
lines, loses table structure, and sometimes misses field pairs. Training on clean text only would
make the model brittle at runtime, so most contexts are noised. The output is still rendered with
aegisframe.ocr.layout.render(), the same format the runtime produces.
"""

from __future__ import annotations

import random

from aegisframe.ocr.layout import Layout, Table, render

CONFUSE = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5", "8": "B", "B": "8", "₹": "2", "€": "E",
           "£": "E", ",": ".", ".": ",", "rn": "m", "m": "rn", "I": "l", "₦": "N", "¥": "Y"}


def _char_noise(s: str, rng: random.Random, p: float) -> str:
    out, i = [], 0
    while i < len(s):
        two = s[i:i + 2]
        if two in CONFUSE and rng.random() < p:
            out.append(CONFUSE[two])
            i += 2
            continue
        c = s[i]
        r = rng.random()
        if c in CONFUSE and r < p:
            out.append(CONFUSE[c])
        elif r < p * 0.3 and c != " ":
            pass  # dropped character
        elif r < p * 0.4:
            out.append(c + " ")  # split token
        else:
            out.append(c)
        i += 1
    return "".join(out)


def noisy(layout: Layout, rng: random.Random, level: str | None = None) -> Layout:
    """level: clean | light | heavy (random if None)."""
    level = level or rng.choices(["clean", "light", "heavy"], [0.25, 0.5, 0.25])[0]
    if level == "clean":
        return layout
    p = 0.01 if level == "light" else 0.04
    lay = Layout(layout.kind, _char_noise(layout.title, rng, p), layout.issuer, [], [], [])
    for k, v in layout.fields:
        r = rng.random()
        if r < (0.03 if level == "light" else 0.1):
            continue  # line not detected
        if r < (0.08 if level == "light" else 0.25):
            lay.text.append(_char_noise(f"{k} {v}", rng, p))  # label and value not paired
        else:
            lay.fields.append((_char_noise(k, rng, p / 2), _char_noise(v, rng, p)))
    for t in layout.tables:
        if rng.random() < (0.2 if level == "light" else 0.5):
            # table structure lost: rows come back as loose lines, sometimes header merged in
            lay.text.append(" ".join(t.header))
            lay.text += [_char_noise(" ".join(r), rng, p) for r in t.rows]
        else:
            lay.tables.append(Table(t.header, [[_char_noise(c, rng, p) for c in r] for r in t.rows]))
    lay.text += [_char_noise(x, rng, p) for x in layout.text]
    if level == "heavy" and len(lay.fields) > 3 and rng.random() < 0.4:
        i = rng.randrange(len(lay.fields) - 1)  # reading-order swap
        lay.fields[i], lay.fields[i + 1] = lay.fields[i + 1], lay.fields[i]
    if rng.random() < 0.15:
        lay.issuer = ""  # logo text not read
    return lay


def context(layout: Layout | None, rng: random.Random, level: str | None = None) -> str:
    return render(None if layout is None else noisy(layout, rng, level))
