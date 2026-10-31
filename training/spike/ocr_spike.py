"""Phase 1b spike: how well does RapidOCR + geometric layout rebuild a record's fields and tables?

    uv run python training/spike/ocr_spike.py --n 24 [--image my_marksheet.png]

Renders doclib records (clean, scan-like, phone-photo-like), reads them with RapidOCR (PP-OCRv6 by
default), rebuilds the Layout with aegisframe.ocr.layout.from_lines and reports:
  CER            character error rate of all text vs the rendered text
  field F1       exact (label, value) pairs recovered
  cell accuracy  table cells recovered in the right row and column
  latency        OCR + layout per image
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from aegisframe.ground import RapidOcrEngine
from aegisframe.ocr.layout import from_lines, render
from reframebench.datagen_v2 import doclib, regions
from reframebench.datagen_v2.render import degrade, draw, png


def cer(ref: str, hyp: str) -> float:
    a, b = ref, hyp
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


def norm(s: str) -> str:
    return " ".join(s.lower().replace(":", " ").split())


def scores(ref, got) -> tuple[float, float]:
    rf = {(norm(k), norm(v)) for k, v in ref.fields}
    gf = {(norm(k), norm(v)) for k, v in got.fields}
    tp = len(rf & gf)
    f1 = tp / max(1, len(rf))  # recall: extra "fields" are often 2-column table rows, which is fine
    # a table cell counts if it comes back in the same row (matched by its first cell) and column;
    # two-column tables (label | amount) may come back as fields, which carry the same information
    ref_cells = {(norm(row[0]), c, norm(x)) for t in ref.tables for row in t.rows for c, x in enumerate(row)}
    got_cells = {(norm(row[0]), c, norm(x)) for t in got.tables for row in t.rows for c, x in enumerate(row)}
    got_cells |= {(k, c, x) for k, v in gf for c, x in ((0, k), (1, v))}
    acc = len(ref_cells & got_cells) / len(ref_cells) if ref_cells else float("nan")
    return f1, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--per-family", action="store_true")
    ap.add_argument("--image", help="also read this image and print the rebuilt layout")
    a = ap.parse_args()
    ocr = RapidOcrEngine()
    rng = random.Random(0)
    fams = ["marksheet", "payslip", "bank_statement", "tax_invoice", "lab_report", "payment_screenshot",
            "national_id", "utility_bill"]
    res: dict[str, list] = {"clean": [], "scan": [], "photo": []}
    for i in range(a.n):
        doc = doclib.make(fams[i % len(fams)], rng.choice(["IN", "US", "UK", "DE", "BR", "ID", "KE", "MX"]), rng)
        ref = doc.layout()
        base = draw(ref)
        for level in res:
            img = base if level == "clean" else degrade(base, rng, level)
            t = time.perf_counter()
            got = from_lines(ocr.read(png(img)))
            ms = (time.perf_counter() - t) * 1000
            f1, acc = scores(ref, got)
            res[level].append((cer(render(ref), render(got)), f1, acc, ms))
    if a.per_family:
        for fam in fams:
            rows = [x for x, f in zip(res["clean"], [fams[i % len(fams)] for i in range(a.n)]) if f == fam]
            print(f"  {fam:20} clean: field recall {statistics.mean(r[1] for r in rows):.3f}  "
                  f"cell acc {statistics.mean(r[2] for r in rows):.3f}")
    print(f"{'condition':8} {'CER':>6} {'field rec':>9} {'cell acc':>9} {'ms p50':>7}")
    for level, rows in res.items():
        c, f, acc, ms = zip(*rows)
        accs = [x for x in acc if x == x]
        print(f"{level:8} {statistics.mean(c):6.3f} {statistics.mean(f):9.3f} "
              f"{statistics.mean(accs) if accs else float('nan'):9.3f} {statistics.median(ms):7.0f}")
    if a.image:
        with open(a.image, "rb") as fh:
            print(render(from_lines(ocr.read(fh.read()))))


if __name__ == "__main__":
    main()
