"""Export ReframeBench for sharing and for human annotation.

    uv run python -m reframebench.build --out data/reframebench_v1

Writes:
  images/<doc>.png          the fictional records
  cases.jsonl               all 320 cases with expected labels
  annotation_sheet.csv      plain requests + controls only, WITHOUT our labels or schema,
                            for 2-3 independent annotators (target agreement kappa >= 0.7)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

from .cases import build_cases
from .docs import CATALOG, render, render_photo, to_png


def export(out: str | Path) -> None:
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    for key, tpl in CATALOG.items():
        (out / "images" / f"{key}.png").write_bytes(to_png(render(tpl)))
    (out / "images" / "photo.png").write_bytes(to_png(render_photo()))

    cases = build_cases()
    with (out / "cases.jsonl").open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(dataclasses.asdict(c), ensure_ascii=False) + "\n")

    label_col = "your_label (allow/review/block)"
    with (out / "annotation_sheet.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "kind", "image", "request", "verified_role", label_col, "note"])
        w.writeheader()
        for c in cases:
            if c.kind not in {"plain", "control"}:
                continue  # framed / multi-turn / retry variants inherit the plain label by definition
            w.writerow({
                "case_id": c.id,
                "kind": c.kind,
                "image": f"images/{c.image_ref}.png" if c.image_ref else "(none - generate from scratch)",
                "request": " / ".join(c.turns),
                "verified_role": ", ".join(c.verified_roles),
                label_col: "",
                "note": "",
            })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reframebench_v1")
    export(ap.parse_args().out)


if __name__ == "__main__":
    main()
