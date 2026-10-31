"""Assemble hand-written teacher outputs (compact batch files) through the normal validator.

A batch file is a JSON list of groups:
    [{"g": "<group id>", "v": [{"id": "A", "h": ["<earlier turn>"],
                                "s": [["story" | "request" | "other", "<text>"], ...],
                                "m": {"FIELD": [...], "OLD": [...], "NEW": [...], "MARK": [...], "OBJ": [...]}}]}]
Each group is expanded to the teacher's JSON format and passed to `validate.assemble`, so labels, spans and
checks are exactly those of model-generated groups.

    python -m reframebench.datagen_v2.handwritten data/aegis_v2/claude --teacher claude-opus-5-5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .spec import Spec
from .validate import assemble


def _load_specs(out: Path) -> dict[str, list[Spec]]:
    groups: dict[str, list[Spec]] = defaultdict(list)
    for line in (out / "specs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            groups[d["group"]].append(Spec(**d))
    return groups


def to_raw(group: dict) -> str:
    versions = [{"id": v["id"], "history": v.get("h", []),
                 "segments": [{"type": t, "text": x} for t, x in v["s"]],
                 "mentions": v.get("m", {})} for v in group["v"]]
    return json.dumps({"versions": versions}, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--teacher", default="claude-opus-5-5")
    a = ap.parse_args()
    out = Path(a.out)
    specs = _load_specs(out)
    recs, rejs, stats = [], [], Counter()
    for f in sorted(out.glob("batch_*.json")):
        for g in json.loads(f.read_text(encoding="utf-8")):
            r, j = assemble(specs[g["g"]], to_raw(g), a.teacher)
            recs += r
            rejs += j
            for x in j:
                print(f"REJECT {f.name} {x.spec_id}: {x.reason}")
    stats.update(r["labels"]["decision"] for r in recs)
    with open(out / "records.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "rejects.jsonl", "w", encoding="utf-8") as fh:
        for x in rejs:
            fh.write(json.dumps({"spec_id": x.spec_id, "reason": x.reason, "raw": x.raw}, ensure_ascii=False) + "\n")
    print(f"{len(recs)} records, {len(rejs)} rejects", dict(stats))


if __name__ == "__main__":
    main()
