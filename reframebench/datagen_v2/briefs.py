"""Turn a round's specs into compact writing briefs, one block per twin group.

    python -m reframebench.datagen_v2.briefs data/aegis_v2/en_r1 > briefs.txt

The writer (a person or a model) reads the briefs and writes batch_NN.json files in the format that
`handwritten.py` assembles; labels never come from the writer, only from the specs.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from . import persuasion as P
from .spec import Spec
from .teacher import spec_names


def briefs(out: Path) -> str:
    groups: dict[str, list[Spec]] = defaultdict(list)
    for line in (out / "specs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            s = Spec(**json.loads(line))
            groups[s.group].append(s)
    lines = []
    for gid, gr in groups.items():
        s0 = gr[0]
        lines.append(f"## {gid} | {s0.family}/{s0.locale} ({'/'.join(spec_names(s0))}) | {s0.lang} | {s0.persona}; "
                     f"{s0.style} | story={s0.story} | tech={s0.technique} | tactics={','.join(s0.tactics) or '-'} | "
                     f"{s0.length} {P.LENGTH[s0.length]} | img={'yes' if s0.context_layout else 'NO'}")
        for letter, s in zip("ABC", gr):
            lines.append(f"  {letter} [{s.role}/{s.plan['intent']}] {s.edit.describe}"
                         + (f"  <<DECEPTIVE: {s.reliance_party}>>" if s.deceptive_purpose else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(briefs(Path(sys.argv[1])))
