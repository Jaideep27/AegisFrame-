"""Score a small hand-written prompt set against the shipped v0.7 runtime.

Each line of the input is {"id", "context", "prompt"} plus optional "expected" and "probes". Use it for
Claude-written probes, prompts from another model, and the human-written test set later - same shape.
Without "expected" it just reports what the model decided.

    uv run python -m training.run_probe_set data/probes/claude_offstyle_v1.jsonl
"""

from __future__ import annotations

import json
import sys

from aegisframe.local.harrier import HarrierModel


def main(path: str, model_dir: str = "models/aegis-v07-r1-onnx") -> None:
    m = HarrierModel(model_dir, threads=8)
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    labelled = any("expected" in r for r in rows)
    ok = harmful_allowed = benign_blocked = 0
    print(f"{'':2}{'id':>3}  {'want':<7}{'got':<7}{'p_block':>8}  {'intent':<19}prompt")
    for r in rows:
        a = m.analyze(r["context"], r["prompt"])
        want = r.get("expected", "-")
        right = a.decision == want
        ok += right
        harmful_allowed += want == "BLOCK" and a.decision == "ALLOW"
        benign_blocked += want == "ALLOW" and a.decision == "BLOCK"
        mark = "x " if labelled and not right else "  "
        print(f"{mark}{r['id']:>3}  {want:<7}{a.decision:<7}{a.p_block:>8.3f}  "
              f"{a.top_intent[0]:<19}{r['prompt'][:70]}")
    if labelled:
        print(f"\ncorrect {ok}/{len(rows)}   harmful allowed {harmful_allowed}   benign blocked {benign_blocked}")
    else:
        counts = {}
        for r in rows:
            pass
        print(f"\n{len(rows)} prompts, unlabelled")


if __name__ == "__main__":
    main(*sys.argv[1:])
