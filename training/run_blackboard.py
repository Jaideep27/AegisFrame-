"""Score a probe set through the full v0.7 blackboard (model + value-diff + one-sided agents).

    uv run python -m training.run_blackboard data/probes/gpt_redteam_v1.jsonl --f 1
"""
from __future__ import annotations
import argparse, json
from aegisframe.council.blackboard import Blackboard
from aegisframe.local.harrier import HarrierModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--model", default="models/aegis-v07-r1-onnx")
    ap.add_argument("--f", type=int, default=1)
    a = ap.parse_args()
    bb = Blackboard(HarrierModel(a.model, threads=8), f=a.f)
    rows = [json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()]
    print(f"{'id':>3}  {'decision':<8}{'p_block':>8}  {'changes':<8} prompt")
    for r in rows:
        v = bb.assess(r["context"], r["prompt"])
        ch = f"{len(v.value_changes)} chg" if v.value_changes else "-"
        print(f"{r['id']:>3}  {v.decision:<8}{v.p['BLOCK']:>8.3f}  {ch:<8} {r['prompt'][:58]}")


if __name__ == "__main__":
    main()
