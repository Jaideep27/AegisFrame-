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
    ap.add_argument("--model", default="models/aegis-v07-r2-onnx")
    ap.add_argument("--f", type=int, default=1)
    a = ap.parse_args()
    from aegisframe.harrier_guard import canonical_instruction

    bb = Blackboard(HarrierModel(a.model, threads=8), f=a.f)
    rows = [json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()]
    labelled = any("expected" in r for r in rows)
    ok = ha = bb_ = 0
    harmful = benign = 0
    print(f"{'':2}{'id':>3}  {'want':<7}{'got':<8}{'p_block':>8}  {'changes':<7} prompt")
    for r in rows:
        v = bb.assess(r["context"], r["prompt"], plain=canonical_instruction(r["prompt"]))
        ch = f"{len(v.value_changes)}chg" if v.value_changes else "-"
        want = r.get("expected", "-")
        right = v.decision == want
        ok += right
        if want == "BLOCK":
            harmful += 1; ha += v.decision == "ALLOW"
        if want == "ALLOW":
            benign += 1; bb_ += v.decision == "BLOCK"
        mark = "x " if labelled and not right else "  "
        print(f"{mark}{r['id']:>3}  {want:<7}{v.decision:<8}{v.p['BLOCK']:>8.3f}  {ch:<7} {r['prompt'][:56]}")
    if labelled:
        print(f"\ncorrect {ok}/{len(rows)}   harmful allowed {ha}/{harmful}   benign blocked {bb_}/{benign}")


if __name__ == "__main__":
    main()
