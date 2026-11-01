"""Honest v0.7 evaluation: model-only vs the fused blackboard vs ablations, on every test set we have.

The model is run once per record and its analysis cached, so the fusion ablations are cheap. Reports
accuracy, harmful-allowed and benign-blocked per split, plus a keyword-filter baseline. Writes
results/eval_v2.json with every number and the per-record decisions.

    uv run python -m training.eval_v2 --model models/aegis-v07-r2-onnx --data data/aegis_v2/en_all

Baselines deliberately omitted here: Llama Guard 3 and ShieldGemma are licence-gated (need a Hugging
Face account the project has not set up); Qwen3Guard is ungated and can be added with --qwen once a
local copy exists. The v0.6 engine is a different (image) pipeline and is compared separately.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from aegisframe.council.blackboard import DEFAULT_PRIOR, Blackboard
from aegisframe.council.casebank import CaseBank
from aegisframe.local.harrier import HarrierModel
from reframebench.datagen_v2.validate import plain_text

DEC = ["ALLOW", "REVIEW", "BLOCK"]
_KEYWORD = re.compile(r"\b(change|edit|make it|set|update|replace|increase|raise|remove|delete|add|"
                      r"forge|fake|without (?:the )?(?:watermark|sample))\b", re.I)


class Cached:
    """A model stand-in that replays a precomputed analysis, so ablations do not re-run the net."""

    def __init__(self, analysis):
        self._a = analysis

    def analyze(self, context, prompt, plain=None):
        return self._a


def load_splits(path: Path):
    by = defaultdict(list)
    for line in (path / "dataset.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["split"]].append(r)
    return by


def as_record(r: dict) -> tuple[str, str, str, str]:
    ctx = "Record: [no image attached]\n" if r["context"] == "[NO IMAGE]" else f"Record:\n{r['context']}\n"
    return ctx, f"Message: {r['prompt']}", f"Message: {plain_text(r)}", r["labels"]["decision"]


def metrics(gold: list[str], pred: list[str]) -> dict:
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    harmful = [i for i, g in enumerate(gold) if g == "BLOCK"]
    benign = [i for i, g in enumerate(gold) if g == "ALLOW"]
    ha = sum(pred[i] == "ALLOW" for i in harmful)
    bb = sum(pred[i] == "BLOCK" for i in benign)
    return {"n": n, "accuracy": round(acc, 3),
            "harmful_allowed": f"{ha}/{len(harmful)}", "benign_blocked": f"{bb}/{len(benign)}"}


def keyword_decision(prompt: str) -> str:
    return "BLOCK" if _KEYWORD.search(prompt) else "ALLOW"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/aegis-v07-r2-onnx")
    ap.add_argument("--data", default="data/aegis_v2/en_all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/eval_v2.json")
    a = ap.parse_args()

    model = HarrierModel(a.model, threads=a.threads)
    fusion_dir = Path(a.model) / "fusion"
    casebank = CaseBank.load(fusion_dir / "casebank.npz") if (fusion_dir / "casebank.npz").exists() else None

    splits = load_splits(Path(a.data))
    datasets = {k: splits[k] for k in ("val", "test_id", "test_ood") if splits.get(k)}

    configs = {
        "keyword_baseline": None,
        "model_only": ("model", None, 0),
        "blackboard_f0": ("fuse", casebank, 0),
        "blackboard_f0_no_casebank": ("fuse", None, 0),
        "blackboard_f1": ("fuse", casebank, 1),
    }

    results: dict = {"model": a.model, "data": a.data, "splits": {}}
    per_record: dict = defaultdict(dict)

    for split, rows in datasets.items():
        gold = [r["labels"]["decision"] for r in rows]
        # one model pass per record
        cached = []
        for r in rows:
            ctx, prompt, plain, _ = as_record(r)
            cached.append((ctx, prompt, plain, model.analyze(ctx, prompt, plain)))
        split_res = {}
        for name, cfg in configs.items():
            preds = []
            for (ctx, prompt, plain, analysis), r in zip(cached, rows):
                if cfg is None:
                    preds.append(keyword_decision(r["prompt"]))
                elif cfg[0] == "model":
                    preds.append(analysis.decision)
                else:
                    bb = Blackboard(Cached(analysis), casebank=cfg[1], prior=DEFAULT_PRIOR, f=cfg[2])
                    preds.append(bb.assess(ctx, prompt, plain=plain).decision)
            split_res[name] = metrics(gold, preds)
            per_record[split][name] = preds
        results["splits"][split] = split_res
        print(f"\n{split} (n={len(rows)})")
        for name, m in split_res.items():
            print(f"  {name:<28} acc {m['accuracy']:.3f}  harmful_allowed {m['harmful_allowed']:>7}"
                  f"  benign_blocked {m['benign_blocked']:>7}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
