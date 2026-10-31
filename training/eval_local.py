"""End-to-end evaluation of the Guard on the generated corpus.

    uv run python training/eval_local.py --model models/aegis-local-v1 --split test_heldout --limit 3000
    uv run python training/eval_local.py --model models/aegis-local-v1 --images 300      # real OCR on rendered images

Systems compared:  rules (no model)  vs  local (rules + local model).
Every record is re-rendered and read either by the oracle OCR (isolates the model) or by
RapidOCR (--images: the real pipeline). Labels come from the corpus (decision_final).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aegisframe import AegisConfig, Guard  # noqa: E402
from aegisframe.local.runtime import LocalModel  # noqa: E402
from reframebench.datagen.export import render_example  # noqa: E402
from reframebench.datagen.builder import Example  # noqa: E402
from reframebench.docs import OracleOcr  # noqa: E402

ORDER = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2}


def rows_from(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[:limit] if limit else rows


def as_example(r: dict) -> Example:
    return Example(id=r["id"], split=r["split"], context=r["context"], prompt=r["prompt"],
                   spans=[tuple(s) for s in r["spans"]], canonical=r["canonical"], labels=r["labels"],
                   reason=r["reason"], scene=r["scene"])


def run_system(name: str, guard: Guard, rows: list[dict], images: dict[str, bytes | None]) -> dict:
    preds, latency = [], []
    for r in rows:
        turns = r["prompt"].split(" [TURN] ")
        image = images[r["id"]]
        t = time.perf_counter()
        d = None
        for turn in turns:
            d = guard.check_request(turn, image=image, session_id=f"{name}-{r['id']}")
        latency.append((time.perf_counter() - t) * 1000 / len(turns))
        preds.append(d.action.name)
    return score(rows, preds, latency)


def score(rows: list[dict], preds: list[str], latency: list[float]) -> dict:
    n = len(rows)
    truth = [r["labels"]["decision_final"] for r in rows]
    harmful = [i for i, t in enumerate(truth) if t != "ALLOW"]
    benign = [i for i, t in enumerate(truth) if t == "ALLOW"]
    by_intent, by_twist = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        ok = preds[i] == truth[i]
        by_intent[r["labels"]["intent"]].append(ok)
        for t in r["labels"]["twist"] or ["none"]:
            by_twist[t].append(ok)
    lat = sorted(latency)
    return {
        "n": n,
        "accuracy": sum(p == t for p, t in zip(preds, truth)) / n,
        "harmful_allow_rate": sum(preds[i] == "ALLOW" for i in harmful) / max(1, len(harmful)),
        "harmful_halt_rate": sum(preds[i] != "ALLOW" for i in harmful) / max(1, len(harmful)),
        "benign_block_rate": sum(preds[i] == "BLOCK" for i in benign) / max(1, len(benign)),
        "benign_halt_rate": sum(preds[i] != "ALLOW" for i in benign) / max(1, len(benign)),
        "too_strict": sum(ORDER[p] > ORDER[t] for p, t in zip(preds, truth)) / n,
        "too_lenient": sum(ORDER[p] < ORDER[t] for p, t in zip(preds, truth)) / n,
        "latency_ms_p50": lat[len(lat) // 2], "latency_ms_p95": lat[int(len(lat) * 0.95)],
        "by_intent": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_intent.items())},
        "by_twist": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_twist.items())},
        "confusion": dict(Counter(f"{t}->{p}" for p, t in zip(preds, truth))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/aegis-local-v1")
    ap.add_argument("--data", default="data/aegis_corpus")
    ap.add_argument("--split", default="test_heldout")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--images", type=int, default=0, help="evaluate N rendered images with real OCR instead")
    ap.add_argument("--out", default="results/local_eval.json")
    a = ap.parse_args()

    data = Path(a.data)
    if a.images:
        from aegisframe.ground import RapidOcrEngine

        rows = [r for r in rows_from(data / "images.jsonl", 0)][: a.images]
        ocr = RapidOcrEngine()
        images = {r["id"]: (data / r["image"]).read_bytes() if r["image"] else None for r in rows}
    else:
        rows = rows_from(data / f"{a.split}.jsonl", a.limit)
        ocr = OracleOcr()
        images = {}
        for r in rows:
            rendered = render_example(as_example(r))
            if rendered is None:
                images[r["id"]] = None
                continue
            png, lines = rendered
            from aegisframe.schema import OcrLine

            images[r["id"]] = ocr.register(png, [OcrLine(**ln) for ln in lines]) if lines else png

    off = AegisConfig.model_validate({"local_model": {"enabled": "off"}})
    on = AegisConfig.model_validate({"local_model": {"enabled": "on", "path": a.model}})
    results = {"rules": run_system("rules", Guard(ocr=ocr, config=off), rows, images)}
    model = LocalModel.load(a.model)
    results["local"] = run_system("local", Guard(ocr=ocr, config=on, local_model=model), rows, images)
    results["meta"] = {"split": "images(real OCR)" if a.images else a.split, "model": a.model,
                       "model_metrics": model.config.get("metrics")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name in ("rules", "local"):
        r = results[name]
        print(f"\n== {name} (n={r['n']}) ==")
        for k in ("accuracy", "harmful_allow_rate", "benign_block_rate", "benign_halt_rate", "too_strict", "too_lenient",
                  "latency_ms_p50", "latency_ms_p95"):
            print(f"  {k:22} {r[k]:.3f}")
        print("  by_twist ", r["by_twist"])
        print("  by_intent", r["by_intent"])


if __name__ == "__main__":
    main()
