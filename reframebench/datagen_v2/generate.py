"""Generate the v2 dataset.

    # CPU pipeline test with a small local model
    uv run --group train python -m reframebench.datagen_v2.generate --groups 6 --backend hf \
        --model Qwen/Qwen3-1.7B --out data/aegis_v2/pilot

    # Kaggle GPU (vLLM)
    python -m reframebench.datagen_v2.generate --groups 12000 --backend vllm --model <open-weight instruct model> \
        --out /kaggle/working/aegis_v2 --judge

    # any OpenAI-compatible server (vllm serve / llama.cpp / Ollama)
    python -m reframebench.datagen_v2.generate --groups 200 --backend openai --base-url http://localhost:8000/v1 \
        --model <model>

Writes <out>/records.jsonl, <out>/rejects.jsonl and <out>/specs.jsonl. Re-running resumes: groups already
written (as records or rejects) are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from . import split as S, teacher as T
from .spec import Spec, sample
from .validate import assemble


def _backend(a) -> T.Backend:
    if a.backend == "hf":
        return T.HFLocal(a.model)
    if a.backend == "openai":
        return T.OpenAICompat(a.base_url, a.model, workers=a.workers)
    kw = json.loads(a.vllm_kwargs) if a.vllm_kwargs else {}
    return T.VLLMOffline(a.model, **kw)


def _done(out: Path) -> set[str]:
    seen = set()
    for name in ("records.jsonl", "rejects.jsonl"):
        f = out / name
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    seen.add(r.get("group") or r["spec_id"].rsplit("-", 1)[0])
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", choices=["hf", "openai", "vllm"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--vllm-kwargs", default="", help='JSON, e.g. {"tensor_parallel_size": 2, "max_model_len": 8192}')
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=1, help="parallel requests for --backend openai")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--judge", action="store_true", help="also score realism and label agreement with the teacher")
    ap.add_argument("--out", default="data/aegis_v2/pilot")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = sample(a.groups, a.seed)
    with open(out / "specs.jsonl", "w", encoding="utf-8") as f:
        for s in specs:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    groups: dict[str, list[Spec]] = defaultdict(list)
    for s in specs:
        groups[s.group].append(s)
    todo = [g for gid, g in groups.items() if gid not in _done(out)]
    print(f"{len(groups)} groups, {len(todo)} to generate with {a.backend}:{a.model}")

    be = _backend(a)
    stats: Counter = Counter()
    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        raws = be.generate([T.build_prompt(g) for g in chunk], a.temperature, a.max_tokens)
        with open(out / "raw.jsonl", "a", encoding="utf-8") as f:  # keep every output, parsed or not
            for g, raw in zip(chunk, raws):
                f.write(json.dumps({"group": g[0].group, "raw": raw}, ensure_ascii=False) + "\n")
        recs, rejs = [], []
        for g, raw in zip(chunk, raws):
            if raw.startswith("[backend error]"):  # not written anywhere, so a re-run retries it
                stats.update({"backend_error": 1})
                continue
            r, j = assemble(g, raw, be.name)
            recs += r
            rejs += j
        if a.judge and recs:
            by_id = {s.id: s for g in chunk for s in g}
            verdicts = be.generate([T.judge_prompt(by_id[r["id"]], r["context_clean"], r["prompt"], r["history"])
                                    for r in recs], 0.0, 200)
            for r, v in zip(recs, verdicts):
                r["judge"] = T.parse_json(v) or {"error": v[:300]}
        with open(out / "records.jsonl", "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out / "rejects.jsonl", "a", encoding="utf-8") as f:
            for j in rejs:
                f.write(json.dumps({"spec_id": j.spec_id, "reason": j.reason, "raw": j.raw}, ensure_ascii=False) + "\n")
        stats.update({"records": len(recs)})
        stats.update(j.reason.split(":")[0] for j in rejs)
        print(f"[{i + len(chunk)}/{len(todo)} groups, {time.time() - t0:.0f}s] {dict(stats)}", flush=True)


def finalize(out: str, ood_teachers: list[str] | None = None) -> None:
    """Dedup and split all records under <out> -> <out>/dataset.jsonl."""
    p = Path(out)
    recs = [json.loads(x) for x in (p / "records.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    kept, dropped = S.dedup(recs)
    S.assign(kept, frozenset(ood_teachers or []))
    with open(p / "dataset.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(kept)} records ({dropped} near-duplicates dropped):",
          dict(Counter(r["split"] for r in kept)), dict(Counter(r["labels"]["decision"] for r in kept)))


if __name__ == "__main__":
    main()
