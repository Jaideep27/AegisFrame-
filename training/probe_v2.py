"""Behavioural probes for a trained v0.7 checkpoint: what is the model actually using to decide?

Run where the checkpoint lives (the Mac):

    uv run --group train python -m training.probe_v2 --model models/aegis-v07-r1 \
        --data data/aegis_v2/en_all --device mps --out runs/2026-11-01-probe

Probes
  reeval      reproduce the training run's val/test_id/test_ood numbers (sanity check on the reload)
  nocontext   blank the record context -> how much of the decision comes from the prompt alone
  storyleak   feed the raw story-included text to the story-blind head -> is story stripping load-bearing
  maxlen      re-score test_ood at 128/256/384/512 tokens -> length robustness
  latency     single-request CPU latency p50/p95 at 64/256/1024 tokens
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from training import train_v2 as T


def build(rows: list[dict], enc: T.Encoder, model: T.PilotModel, plain_is_full: bool = False) -> list[dict]:
    items = []
    for r in rows:
        ctx, full, plain = T.texts(r)
        items.append({"r": r, "full": enc.ids(ctx, full),
                      "plain": enc.ids(ctx, full if plain_is_full else plain), "y": T.targets(r)})
    T.cache_bottom(model, {"x": items})
    return items


def scored(model: T.PilotModel, items: list[dict], use: str = "final") -> dict:
    preds = T.predict(model, items, 0)
    gold = [p["gold"] for p in preds]
    return {**T.metrics(gold, [p[use] for p in preds]),
            "intent_accuracy": round(sum(p["intent"] == p["gold_intent"] for p in preds) / len(preds), 3)}


def latency(model: T.PilotModel, pad_id: int, lengths=(64, 256, 1024), reps: int = 12) -> dict:
    """End-to-end for one request: both passes bottom + top + heads, on whatever device is selected."""
    out = {}
    for n in lengths:
        ids = [pad_id or 1] * n
        ms = []
        for i in range(reps + 2):
            t0 = time.perf_counter()
            with torch.no_grad():
                fh, ph = model.bottom(ids), model.bottom(ids)
                model([fh], [ph])
            if i >= 2:  # discard warm-up
                ms.append((time.perf_counter() - t0) * 1000)
        ms.sort()
        out[str(n)] = {"p50_ms": round(statistics.median(ms), 1),
                       "p95_ms": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 1)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/aegis-v07-r1")
    ap.add_argument("--data", default="data/aegis_v2/en_all")
    ap.add_argument("--out", default="runs/probe")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--train-layers", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--latency-device", default="cpu", help="users run on CPU, so time it there")
    ap.add_argument("--skip", default="", help="comma-separated probes to skip")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N rows per split")
    a = ap.parse_args()

    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    torch.set_num_threads(a.threads)
    dev = a.device
    if dev == "auto":
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    T.DEV = torch.device(dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = T.AutoTokenizer.from_pretrained(T.ENCODER)
    model = T.PilotModel(a.train_layers).to(T.DEV).eval()
    state = torch.load(Path(a.model) / "checkpoint.pt", map_location=T.DEV)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"device {dev} | loaded {len(state)} tensors | unexpected {len(unexpected)}", flush=True)
    enc = T.Encoder(tok, a.max_len)
    splits = T.load(Path(a.data))
    if a.limit:
        splits = {k: v[: a.limit] for k, v in splits.items()}
    res: dict = {"model": a.model, "data": a.data, "device": dev, "max_len": a.max_len,
                 "checkpoint_tensors": len(state), "unexpected_keys": list(unexpected)[:5]}

    if "reeval" not in skip:
        res["reeval"] = {}
        for name in ("val", "test_id", "test_ood"):
            items = build(splits[name], enc, model)
            res["reeval"][name] = {k: scored(model, items, k) for k in ("final", "plain", "sentinel")}
            print(f"reeval {name}: {json.dumps(res['reeval'][name]['final'])}", flush=True)

    if "nocontext" not in skip:
        rows = [r for r in splits["test_ood"] if r["context"] != "[NO IMAGE]"]
        blank = []
        for r in rows:
            b = copy.deepcopy(r)
            b["context"] = "[NO IMAGE]"
            blank.append(b)
        res["nocontext"] = {"n_with_record": len(rows),
                            "with_context": scored(model, build(rows, enc, model)),
                            "context_blanked": scored(model, build(blank, enc, model))}
        print(f"nocontext: {json.dumps(res['nocontext'])}", flush=True)

    if "storyleak" not in skip:
        rows = [r for r in splits["test_ood"] if any(s["type"] == "STORY" for s in r.get("spans") or [])]
        res["storyleak"] = {"n_with_story": len(rows),
                            "story_stripped": scored(model, build(rows, enc, model), "plain"),
                            "story_included": scored(model, build(rows, enc, model, plain_is_full=True), "plain")}
        print(f"storyleak: {json.dumps(res['storyleak'])}", flush=True)

    if "maxlen" not in skip:
        res["maxlen"] = {}
        for n in (128, 256, 384, 512):
            items = build(splits["test_ood"], T.Encoder(tok, n), model)
            res["maxlen"][str(n)] = scored(model, items)
            print(f"maxlen {n}: {json.dumps(res['maxlen'][str(n)])}", flush=True)

    if "latency" not in skip:
        T.DEV = torch.device(a.latency_device)
        res["latency"] = {"device": a.latency_device, "threads": a.threads,
                          **latency(model.to(T.DEV), tok.pad_token_id)}
        print(f"latency: {json.dumps(res['latency'])}", flush=True)

    (out / "probe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {out / 'probe.json'}", flush=True)


if __name__ == "__main__":
    main()
