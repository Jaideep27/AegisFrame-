"""Train the local AegisFrame model on CPU and export it to ONNX.

    uv run --group train python training/train_local.py --data data/aegis_corpus --out models/aegis-local-v1

Output directory (what the SDK loads):
  model.onnx            fp32 graph       model.int8.onnx   dynamic int8 (default at runtime)
  tokenizer.json        fast tokenizer   config.json       label maps, max lengths, metrics
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aegisframe.local.features import LABELS  # noqa: E402
from training.model import POOLED, AegisLocalModel, ExportWrapper  # noqa: E402

TAG_ID = {t: i for i, t in enumerate(LABELS["tags"])}
IDX = {k: {v: i for i, v in enumerate(vals)} for k, vals in LABELS.items()}
MAX_FULL, MAX_PLAIN = 112, 96


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def encode(tok, rows: list[dict]) -> dict:
    """Tokenise both passes; align character spans to BIO tags on the prompt tokens."""
    full = tok([r["context"] for r in rows], [r["prompt"] for r in rows], truncation="only_first",
               max_length=MAX_FULL, padding=True, return_offsets_mapping=True, return_tensors="pt")
    plain = tok([r["context"] for r in rows], [r["canonical"] for r in rows], truncation="only_first",
                max_length=MAX_PLAIN, padding=True, return_tensors="pt")
    tags = torch.full(full["input_ids"].shape, -100, dtype=torch.long)
    for i, r in enumerate(rows):
        seq = full.sequence_ids(i)
        offsets = full["offset_mapping"][i].tolist()
        prev_tag = None
        for j, (s, e) in enumerate(offsets):
            if seq[j] != 1 or s == e:
                continue
            label = "O"
            for a, b, t in r["spans"]:
                if s >= a and e <= b:
                    label = ("I-" if prev_tag == (t, a) else "B-") + t
                    prev_tag = (t, a)
                    break
            else:
                prev_tag = None
            tags[i, j] = TAG_ID[label]
    lab = [r["labels"] for r in rows]
    twist = torch.zeros(len(rows), len(LABELS["twist"]))
    for i, l in enumerate(lab):
        for t in l["twist"]:
            twist[i, IDX["twist"][t]] = 1
    y = {
        "tags": tags,
        "intent": torch.tensor([IDX["intent"][l["intent"]] for l in lab]),
        "twist": twist,
        "purpose": torch.tensor([[float(l["purpose"])] for l in lab]),
        "decision": torch.tensor([IDX["decision"][l["decision"]] for l in lab]),
        "field_class": torch.tensor([IDX["field_class"][l["field_class"]] for l in lab]),
        "pattern": torch.tensor([IDX["pattern"][l["pattern"]] for l in lab]),
        "misuse": torch.tensor([IDX["misuse"][l["misuse"]] for l in lab]),
        "benign_need": torch.tensor([IDX["benign_need"][l["benign_need"]] for l in lab]),
    }
    full.pop("offset_mapping")
    return {"full": dict(full), "plain": dict(plain), "y": y}


CE = nn.CrossEntropyLoss(ignore_index=-100)
BCE = nn.BCEWithLogitsLoss()
WEIGHTS = {"tags": 1.0, "intent": 1.0, "twist": 0.5, "purpose": 0.5, "decision": 1.5, "field_class": 0.7,
           "pattern": 0.7, "misuse": 0.5, "benign_need": 0.5}


def losses(model, batch) -> tuple[torch.Tensor, dict]:
    f = model(**batch["full"])
    p = model(**batch["plain"])
    y = batch["y"]
    parts = {
        "tags": CE(f["tags"].reshape(-1, f["tags"].shape[-1]), y["tags"].reshape(-1)),
        "intent": CE(f["intent"], y["intent"]),
        "twist": BCE(f["twist"], y["twist"]),
        "purpose": BCE(f["purpose"], y["purpose"]),
    }
    for k in ("decision", "field_class", "pattern", "misuse", "benign_need"):
        parts[k] = CE(p[k], y[k])
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    return total, {k: float(v) for k, v in parts.items()}


@torch.no_grad()
def evaluate(model, tok, rows: list[dict], bs: int = 64) -> dict:
    model.eval()
    hits = {k: 0 for k in ("intent", "decision", "field_class", "pattern", "misuse", "benign_need", "purpose")}
    tag_ok = tag_n = 0
    harmful = harmful_allowed = benign = benign_blocked = 0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        b = encode(tok, chunk)
        f = model(**b["full"])
        p = model(**b["plain"])
        y = b["y"]
        hits["intent"] += int((f["intent"].argmax(-1) == y["intent"]).sum())
        hits["purpose"] += int(((f["purpose"] > 0).float() == y["purpose"]).sum())
        for k in ("decision", "field_class", "pattern", "misuse", "benign_need"):
            hits[k] += int((p[k].argmax(-1) == y[k]).sum())
        mask = y["tags"] != -100
        tag_ok += int((f["tags"].argmax(-1)[mask] == y["tags"][mask]).sum())
        tag_n += int(mask.sum())
        pred = p["decision"].argmax(-1)
        for j, r in enumerate(chunk):
            truth = r["labels"]["decision"]
            if truth != "ALLOW":
                harmful += 1
                harmful_allowed += int(pred[j] == 0)
            else:
                benign += 1
                benign_blocked += int(pred[j] == 2)
    n = len(rows)
    model.train()
    return {**{f"acc_{k}": v / n for k, v in hits.items()}, "acc_tags": tag_ok / max(tag_n, 1),
            "harmful_allow_rate": harmful_allowed / max(harmful, 1), "benign_block_rate": benign_blocked / max(benign, 1)}


def batches(rows: list[dict], bs: int, rng: random.Random) -> list[list[dict]]:
    """Length-bucketed batches: similar lengths together means less padding on CPU."""
    rows = sorted(rows, key=lambda r: len(r["context"]) + len(r["prompt"]))
    out = [rows[i:i + bs] for i in range(0, len(rows), bs)]
    rng.shuffle(out)
    return out


def export_onnx(model: AegisLocalModel, tok, out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model.eval()
    sample = tok(["[DOC] x | Amount: Rs. 500"], ["change the amount to 5000"], return_tensors="pt")
    names = ["tags", *POOLED.keys()]
    torch.onnx.export(
        ExportWrapper(model), (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(out / "model.onnx"), input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=names,
        dynamic_axes={**{n: {0: "batch", 1: "seq"} for n in ("input_ids", "attention_mask", "token_type_ids")},
                      "tags": {0: "batch", 1: "seq"}, **{k: {0: "batch"} for k in POOLED}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(out / "model.onnx"), str(out / "model.int8.onnx"), weight_type=QuantType.QInt8)
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aegis_corpus")
    ap.add_argument("--out", default="models/aegis-local-v1")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="continue from a checkpoint.pt (failure-hunting rounds)")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = load(data / "train.jsonl"), load(data / "val.jsonl"), load(data / "test_heldout.jsonl")
    if a.limit:
        train, val, test = train[:a.limit], val[: a.limit // 10 or 50], test[: a.limit // 10 or 50]

    tok = AutoTokenizer.from_pretrained(a.encoder)
    model = AegisLocalModel(a.encoder)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
        print("initialised from", a.init, flush=True)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": head_params, "lr": a.lr * 10}], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / a.bs)
    total = int(steps_per_epoch * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * total))) * max(0.0, (total - s) / total))

    log = (out / "train_log.jsonl").open("w", encoding="utf-8")
    step, t0 = 0, time.time()
    model.train()
    while step < total:
        for chunk in batches(train, a.bs, rng):
            if step >= total:
                break
            loss, parts = losses(model, encode(tok, chunk))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0 or step == 1:
                rec = {"step": step, "of": total, "loss": float(loss), **parts, "elapsed_min": (time.time() - t0) / 60}
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % max(1, steps_per_epoch // 2) == 0:
                m = evaluate(model, tok, val[:800])
                print("VAL", json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
                torch.save(model.state_dict(), out / "checkpoint.pt")

    metrics = {"val": evaluate(model, tok, val), "test_heldout": evaluate(model, tok, test),
               "train_minutes": (time.time() - t0) / 60, "encoder": a.encoder, "train_examples": len(train)}
    print("FINAL", json.dumps(metrics, indent=2), flush=True)
    torch.save(model.state_dict(), out / "checkpoint.pt")
    export_onnx(model, tok, out)
    (out / "config.json").write_text(json.dumps({
        "format": 1, "encoder": a.encoder, "labels": LABELS, "pooled_heads": list(POOLED),
        "max_full": MAX_FULL, "max_plain": MAX_PLAIN, "metrics": metrics,
    }, indent=2), encoding="utf-8")
    print("exported to", out)


if __name__ == "__main__":
    main()
