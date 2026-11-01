"""Build the fusion artifacts for a trained v0.7 model: the case bank and per-agent calibration.

Runs the shipped runtime over the dataset, embeds every story-blind request, builds the harmful and
benign case banks from the train split, calibrates the plain-judge temperature on val, and writes
everything next to the ONNX model as a `fusion/` directory the Guard loads.

    uv run python -m training.train_fusion --model models/aegis-v07-r2-onnx --data data/aegis_v2/en_all

Fitting the dependence-correction weights (w, V from the plan) needs out-of-fold messages, which needs
K encoder retrains; with 813 records that is not estimable, so the pool runs at unit weights (Prop. 1:
exact when the views are conditionally independent and calibrated). This is recorded in fusion.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aegisframe.council.blackboard import DEFAULT_PRIOR
from aegisframe.council.calibrate_fusion import fit_temperature
from aegisframe.council.casebank import CaseBank
from aegisframe.council.fusion import DECISION
from aegisframe.local.harrier import HarrierModel
from reframebench.datagen_v2.validate import plain_text


def load_split(path: Path) -> dict[str, list[dict]]:
    from collections import defaultdict

    by = defaultdict(list)
    for line in (path / "dataset.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["split"]].append(r)
    return by


def embed_split(model: HarrierModel, rows: list[dict]):
    """Return (embeddings, kept_rows, p_block) over a split, one model call each."""
    embs, kept, pblock = [], [], []
    for r in rows:
        ctx = "Record: [no image attached]\n" if r["context"] == "[NO IMAGE]" else f"Record:\n{r['context']}\n"
        a = model.analyze(ctx, f"Message: {r['prompt']}", f"Message: {plain_text(r)}")
        if a.embedding is not None:
            embs.append(a.embedding)
            kept.append(r)
            pblock.append(a.p_block)
    return np.stack(embs), kept, np.array(pblock)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/aegis-v07-r2-onnx")
    ap.add_argument("--data", default="data/aegis_v2/en_all")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    model = HarrierModel(a.model, threads=a.threads)
    splits = load_split(Path(a.data))
    out = Path(a.model) / "fusion"
    out.mkdir(parents=True, exist_ok=True)

    print(f"embedding {len(splits['train'])} train records ...", flush=True)
    train_emb, train_rows, _pb = embed_split(model, splits["train"])
    harmful = np.stack([e for e, r in zip(train_emb, train_rows) if r["labels"]["decision"] == "BLOCK"])
    benign = np.stack([e for e, r in zip(train_emb, train_rows) if r["labels"]["decision"] == "ALLOW"])
    CaseBank(harmful, benign, tau=a.tau, k=a.k).save(out / "casebank.npz")
    print(f"case bank: {len(harmful)} harmful, {len(benign)} benign", flush=True)

    print(f"calibrating on {len(splits['val'])} val records ...", flush=True)
    _, val_rows, p_block = embed_split(model, splits["val"])
    y_block = np.array([int(r["labels"]["decision"] == "BLOCK") for r in val_rows])
    logit = np.log(np.clip(p_block, 1e-6, 1 - 1e-6) / np.clip(1 - p_block, 1e-6, 1 - 1e-6))
    temp = fit_temperature(logit, y_block)

    (out / "fusion.json").write_text(json.dumps({
        "prior": DEFAULT_PRIOR,
        "temperatures": {"plain_judge": round(temp, 3)},
        "weights": {},  # unit weights: dependence correction not estimable at 813 records
        "dependence_correction": "not estimated (n too small); pool runs at unit weight per Prop. 1",
        "casebank": {"tau": a.tau, "k": a.k, "n_harmful": int(len(harmful)), "n_benign": int(len(benign))},
        "f_default": 0,  # operating point; f=1 reviews most benign with only 2 two-sided agents
        "classes": DECISION,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}/casebank.npz and fusion.json (plain_judge T={temp:.3f})", flush=True)


if __name__ == "__main__":
    main()
