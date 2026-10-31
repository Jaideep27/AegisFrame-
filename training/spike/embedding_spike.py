"""Phase 1 spike: Harrier-OSS-v1-270M vs EmbeddingGemma 2 (text) vs the v0.6 MiniLM, before any fine-tuning.

    uv run --group train python training/spike/embedding_spike.py

1. Frozen embeddings + a linear probe on hand-written minimal-edit twins (training/spike/pilot_twins.jsonl),
   scored leave-one-pair-out: probe accuracy, and twin accuracy (the harmful twin gets the higher score).
2. CPU latency per request at 64 / 256 / 1,024 tokens.

The pilot set is a quick signal only; it is never used for training or for reported results.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
TASK = "Decide whether this image-edit request would produce a falsified record that passes as genuine"

MODELS = {
    "harrier-270m": dict(name="microsoft/harrier-oss-v1-270m", kw={}, encode=dict(prompt=f"Instruct: {TASK}\nQuery: ")),
    "embeddinggemma-2": dict(name="google/embeddinggemma-2",
                             kw={"config_kwargs": {"vision_config": None, "audio_config": None}},
                             encode=dict(prompt_name="Classification")),
    "minilm-l6 (v0.6)": dict(name="sentence-transformers/all-MiniLM-L6-v2", kw={}, encode={}),
}


def load_pilot() -> list[dict]:
    return [json.loads(x) for x in (HERE / "pilot_twins.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def as_text(r: dict) -> str:
    return f"Context: {r['context']}\nRequest: {r['prompt']}"


def probe(emb: np.ndarray, rows: list[dict]) -> tuple[float, float]:
    """Leave-one-pair-out logistic regression: (example accuracy, twin accuracy)."""
    labels = np.array([r["label"] for r in rows])
    pairs = np.array([r["pair"] for r in rows])
    scores = np.zeros(len(rows))
    for p in sorted(set(pairs)):
        test = pairs == p
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(emb[~test], labels[~test])
        scores[test] = clf.predict_proba(emb[test])[:, 1]
    acc = float(((scores >= 0.5) == labels).mean())
    twin = [scores[(pairs == p) & (labels == 1)][0] > scores[(pairs == p) & (labels == 0)][0] for p in sorted(set(pairs))]
    return acc, float(np.mean(twin))


def latency(model: SentenceTransformer, encode_kw: dict, n_tokens: int, reps: int = 5) -> float:
    text = " ".join(["change the net pay on this payslip to ninety five thousand"] * (n_tokens // 10 + 1))
    ids = model.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    text = model.tokenizer.decode(ids)
    model.encode([text], **encode_kw)  # warm-up
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        model.encode([text], **encode_kw)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main() -> None:
    torch.set_num_threads(4)
    rows = load_pilot()
    print(f"pilot: {len(rows)} examples, {len({r['pair'] for r in rows})} twin pairs\n")
    print(f"{'model':20} {'params':>7} {'dim':>5} {'probe acc':>10} {'twin acc':>9} {'ms@64':>7} {'ms@256':>7} {'ms@1024':>8}")
    for label, spec in MODELS.items():
        m = SentenceTransformer(spec["name"], model_kwargs={"dtype": torch.float32}, **spec["kw"])
        m.max_seq_length = min(m.max_seq_length or 512, 2048)
        emb = m.encode([as_text(r) for r in rows], normalize_embeddings=True, batch_size=8, **spec["encode"])
        acc, twin = probe(np.asarray(emb), rows)
        lat = [latency(m, spec["encode"], n) if n <= m.max_seq_length else float("nan") for n in (64, 256, 1024)]
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{label:20} {params:6.0f}M {emb.shape[1]:5d} {acc:10.3f} {twin:9.3f} "
              f"{lat[0]:7.0f} {lat[1]:7.0f} {lat[2]:8.0f}")


if __name__ == "__main__":
    main()
