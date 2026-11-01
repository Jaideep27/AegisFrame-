"""Export a trained v0.7 checkpoint to an ONNX model directory the shipped package can load.

The package runs on onnxruntime + tokenizers, with no PyTorch, so a torch checkpoint is not shippable.
This writes the directory layout aegisframe expects: model.onnx, model.int8.onnx, tokenizer.json, config.json.

The graph is ONE pass: (input_ids, attention_mask) -> the six head logits. The runtime calls it twice,
taking the full-pass heads from the raw message and the plain-pass heads from the story-stripped message,
which is exactly what training did with two passes through one encoder.

    uv run --group train python -m training.export_v2 --model models/aegis-v07-r2 \
        --out models/aegis-v07-r2-onnx --max-len 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer

from aegisframe.local.harrier import HEADS, TASK, FileTokenizer, HFTokenizer, build_ids

from . import train_v2 as T


class ExportWrapper(nn.Module):
    """Backbone + last-token pooling + every head, in one exportable forward.

    Training pooled through a split layer loop with a cached frozen bottom; this uses the encoder's own
    forward. They compute the same thing, and tests/test_export_v2.py checks that they agree to 1e-3.
    """

    def __init__(self, model: T.PilotModel):
        super().__init__()
        self.encoder = model.encoder
        self.full_heads = model.full_heads
        self.plain_heads = model.plain_heads
        self.proj = model.proj

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        out = {k: head(pooled) for k, head in self.full_heads.items()}
        out.update({k: head(pooled) for k, head in self.plain_heads.items()})
        embedding = nn.functional.normalize(self.proj(pooled), dim=-1)
        return (*(out[h] for h in HEADS), embedding)


def load_model(model_dir: str | Path, train_layers: int = 6) -> tuple[T.PilotModel, object]:
    T.DEV = torch.device("cpu")
    tok = AutoTokenizer.from_pretrained(T.ENCODER)
    model = T.PilotModel(train_layers).to(T.DEV).eval()
    state = torch.load(Path(model_dir) / "checkpoint.pt", map_location=T.DEV)
    model.load_state_dict(state, strict=False)
    return model, tok


def encode(tokenizer, context: str, message: str, max_len: int, bos: int, eos: int) -> np.ndarray:
    return np.array([build_ids(tokenizer, context, message, max_len, bos, eos)], dtype=np.int64)


def torch_logits(model_dir: str | Path, samples: list[tuple[str, str]], max_len: int) -> dict[str, np.ndarray]:
    """The training-time computation: cached frozen bottom, split top, final norm, last-token pool."""
    model, tok = load_model(model_dir)
    wrapped = HFTokenizer(tok)
    rows: dict[str, list[np.ndarray]] = {h: [] for h in HEADS}
    with torch.no_grad():
        for context, message in samples:
            ids = build_ids(wrapped, context, message, max_len, tok.bos_token_id, tok.eos_token_id)
            pooled = model.pool([model.bottom(ids)])
            out = {k: head(pooled) for k, head in model.full_heads.items()}
            out.update({k: head(pooled) for k, head in model.plain_heads.items()})
            for h in HEADS:
                rows[h].append(out[h][0].cpu().numpy())
    return {h: np.stack(v) for h, v in rows.items()}


def torch_embeddings(model_dir: str | Path, samples: list[tuple[str, str]], max_len: int) -> np.ndarray:
    """The training-time contrastive embedding: normalize(proj(pooled)), as supcon saw it."""
    model, tok = load_model(model_dir)
    wrapped = HFTokenizer(tok)
    rows = []
    with torch.no_grad():
        for context, message in samples:
            ids = build_ids(wrapped, context, message, max_len, tok.bos_token_id, tok.eos_token_id)
            pooled = model.pool([model.bottom(ids)])
            rows.append(nn.functional.normalize(model.proj(pooled), dim=-1)[0].cpu().numpy())
    return np.stack(rows)


def session_logits(session, model_dir: str | Path, samples: list[tuple[str, str]],
                   max_len: int) -> dict[str, np.ndarray]:
    """What the shipped runtime will compute: the exported graph plus the exported tokenizer.json."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    wrapped = FileTokenizer(Path(model_dir) / "tokenizer.json")
    names = [o.name for o in session.get_outputs()]
    rows: dict[str, list[np.ndarray]] = {h: [] for h in HEADS}
    for context, message in samples:
        ids = encode(wrapped, context, message, max_len, cfg["bos_token_id"], cfg["eos_token_id"])
        got = dict(zip(names, session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})))
        for h in HEADS:
            rows[h].append(got[h][0])
    return {h: np.stack(v) for h, v in rows.items()}


def export(model_dir: str | Path, out: str | Path, max_len: int = 128, quantize: bool = True,
           train_layers: int = 6) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(model_dir, train_layers)
    wrapper = ExportWrapper(model).eval()

    sample = encode(HFTokenizer(tok), "Record:\nPayslip\nNet Pay: 42,000\n",
                    "Message: change Net Pay to 92,000", max_len, tok.bos_token_id, tok.eos_token_id)
    ids = torch.from_numpy(sample)
    mask = torch.ones_like(ids)
    torch.onnx.export(
        wrapper, (ids, mask), str(out / "model.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=[*HEADS, "embedding"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      **{h: {0: "batch"} for h in (*HEADS, "embedding")}},
        opset_version=18, dynamo=False,
    )
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(out / "model.onnx"), str(out / "model.int8.onnx"), weight_type=QuantType.QInt8)
    tok.save_pretrained(str(out))

    vocab = {**T.FULL_HEADS, **T.PLAIN_HEADS}
    (out / "config.json").write_text(json.dumps({
        "format": 2,
        "encoder": T.ENCODER,
        "task": TASK,
        "max_len": max_len,
        "heads": list(HEADS),
        "embedding_dim": int(model.proj.out_features),
        "labels": {h: list(vocab[h]) for h in HEADS},
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "source_checkpoint": str(Path(model_dir) / "checkpoint.pt"),
        "version": Path(model_dir).name,
    }, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/aegis-v07-r2")
    ap.add_argument("--out", default="models/aegis-v07-r2-onnx")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--train-layers", type=int, default=6)
    ap.add_argument("--no-quantize", action="store_true")
    a = ap.parse_args()
    out = export(a.model, a.out, a.max_len, not a.no_quantize, a.train_layers)
    for f in sorted(out.iterdir()):
        print(f"  {f.name:<24} {f.stat().st_size / 1e6:>8.1f} MB")


if __name__ == "__main__":
    main()
