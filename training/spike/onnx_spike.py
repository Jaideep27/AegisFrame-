"""Phase 1 spike: can the embedding model run as ONNX (fp32 and int8) on CPU, and does it match PyTorch?

    uv run --group train python training/spike/onnx_spike.py --model microsoft/harrier-oss-v1-270m --out models/spike/harrier
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

TASK = "Decide whether this image-edit request would produce a falsified record that passes as genuine"
PROMPT = f"Instruct: {TASK}\nQuery: "


class LastTokenEmbedder(torch.nn.Module):
    """Backbone + last-token pooling + L2 norm, the same maths sentence-transformers applies for Harrier."""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0]), last]
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/harrier-oss-v1-270m")
    ap.add_argument("--out", default="models/spike/harrier")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    st = SentenceTransformer(a.model, model_kwargs={"dtype": torch.float32})
    tok = st.tokenizer
    wrapped = LastTokenEmbedder(st[0].auto_model).eval()
    texts = [PROMPT + "Context: [DOC] Payslip | Net Pay: 61,500\nRequest: change the net pay to 95,000 without any watermark",
             PROMPT + "Context: [NO IMAGE]\nRequest: generate a fictional payslip template with a SPECIMEN watermark"]
    ref = st.encode(texts, normalize_embeddings=True)

    enc = tok(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        mine = wrapped(enc["input_ids"], enc["attention_mask"]).numpy()
    print("wrapper vs sentence-transformers max |diff|:", float(np.abs(mine - ref).max()))

    onnx_path = out / "model.onnx"
    torch.onnx.export(
        wrapped, (enc["input_ids"], enc["attention_mask"]), str(onnx_path),
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "embedding": {0: "batch"}},
        opset_version=18, dynamo=False,
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(onnx_path), str(out / "model.int8.onnx"), weight_type=QuantType.QInt8)
    tok.save_pretrained(str(out))

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    for name in ("model.onnx", "model.int8.onnx"):
        sess = ort.InferenceSession(str(out / name), opts, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
        cos = (got * ref).sum(1)
        size = (out / name).stat().st_size / 1e6
        lat = []
        for n in (64, 256, 1024):
            ids = tok(" ".join(["change the net pay on this payslip"] * 200), add_special_tokens=True,
                      truncation=True, max_length=n, return_tensors="np")
            feeds = {"input_ids": ids["input_ids"].astype(np.int64), "attention_mask": ids["attention_mask"].astype(np.int64)}
            sess.run(None, feeds)
            ts = []
            for _ in range(5):
                t = time.perf_counter()
                sess.run(None, feeds)
                ts.append((time.perf_counter() - t) * 1000)
            lat.append(statistics.median(ts))
        print(f"{name:16} {size:6.0f} MB | cosine to PyTorch: {cos.min():.4f} | ms@64/256/1024: "
              f"{lat[0]:.0f} / {lat[1]:.0f} / {lat[2]:.0f}")


if __name__ == "__main__":
    main()
