# Phase 1 spike: choosing the embedding model (2026-10-30)

Candidates: **Harrier-OSS-v1-270M** (Microsoft, MIT) and **EmbeddingGemma 2, text-only** (Google, Apache-2.0), with the
v0.6 **MiniLM-L6** as the baseline. These are frozen models; nothing is fine-tuned yet. The machine is an Intel Core Ultra 7 155H
throttled to about 1.4 GHz, using 4 threads.

## 1. Frozen embeddings + linear probe on minimal-edit twins

`training/spike/pilot_twins.jsonl` holds 30 hand-written pairs: the same document and almost the same words, one harmful and one legitimate.
They cover 12 languages, long prompts, negated markings, prompt-only requests, role-play and "correcting a printing error".
The probe is scored leave-one-pair-out.
This is a quick signal from a tiny set: **these numbers are never reported as results**.

| Model | Params | Dim | Probe accuracy | Twin accuracy (harmful twin scored higher) |
|---|---|---|---|---|
| **Harrier-270M** | 268M | 640 | 0.733 | **0.967** (29/30) |
| EmbeddingGemma 2 (text) | 271M | 768 | 0.733 | 0.833 (25/30) |
| MiniLM-L6 (v0.6) | 23M | 384 | 0.717 | 0.767 (23/30) |

## 2. CPU latency per request (PyTorch fp32, median of 5)

| Model | 64 tokens | 256 tokens | 1,024 tokens |
|---|---|---|---|
| Harrier-270M | 214 ms | 585 ms | 2,574 ms |
| EmbeddingGemma 2 | 271 ms | 1,102 ms | 4,267 ms |
| MiniLM-L6 | 19 ms | 124 ms | n/a (256-token limit) |

## 3. Harrier as ONNX (`training/spike/onnx_spike.py`)

Export: backbone + last-token pooling + L2 norm, opset 18. In fp32 the export matches sentence-transformers to within 1.5e-8.

| Variant | Size | Min cosine to fp32 | 64 / 256 / 1,024 tokens |
|---|---|---|---|
| fp32 | 1,073 MB | 1.0000 | 79 / 360 / 2,375 ms |
| int8 dynamic (default ops) | 270 MB | 0.979 | 51 / 242 / 1,369 ms |
| int8 per-channel, MatMul only | 774 MB | 0.983 | 136 ms @256 |
| int8 per-channel, MatMul + Gather | 270 MB | 0.978 | 169 ms @256 |

Most of the size is the 262K-token vocabulary: 262,144 × 640 ≈ 168M of the 268M parameters.

## Architecture facts confirmed from the config
- **Harrier-270M** is a `Gemma3TextModel`: 18 layers, hidden size 640, 4 heads and 1 KV head, full (non-sliding) attention.
  `use_bidirectional_attention: false`, so it's causal. The flag exists, so bidirectional attention can be switched on
  for the word-tagging head, the way EmbeddingGemma converts Gemma.
- **EmbeddingGemma 2** text-only loads as 271M parameters. It needs `torchvision` installed even for text-only use.

## Decision
- **Harrier-270M becomes the primary model.** It separates the twins best (0.967 vs 0.833), is about 2× faster, has 4× the context and an MIT licence.
  EmbeddingGemma 2 stays as the second candidate in the fine-tuned bake-off, which is cheap on a GPU.
- **Deployment format:** int8 ONNX (270 MB), **provided the fine-tuned heads keep their accuracy after quantisation**.
  This is checked on the validation set; if they don't, ship fp16.
- **Tagging:** compare causal vs `use_bidirectional_attention: true` for the BIO tagging head during fine-tuning.
