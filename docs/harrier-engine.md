# The v0.7 prompt-level engine

AegisFrame 0.7 judges a request to edit a document from the **prompt** plus the OCR-style **record text**,
rather than from the image pipeline. The 0.6 engine is unchanged and still handles the image API; the v0.7
engine is a separate entry point, `HarrierGuard`.

## Using it

```python
from aegisframe import HarrierGuard

guard = HarrierGuard.load("models/aegis-v07-r1-onnx")   # or a cached/installed model
v = guard.check("change the net pay to 90,000",
                context="Record:\nPayslip\nNet Pay: 42,700\n")
v.decision        # "BLOCK"
v.reason          # why, from the agents that fired
v.trace           # one row per agent: name, LLR message, whether it fired, a note
```

`check(prompt, context="", history=None)` returns a `HarrierVerdict`: `decision`
(`ALLOW` / `REVIEW` / `BLOCK`), the posterior `p`, `p_block`, the read `intent`, a `reason`, the detected
`value_changes`, and the full `trace`.

## How a decision is made

1. **The model** reads the request twice through one encoder: a story-blind pass (the decision) and a
   raw-text pass (a sentinel that may only raise it).
2. **The agents** each read one view and write a calibrated log-likelihood-ratio message:
   `plain_judge` and `casebank` are two-sided; `sentinel`, `intent_criticality`, `value_diff`,
   `output_authenticity`, `purpose_auditor` and `history` can only raise risk.
3. **The pool** fuses the messages and decides by minimum Bayes risk. Two properties hold by construction:
   a story can never lower risk, and a detected marking removal can never end in ALLOW.

The fault-tolerance knob `f` (default 0) controls how many agents' allow-ward votes must be removable
before an ALLOW stands. At the current data scale `f=1` reviews most benign requests, so it ships off.

## Building the model artifacts

```bash
uv run python -m training.export_v2  --model models/aegis-v07-r1 --out models/aegis-v07-r1-onnx --max-len 128
uv run python -m training.train_fusion --model models/aegis-v07-r1-onnx --data data/aegis_v2/en_all
uv run python -m training.eval_v2       --model models/aegis-v07-r1-onnx --data data/aegis_v2/en_all
```

`export_v2` writes the ONNX model and tokenizer; `train_fusion` writes the `fusion/` directory (case bank
and calibration); `eval_v2` writes `results/eval_v2.json`.

## MCP server

```bash
pip install 'aegisframe[mcp]'
aegisframe mcp                 # stdio, for Claude Desktop / Claude Code / Codex
aegisframe mcp --http          # streamable HTTP for remote use
```

Tools: `check_request(prompt, context?, session_id?, request_id?)`, `verify_output`,
`explain_decision(request_id)`. An agent should call `check_request` before any document image edit and
refuse on `BLOCK`.

## What is measured, and what is not

All reported numbers are on Claude-written synthetic test sets (`results/eval_v2.json`). No human-written
test set exists yet, so no real-world accuracy is claimed. A bare question that names a field can still be
over-blocked by the model; the council cannot override a confident wrong head.

## Judging an image directly

`check_image(image_bytes, prompt)` OCRs the record with RapidOCR, rebuilds its layout (header,
key-values, table rows) and renders it in the same `[DOC] … [TABLE] …` form the model was trained on,
then judges the prompt against it. A forged table cell is seen in context, not lost in a flat blob.

Any OCR engine implementing the `OcrEngine` protocol can be passed as `ocr=` — this is the extension
point for the accurate (LightOnOCR) and native (PaddleOCR/PP-StructureV3) tiers; install those yourself
and pass an adapter. The default RapidOCR tier ships with the package.

The MCP `check_request` tool also accepts a base64 `image_b64` and routes it through `check_image`.
