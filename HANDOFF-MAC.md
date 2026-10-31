# Handoff: train the AegisFrame v0.7 pilot on this Mac

You are continuing work that started in a Claude Code session on the user's Windows laptop. The user wants you to do every
step yourself on this Mac (MacBook Air M5, 16 GB) and report results back. Use simple, short explanations with the user.

## Project in one paragraph
AegisFrame is a guardrail for AI image editors: it decides whether a request would produce a falsified record (payslip,
marksheet, ID, bank statement…) that could pass as genuine. v0.7 fine-tunes the embedding model **Harrier-OSS-v1-270M**
(`microsoft/harrier-oss-v1-270m`, MIT, decoder-only, last-token pooling) with classifier heads. Training data:
`data/aegis_v2/claude_en/dataset.jsonl`, 510 English messages (BLOCK 242 / ALLOW 232 / REVIEW 36),
splits train 327 / val 67 / test_id 34 / test_ood 82.

## What to do
1. You are in the project folder (it contains `pyproject.toml`, `training/`, `data/`). The first `uv sync` failed because
   uv picked Python 3.15 and torch 2.14.1 has no cp315 wheels. Fix and install:
   ```bash
   rm -rf .venv
   uv python pin 3.13
   uv sync --group train
   ```
2. Run the training (the trainer is `training/train_v2.py`; `--device mps` uses the Apple GPU):
   ```bash
   PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --group train python -m training.train_v2 --device mps \
       --data data/aegis_v2/claude_en --out models/aegis-v07-mac --epochs 6 2>&1 | tee train-mac.log
   ```
   The first line must say `device: mps`. It downloads Harrier (~1 GB) the first time; no Hugging Face login needed.
   The Mac should stay plugged in with the lid open.
3. If it errors on MPS, fix it in `training/train_v2.py` (keep it working on CPU too), and say what you changed.
   The MPS path has never been run before; on Windows CPU it is tested.
4. When done, report to the user in a short table: time per epoch, total time, and the final `val` / `test_id` / `test_ood`
   numbers from `models/aegis-v07-mac/results.json` next to the laptop numbers below. Report only measured numbers, and say
   these are synthetic test sets written by Claude, not human-written prompts.

## How the trainer works (so you can debug it)
- Two passes per message through one encoder: **full** (record OCR context + earlier turns + raw message) and **plain**
  (context + message with STORY spans removed). Heads: plain → decision, fact_class; full → sentinel decision, intent,
  purpose, distinguishable. Final decision = plain decision, raised (never lowered) by the sentinel.
- Only the top 6 of 18 layers are trained. The frozen bottom 12 layers are run **once** per sequence and cached
  (`PilotModel.bottom`), then training runs only the top layers (`run_layers` copies Gemma3TextModel's own layer loop).
  The cache must stay **fp32**: mid-layer activations reach ~124,000, beyond the fp16 range (fp16 gave NaN).
  The split forward was verified equal to the normal forward (max abs diff 0.0) on Windows.
- Loss: cross-entropy over heads (class-weighted decision) + supervised contrastive term; twin groups share a batch.
- Outputs: `checkpoint.pt`, `results.json` (incl. a frozen-Harrier + logistic-regression baseline), `predictions_*.csv`.

## Laptop results to compare against (Windows CPU, same data, same settings)
| split | n | frozen baseline acc | fine-tuned acc | harmful allowed | benign blocked |
|---|---|---|---|---|---|
| val | 67 | 0.657 | 0.910 | 0/31 | 2/31 |
| test_id | 34 | 0.588 | 0.882 | 1/15 | 1/15 |
| test_ood | 82 | 0.671 | 0.866 | 2/38 | 0/36 |

Laptop speed: ~264–289 s per epoch, ~200 s to build the cache, ~35 min total. Known weaknesses: REVIEW is almost never
predicted (19 training examples); misses were "remove the watermark" requests and an edit hidden inside a translation request.

## Rules from the user
- Only fictional people and records. Don't publish anything or commit/push unless asked.
- The user runs `uv publish` themselves; never use tokens pasted in chat.
