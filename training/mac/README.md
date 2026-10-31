# Training the v0.7 pilot on a Mac (Apple Silicon)

1. Copy `aegis-mac.zip` (made on Windows in the project folder) to the Mac and unzip it to `~/aegis`.
2. In Terminal:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh      # skip if uv is already installed
   cd ~/aegis
   uv sync --group train                                  # PyTorch with Apple-GPU support, transformers, etc.
   PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --group train python -m training.train_v2 \
       --device mps --data data/aegis_v2/claude_en --out models/aegis-v07-mac --epochs 6 2>&1 | tee train-mac.log
   ```
   The first run downloads Harrier-OSS-v1-270M (about 1 GB) from Hugging Face; no account is needed.
   `PYTORCH_ENABLE_MPS_FALLBACK=1` runs the few operations the Apple GPU lacks on the CPU instead of failing.
3. Send back `models/aegis-v07-mac/results.json`, the `predictions_*.csv` files and `train-mac.log`.
   The log's "epoch N done in …s" lines give the real Mac speed.

Keep the Mac plugged in. The run prints its device on the first line; it should say `mps`.

# Running the v0.7 data generation on a Mac (Apple Silicon, 16 GB)

The teacher model runs in **Ollama** on the Mac's GPU. The generator talks to it through the
OpenAI-compatible endpoint (`--backend openai`). You don't need any account or API key.

**Model: `qwen3:8b`** (Apache-2.0, about 5 GB at 4-bit). With 16 GB of RAM, macOS lets the GPU use
about 10–11 GB. 14B models (about 9 GB) don't leave enough room for the context, so part of the model
would run on the CPU and be much slower.

## One-time setup
1. **On Windows**, zip the code:
   ```powershell
   Compress-Archive -Path aegisframe, reframebench, training, tests, pyproject.toml, uv.lock, README.md, LICENSE -DestinationPath aegis-code.zip -Force
   ```
   Copy `aegis-code.zip` to the Mac (AirDrop, USB or cloud drive) and unzip it, e.g. to `~/aegis`.
2. **On the Mac**, in Terminal:
   ```bash
   # uv (Python manager) and Ollama
   curl -LsSf https://astral.sh/uv/install.sh | sh
   brew install ollama          # or download the app from ollama.com
   cd ~/aegis && uv sync         # creates .venv with the project dependencies
   ollama pull qwen3:8b
   ```

## Each run
**Terminal 1**: start Ollama with settings for 3 parallel requests and a 6K context per request. If
the Ollama menu-bar app is running, quit it first so these settings take effect.
```bash
OLLAMA_NUM_PARALLEL=3 OLLAMA_CONTEXT_LENGTH=6144 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

**Terminal 2**: run the generator. `caffeinate -i` keeps the Mac from sleeping while it runs.
```bash
cd ~/aegis
# 1) smoke test: 20 groups, about 30-60 min. Check the speed, reject rate and writing quality.
caffeinate -i uv run python -m reframebench.datagen_v2.generate --groups 20 --backend openai \
    --base-url http://localhost:11434/v1 --model qwen3:8b --workers 3 --batch 12 \
    --out data/aegis_v2/mac_smoke --judge

# 2) full run: resumable. Stop it with Ctrl+C at any time and re-run the same command to continue.
caffeinate -i uv run python -m reframebench.datagen_v2.generate --groups 12000 --seed 7 --backend openai \
    --base-url http://localhost:11434/v1 --model qwen3:8b --workers 3 --batch 24 \
    --out data/aegis_v2/full --judge
```
Keep the Mac **plugged in with the lid open**: a MacBook Air sleeps when the lid is closed. It has no
fan, so expect it to slow down somewhat once it is warm.

## After the smoke test
Send back the `data/aegis_v2/mac_smoke/` folder (records.jsonl, rejects.jsonl, raw.jsonl). It has the
three numbers that decide the next step: seconds per group, reject rate by reason, and whether the
requests read like real people.

## Notes
- Requests that fail (Ollama restarted, timeout) are not saved, so re-running the command retries them.
- `--judge` asks the same model to score realism. That costs about 10% more time; drop it for speed.
- Out-of-distribution teacher: later, run a smaller pass (`--groups 1500 --seed 99`) with a model from a
  different family, e.g. IBM Granite (`granite3.3:8b`, Apache-2.0), and list it in
  `finalize(..., ood_teachers=[...])`. Check the licence on the model card first.
