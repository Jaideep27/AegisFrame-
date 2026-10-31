# Changelog

## 0.6.0

- **Local engine (no API needed).** One fine-tuned encoder (MiniLM, 25 MB int8 ONNX), with heads for
  the separator and every council agent:
  - `local_plain_request`, `local_severity`, `local_fraud`, `local_red_team`, `local_advocate`;
  - the local purpose agent.

  It runs with onnxruntime and tokenizers, without PyTorch. The model ships inside the package and
  is used automatically after `pip install aegisframe`. A model installed with
  `aegisframe model install` overrides it (`aegisframe model info`, `local_model` config section).
- **Data engine** (`reframebench.datagen`):
  - 30 record types (8 held out), photo and real-person scenes, 16 intents and 14 twist types;
  - exact spans, a written reason per decision, and train/val/held-out splits;
  - 1,000 rendered images.
- **Training and evaluation:** `training/train_local.py` (CPU) and `training/eval_local.py`
  (end to end, oracle or real OCR).
- **Answer-only requests:** questions about an image ("what is the cost?") are allowed and generate
  no image.
- **Separators receive the document context.** The model and rule separators are merged by
  confidence; rule-found forged values are always kept.
- **New providers:** `claude_cli` (`claude -p`) and `codex_cli` (`codex exec`), for local testing on
  a personal subscription.
- **Fixes:**
  - Relative `local_model.path` is resolved against the config file.
  - Loaded models are shared across Guards.
  - Photo context ignores OCR fragments.
  - Removed the redundant "separators disagreed → review" rule.

## 0.5.0

SDK release on top of the 0.1 research core.

- **Probabilistic agent council.** Agents exchange beliefs (p, confidence, reason code), pooled in
  log-odds. It supports sequential early stopping, a critical-severity veto, a DeGroot / re-prompt
  communication round on disagreement, a fraud-pattern knowledge base, and an agent registry for
  custom agents.
- **Claim-blind deliberation and a one-sided purpose term**, with property tests showing that a story
  never lowers the posterior or the decision.
- **LLM agents:** `plain_request`, `red_team`, `advocate`, and an LLM `purpose` agent, on any
  configured provider. Providers retry with backoff.
- **Configuration:** `AegisConfig` holds every behaviour as a validated attribute, loaded from YAML
  with `AEGIS__*` env overrides. `Guard.from_config()`.
- **Protected calls:** `guard.protect`, `guard.run` and the `GuardedResult` type. Typed exceptions
  `Blocked`, `ReviewRequired`, `OutputRejected` and `GuardUnavailable`. Async `acheck_*`,
  `on_decision` callbacks, and `fail_mode`.
- **Generators:** OpenAI Images (REST), Gemini image (REST), generic HTTP, callables, and duck-typed
  diffusers pipelines.
- **Integrations:** `protect_openai` (edit / generate / create_variation), `protect_gemini`, and
  FastAPI exception handlers.
- **Human review:** a SQLite queue with HMAC-signed, single-use approval tokens bound to the exact
  edit and image, plus a webhook.
- **Persistence:** SQLite session memory and SQLite / JSONL audit sinks. Stated reasons are hashed
  in records by default.
- **Proxy:** `aegisframe serve`, OpenAI-compatible `/v1/images/edits` and `/v1/images/generations`.
- **CLI:** `check`, `verify`, `policy validate`, `config show|validate`, `agents`,
  `review list|approve|reject`, `calibrate`, `serve` and `bench`.
- **Calibration:** conformal risk control and Clopper–Pearson thresholds (`aegisframe calibrate`).
- **ReframeBench:** `--config` for ablations; records council posterior and agents used.
- **Packaging:** OCR is now a core dependency.

## 0.1.0

The research core:
- separator (rules / LLM / agreement), grounding with local OCR, and the topic-free policy;
- floor with reference judges, verified-role unlock and session memory;
- output verification and the audit log;
- ReframeBench v1 (320 cases).
