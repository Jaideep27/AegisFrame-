# Configuration reference

There are three ways to set configuration. Later sources override earlier ones:

1. Defaults (`AegisConfig()`): offline, fail closed, council on.
2. A YAML file: `Guard.from_config("aegis.yaml")`. Paths inside it are relative to the file.
3. Environment variables `AEGIS__SECTION__KEY`, e.g. `AEGIS__COUNCIL__SCHEDULING=all`.

`aegisframe config show` prints every attribute with its default, and `aegisframe config validate`
checks a file. Unknown keys are rejected, so typos fail loudly. The annotated full file is
[examples/aegis.example.yaml](../examples/aegis.example.yaml).

| Section | Key | Default | What it does |
|---|---|---|---|
| (top) | `policy_file` | built-in | topic-free rules, overrides, verified-role unlocks |
| `providers.<name>` | `kind`, `model`, `base_url`, `api_key_env`, `retries` | – | named LLM/VLM endpoints (`gemini`, `openai_compat` for OpenAI/Azure/Groq/NIM/vLLM/Ollama) |
| `separator` | `mode` | `rules` | `rules`, `llm`, or `agreement` (any disagreement takes the stricter reading) |
| | `providers` | `[]` | which providers parse prompts |
| `grounding` | `ocr` | `rapid` | local CPU OCR; `none` turns off record recognition (testing only) |
| `council` | `enabled` | `true` | the probabilistic agent council |
| | `prior` | `0.10` | prior probability of harm |
| | `agents` | severity, fraud_patterns, history | `{name, weight, provider, options}` per agent |
| | `scheduling` | `sequential` | stop once the decision cannot change; `all` asks everyone |
| | `debate.rounds/spread/openness` | `1 / 0.30 / 0.3` | communication round on disagreement |
| | `thresholds.block/review` | `0.85 / 0.50` | posterior cut-offs |
| | `veto` | `0.80` | one agent at/above this on a CRITICAL change blocks |
| | `purpose.enabled/weight/provider` | `true / 1.0 / null` | the one-sided claim audit |
| | `calibration_file` | `null` | thresholds from `aegisframe calibrate` |
| `decision` | `fail_mode` | `closed` | `open` lets requests through on guard errors (not recommended) |
| | `claim_blind` | `true` | `false` is a research ablation: LLM agents read the story |
| `review` | `on_review` | `raise` | `raise` → ReviewRequired, `block` → Blocked, `allow` (testing) |
| | `queue_path` | `null` | SQLite queue for human review |
| | `secret_env` | `AEGIS_REVIEW_SECRET` | HMAC secret for approval tokens |
| | `webhook_url` | `null` | POST per queued review |
| `memory` | `backend`, `path` | `memory` | `sqlite` keeps retry/multi-turn history across processes |
| `audit` | `sink`, `path` | `memory` | `jsonl` or `sqlite` decision records (hashes, never raw prompts or images) |
| `privacy` | `store_claims` | `hash` | stated reasons in audit/review: `hash`, `text`, `none` |
| `verify` | `enabled` | `true` | check generated images against the original |
| `server` | `upstream` | OpenAI | where the proxy forwards |
| | `trust_identity_header` | `false` | honour `X-Aegis-Identity` (only behind your auth gateway) |

## Policy file

Rules are written over attributes, not industries:
- `is_document`
- `is_evidence`
- `real_source`
- `factual_change`
- `passes_as_genuine`
- `real_person_false_act`
- `from_scratch`
- `has_image`
- `edit_understood`

Each rule has `action` or `action_by_severity`, and optionally `escalate_on_reframing`.

`overrides` (matched on `doc_type_contains`) can only tighten a decision. `verified_roles` unlock a
rule up to a severity ceiling. See
[examples/policy.example.yaml](../examples/policy.example.yaml) and validate with
`aegisframe policy validate`.
