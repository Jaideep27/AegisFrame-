# The local model (no API needed)

AegisFrame can run on its own small model, trained on our own data. No API key is needed, nothing
leaves the machine, and the behaviour is the same whatever LLM a user does or doesn't have.

| | Local model (default when installed) | API LLM (optional) |
|---|---|---|
| Cost per request | free | provider pricing |
| Latency | tens of milliseconds on a laptop CPU | seconds |
| Privacy | nothing leaves the machine | prompt (and image) go to the provider |
| Same behaviour for every user | yes | depends on the user's model |
| New, never-seen tricks | weaker | stronger |

The two combine. The local model is the default engine. LLM agents can be added to the council for
cases the local model is unsure about, and early stopping means they are only called when needed.

## One encoder, every agent

A single fine-tuned encoder (`sentence-transformers/all-MiniLM-L6-v2`, 23M parameters, about 25 MB
after int8 quantisation) has one head per job. It runs **two passes** per request:

| Pass | Input | Head | Plays the role of |
|---|---|---|---|
| full | document context + the user's whole message | `tags` (BIO: FIELD, OLD, NEW, MARK, OBJ, STORY) + `intent` | **separator** |
| | | `twist` (purpose, authority, urgency, hypothetical, roleplay, Hinglish, multi-turn…) | story detection |
| | | `purpose` | **purpose** agent (reads the story; applied last, one-sided) |
| plain | document context + the canonical instruction (story removed) | `decision` | **plain_request** agent |
| | | `field_class` | **severity** agent |
| | | `pattern` | **fraud_patterns** agent |
| | | `misuse` | **red_team** agent |
| | | `benign_need` | **advocate** agent |

The **history** agent stays session logic (retries, edits built up step by step), because it
reasons over past turns rather than text.

Claim-blindness is preserved by construction. The plain pass never sees the story, so the five
deliberating heads are as claim-blind as the rules. The model's separator runs in **agreement**
with the rule separator, so any disagreement resolves to the stricter reading. The rules, the
output check and the "a story can't unlock it" guarantee do not depend on the model at all.

## Training data

Generate the corpus:

```bash
uv run python -m reframebench.datagen.export --out data/aegis_corpus
```

It produces:

- **24,000 training + 2,000 validation examples** over 22 record types (payment screenshots,
  payslips, marksheets, lab reports, tickets, bills, certificates…), photos, and real-person scenes.
- **3,000 held-out examples** from:
  - 8 record types never used in training (fee receipt, vaccination certificate, driving licence,
    property tax, court notice, loan sanction, internship certificate, fuel bill);
  - held-out phrasing templates;
  - two twist types never seen in training (role-play and "I'm just correcting a mistake").
- **966 rendered images**, for end-to-end tests with real OCR.

Each example has:

- the document context (in exactly the format the SDK produces at inference);
- the prompt;
- character spans;
- the canonical instruction (built by the SDK's own function);
- labels for every head;
- a written **reason** for the decision.

The labels follow a written guideline (see `reframebench/datagen/builder.py`). They are **never**
produced by AegisFrame itself, so the evaluation isn't circular.

**Intents covered:**

- forging a fact (replace, remove, signature, removing a VOID/SPECIMEN mark);
- marked samples, with a tiny corner mark as a near-miss;
- fictional from scratch;
- cosmetic edits, redaction and annotation;
- questions (no image generated) and vague record edits;
- photo edits;
- real-person edits: neutral, demeaning, a fabricated act, or labelled satire.

**Twists covered:**

- stories: purpose, dispute, legal, authority, urgency, emotional, hypothetical, ownership,
  role-play\*, correction claim\* (\* held out);
- indirect phrasing;
- typos and lower-casing;
- Hinglish;
- edits split across turns.

## Training (CPU is fine)

```bash
uv sync --group train
uv run --group train python training/train_local.py --data data/aegis_corpus --out models/aegis-local-v3
uv run aegisframe model install models/aegis-local-v3          # -> ~/.cache/aegisframe/models/aegis-local
uv run aegisframe model info
```

Training uses PyTorch and Transformers (the `train` dependency group). The output is an ONNX graph
plus a `tokenizer.json`. Users of `pip install aegisframe` need only `onnxruntime` and `tokenizers`,
not PyTorch.

The released model ships inside the package (`aegisframe/local/model/`) and is used automatically.
A model installed with `aegisframe model install` overrides it, and so does `local_model.path` or
`AEGIS_LOCAL_MODEL`. To ship a retrained model, copy its `config.json`, `tokenizer.json` and
`model.int8.onnx` into `aegisframe/local/model/` before building.

On Windows, background training can get parked on efficiency cores. Run it from a visible terminal
or raise the process priority.

## Using it

```python
guard = Guard()                       # local_model.enabled = auto: used if installed
guard = Guard.from_config("aegis.yaml")
```

```yaml
local_model: {enabled: auto, path: null, quantized: true, threads: null}   # on | off | auto
```

With a local model installed and the default council, the council becomes:

- severity, fraud_patterns and history (the heuristic agents);
- local_plain_request, local_fraud, local_severity, local_red_team and local_advocate;
- the local purpose agent.

LLM agents can still be added in `council.agents`.

## Evaluation

```bash
uv run python training/eval_local.py --model models/aegis-local-v3 --split test_heldout    # oracle OCR
uv run python training/eval_local.py --model models/aegis-local-v3 --images 300            # real OCR
```

Every held-out example goes through the whole Guard. The script compares rules only against rules +
local model, and breaks accuracy down by intent and by twist type.

## Results: three rounds of train, evaluate, hunt failures

End to end through the Guard, on the fresh held-out split of each round's corpus (1,200 examples):

| Round | What changed | Accuracy | Harmful allowed | Harmless blocked | Harmless halted |
|---|---|---|---|---|---|
| rules only | n/a | 53.1% | 41.8% | 6.0% | 22.3% |
| 0 (v1) | 24k examples, 4–6 phrasings per intent | ~79% | 2.7% | 15.5% | 23.7% |
| 1 (v2) | 3× more phrasings, contrasting pairs, rewriting operators; integration fixes | 91.4% | 2.5% | 5.8% | 8.7% |
| 2 (v3) | stories using edit-sounding words, more questions and multi-turn variants; continued from v2 | **92.8%** | **1.0%** | **3.9%** | **6.6%** |

Notes on the table:
- Round 0 was measured on the older held-out set, so treat it as indicative only.
- In round 2, the twist types never seen in training scored 93.2% (role-play) and 95.4%
  (correction claim).
- With real OCR on 300 rendered images, v3 scores 91.0% / 1.9% / 4.3% / 7.9%.

What each round taught us:
- **Round 0:** the decision head was already right, given a correct parse. The intent and parse
  heads were the weak point: with few phrasings they latched onto surface words ("watermark" meant
  removal; "crop" meant a marked sample).
- **Round 0 integration:**
  - Merging with the rule separator on every request let its misreadings add phantom facts.
    Merging is now confidence-gated, and any forged *value* the rules find is still added.
  - The five model heads were over-counted as independent votes. The decision head now leads.
- **Round 1:**
  - Unseen story wording (role-play mentioning "image editor") steered the intent.
  - The labelling guideline was made consistent with the system: a story escalates any review
    except vague edits.
- **Real OCR:** photos produce OCR fragments (a face read as "O"), so photo context now keeps only
  real words.

Files: `results/local_eval_*.json`, training logs `models_train_v*.log`.

## Honest limits

- **Same author:** the corpus and the held-out set come from the same generator author. The
  held-out split uses unseen record types, templates and twist types, but a set written
  independently by other people, plus LLM-generated paraphrases, is still needed before claiming
  general accuracy.
- **English-centric encoder:** romanised Hinglish works because it is in the training data. Other
  scripts (Devanagari, Tamil) need a multilingual encoder (mDeBERTa, MuRIL) and data, best trained
  on a GPU.
- **People in images:** the model can't tell whether a photo shows a real public figure. It relies
  on the request naming one ("the minister"). A small vision classifier is the next step.
