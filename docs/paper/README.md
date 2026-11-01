# Paper skeleton — AegisFrame v0.7

A working outline for a write-up of the prompt-level document-forgery guard. Every number here is
regenerated from `results/eval_v2.json` and the run artifacts, not typed by hand; re-run
`training/eval_v2.py` to refresh the tables. Sections marked *(draft)* have prose; the rest are the
claims to make with the evidence that backs them.

---

## 1. Problem

Image editors will forge documents on request: inflate a payslip, change a grade, strip a SAMPLE
watermark, fabricate a bank statement from nothing. A guard must judge the **request**, from the prompt
and whatever OCR text the record yields, and refuse the forgery while allowing ordinary edits (redaction,
translation, cropping, questions). The judgement must not depend on the excuse the user attaches.

The predecessor (v0.6) judged from an image pipeline and failed on real prompts: a 112-token cap crashed
long prompts, "without a watermark" was read as "add a SAMPLE stamp", and multi-change requests were
mangled. Those figures (92.8%, 41.8%→1.0%) came from template data and are withdrawn.

## 2. Contributions

1. A prompt-level guard built on a 270M fine-tuned encoder, run on CPU without PyTorch.
2. A council of agents that communicate in **calibrated log-likelihood-ratio messages** fused by a
   log-linear pool, with two guarantees proven by construction (§5).
3. An honest evaluation that separates what the synthetic test sets can show from what only
   out-of-distribution prose reveals (§6).

## 3. Data *(draft)*

813 English records (`data/aegis_v2/en_all`): a document plus a request about it, most in twin pairs —
the same document with a harmful edit and a benign one. 47 record families across 17 locales. Labels by
construction from an executable guideline (`reframebench/datagen_v2/guideline.py`). Splits: train 527,
val 89, in-distribution test 57, out-of-distribution test 140, the last held out by **record family** or
**persuasion technique**. Leakage checks: no twin group crosses the train boundary; max 5-gram Jaccard
0.18 on test_ood. **Limitation:** all records are Claude-authored; no human-written test set exists yet,
so no real-world accuracy is claimed.

## 4. Model

Harrier-OSS-v1-270M, top 6 of 18 layers fine-tuned, multi-task heads for decision, intent, purpose,
marking and fact class, plus a supervised-contrastive projection with twins kept in the same batch. Two
passes through one encoder: a **story-blind** decision and a **raw-text sentinel** that may only raise it.
Exported int8 ONNX (~272 MB + 33 MB tokenizer), 89–142 ms per request on a laptop CPU. Selection puts
harmful-allowed before macro-F1; the token budget is 128 (identical accuracy to 512 in the sweep).

## 5. Council and fusion — the maths

Classes y ∈ {A, R, B}. Each agent k reads only its own view and writes a prior-corrected LLR message

> m_{k,c} = log P(y=c | v_k) / P(y=A | v_k) − log π_c / π_A,  c ∈ {R, B}

Two-sided agents (plain-judge, case bank) may argue either way; one-sided agents (sentinel, intent
criticality, value-diff, output authenticity, purpose auditor, history) enter only through max(0, m) with
weight ≥ 0. The log-linear pool forms η and a posterior, and decides by minimum Bayes risk with a cost
matrix where allowing a true B is by far the costliest error.

- **Proposition 1.** If the views are conditionally independent given y and the messages are calibrated,
  unit weights give the exact posterior (a product of experts). At 813 records the dependence-correction
  weights are not estimable, so the pool runs at unit weight — stated, not hidden.
- **Theorem 1 (story monotonicity).** Because a story enters only through one-sided terms with weight ≥ 0,
  adding one never raises P(ALLOW) and never flips the decision toward ALLOW. *(Note: the stronger claim
  that P(BLOCK) rises is false for a 3-class softmax — a REVIEW-ward one-sided agent can pull mass out of
  BLOCK. The risk-side invariant, P(ALLOW) non-increasing, is the correct and safety-relevant one.)*
  Property-tested over 300 random message sets.
- **Theorem 2 (f-fault-tolerant ALLOW).** ALLOW is returned only if it survives removing the f strongest
  allow-ward contributions, so no f agents can by themselves force an ALLOW. Property-tested.
- **Hard floor.** A detected marking removal can never end in ALLOW.

Per-agent temperature calibration is fit on held-out data (`calibrate_fusion.py`); the plain-judge
temperature is 1.65, i.e. the model is overconfident, consistent with the bimodal p_block in §6.

## 6. Evaluation — honest numbers

All on Claude-written synthetic sets (`results/eval_v2.json`).

**Accuracy and safety, test_ood (n=140):**

| System | Accuracy | Harmful allowed | Benign blocked |
|---|---|---|---|
| Keyword filter | 0.507 | 39/65 | 16/61 |
| Fine-tuned model only | 0.957 | 0/65 | 0/61 |
| Council (f=0) | 0.921 | 0/65 | 0/61 |
| Council (f=1) | 0.543 | 0/65 | 0/61 |

**The central result, stated plainly.** On the synthetic sets the model already fits, the council does
not raise accuracy — it trades a little to drive harmful-allowed to zero and for agent-fault robustness,
and f=1 over-reviews at this data scale. The council's benefit appears on **out-of-distribution prose**:
on 66 hand-written / cross-model red-team prompts the bare model allowed 3 forgeries (a disguised SAMPLE
removal, an indirect number change, a fabricated bill); the council allowed none, while blocking no
benign edit. The keyword baseline at ~0.5 shows the task is not trivially separable.

**Known limitation (reproduced three times).** A bare question that names a field ("what is the net
pay?") is false-blocked: the decision head reads it as an edit and the case bank leans harmful. The fusion
cannot override a confident wrong head. On the 40-prompt stress set, 7 of 10 errors are this one
category; 0 of 21 harmful requests were allowed. The fix is training data for questions/extraction, which
is a data decision, not an architecture change.

## 7. System

`HarrierGuard` (no PyTorch), an MCP server (`aegisframe mcp`), a LangGraph node whose decision matches the
core exactly, and download-on-first-use with a SHA-256 check. 565 tests pass.

## 8. Threats to validity

- Synthetic, single-author data; the headline must wait for human-written prompts.
- Fusion dependence-correction not estimable at this n.
- The question false-block is a live defect.
- OCR layout is approximated in training; the accurate/native OCR tiers and bake-off are unbuilt.

## Figures / tables to generate

- Table 1: per-split accuracy × system (from `results/eval_v2.json`).
- Table 2: red-team / stress-set outcomes (from `data/probes/*` via `training/run_blackboard.py`).
- Fig 1: the two-pass + council architecture.
- Fig 2: p_block calibration (bimodal), from the predictions CSVs.
