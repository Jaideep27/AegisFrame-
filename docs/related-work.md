# Related work and positioning

Status: working notes from targeted searches (October 2026). Phase 0 still includes a systematic
review. Every entry below must be re-read in full before it is cited in the paper.

## The one-line gap

No existing defense guarantees that an unverified justification cannot turn a refused edit into an
allowed one, because in all of them **the component that reads the story is the component that
decides**. AegisFrame removes the story from the decision path by construction (closed EditSpec
schema), lets claims only add risk, and accepts only platform-verified identity as a way to lower it.

## Closest work

| Work | What it does | Difference |
|---|---|---|
| CICD-Guard (OpenReview vYtuGBTE8G) | Rewrites a prompt into several variants to expose hidden intent; joint LLM scoring; risk-adaptive decision. DSR ~91–100%, over-refusal 8–12%. | Closest in spirit. Still an LLM judging text that contains the packaging; no guarantee; text only; no image diff; no verified-claim path. **Main baseline to beat (B4).** |
| Erase-and-check (Kumar et al., COLM 2024) | Erases tokens, checks subsequences, certified vs. suffix/insertion attacks. | Token-level certificate. AegisFrame removes the *semantic* justification, and its guarantee is structural. |
| Intention Analysis (Zhang et al., COLING 2025) | Model first analyses essential intent, then answers. | The analysis reads the persuasive story. |
| Backtranslation (Wang et al., ACL Findings 2024) | Infers the prompt from the response; refuses if that is refused. | Text only. AegisFrame diffs the actual image edit. |
| AutoDefense (Zeng et al., 2024) | Multi-agent response filter. | Agents are not claim-blind; no purpose/verification/calibration. Our v3 council is baseline B5. |
| SAID (2025) | Distilled intents; reject if any is unsafe. | Supports veto logic; text only. |
| AuthorizationPairs (Jing et al., 2026, arXiv 2610.16838) | Holds an operation fixed, varies permission evidence; evaluation. | AegisFrame is a defense, separates claimed from verified context, for images. *Verify citation.* |
| CASE-Bench (Sun et al., ICML 2025) | Context changes human safety judgements. | Motivates "context counts only when verified". |
| Consistency training (Irpan et al., 2025) | Fine-tunes to ignore jailbreak wrappers. | Needs weights; AegisFrame works in front of closed APIs. |
| Constitutional Classifiers++ (Anthropic, 2026) | Exchange classifiers + cheap probes. | Provider-internal; AegisFrame is a plug-in for enterprises. |
| Paraphrase-invariance audits (OpenReview UqC9fYtLGY) | Jailbreak detection as permutation tests under benign paraphrase. | Evaluation methodology we can adopt for the flip-rate statistics. |

## Image editing safety

| Work | Note |
|---|---|
| IESBench / VJA (Hou et al., ICML 2026 oral, arXiv 2602.10179) | Vision-centric jailbreaks for image editors; "fabricate or manipulate visual evidence" consistently most exploitable; training-free introspective defense. Does not test justification framing or verified roles. |
| SI2ESBench (2026, arXiv 2610.23909) | Split-intent cross-modal attacks. *Verify citation.* |
| Chain-of-Jailbreak + Think-Twice (OpenReview V7PYbRzD0h) | Step-wise edits to bypass; prompting defense. Our cumulative-diff check targets the same decomposition. |
| Inscriptive jailbreak on T2I (arXiv 2604.05853) | Notes that a benign context justification is needed to get past LLM prompt filters: direct evidence that the attack we target works. |
| PhotoGuard, EditShield | Owner-side perturbations against editing; not applicable when the attacker owns the record. |

## Guard models (baseline B2 candidates)

Llama Guard 3 Vision / 4, ShieldGemma 2, LlavaGuard, LLaVAShield, SingGuard (arXiv 2606.22873,
policy-adaptive multimodal guard; check whether a custom "forged record" policy closes the gap, which
would make it a strong baseline).

## Document forgery detection (after the fact)

DocTamper (CVPR 2023), DocForge-Bench (arXiv 2603.01433), AIForge-Doc (arXiv 2602.20569),
GPT4o-Receipt (arXiv 2603.11442), "When the Forger Is the Judge" (arXiv 2604.25213: humans at chance
on paired receipt forgeries; the generator judging itself only modestly better). These motivate
prevention at generation time and could later feed AegisFrame as an input signal.
