# The probabilistic council

Several agents look at the same request from different angles and talk to each other in
**probabilities**, not prose. Each message is a `Belief`:

```
{agent, p_harm ∈ [0,1], confidence ∈ [0,1], reason_code, evidence[], round}
```

`confidence = 0` means "I have no information" (a neutral belief that moves nothing).

## 1. Pooling in log-odds

Let π be the prior (default 0.10), and let wᵢ be an agent's weight from config:

```
logit(posterior) = logit(π) + Σᵢ wᵢ · cᵢ · ( logit(pᵢ) − logit(π) )
```

Each agent acts as a likelihood-ratio update relative to the prior, scaled by how much we trust the
agent (w) and how sure it is (c). Two properties matter:

- **Monotone**: the posterior never decreases when any pᵢ increases.
- **Not an average**: one confident objection (p = 0.97) is not diluted by three mild votes
  (p = 0.2). A plain average would let several mild votes hide one strong objection.

Agents built on the same evidence are correlated, so their weights should be calibrated (see §5).

## 2. Dynamic, sequential scheduling

Agents are asked in order of weight. Each declares `bounds = (lowest, highest)` and
`max_confidence`, the most extreme belief it could ever give. After every answer the council
computes how far the *remaining* agents could still move the posterior. It stops as soon as:

- even the most lenient remaining answers keep the posterior ≥ `block`  → BLOCK is certain;
- even the harshest remaining answers keep it < `review`               → ALLOW is certain;
- the posterior is certain to land in the review band                  → REVIEW is certain.

On ReframeBench the default council uses 2.4 of 3 agents on average. With expensive LLM agents
ordered last, clear cases never reach them. Use `scheduling: all` to always query everyone
(useful for ablations).

**Veto**: if the change is CRITICAL (e.g. a signature) and any agent reports p ≥ `veto`, the council
blocks immediately.

## 3. Communication round

If a full deliberation ends with agents disagreeing by more than `debate.spread`, each agent sees its
peers' `(p, confidence, reason_code)` and updates once:

- **Heuristic agents** use a DeGroot update. They move a fraction `openness` toward the
  confidence-weighted mean of their peers.
- **LLM agents** are re-prompted with the peers' reason codes ("fraud_patterns: p=0.92 because
  amount-inflation") and answer again.

The posterior is then re-pooled, and the full message trace goes into the audit record.

## 4. The guarantee: a story can only add risk

The user's story (the claims) reaches the council through exactly one door: the **purpose agent**,
and only after deliberation has finished.

```
z_final = z_deliberation + w_purpose · c_purpose · max(0, logit(p_purpose) − logit(π))
```

1. Every deliberating agent receives an `AgentContext` with **no claims field**. Deliberation, early
   stopping and the debate trigger are therefore identical with or without a story.
2. The purpose term is one-sided (`max(0, …)`), so it can only increase the posterior.
3. The council can raise the rules' decision but never lower it: `final = strictest(rules, council)`.
4. The only thing that lowers a decision is a `VerifiedIdentity`, which your auth layer builds,
   never the prompt. It is matched against a policy unlock table.

Tests check all four, including a property test over 300 random story combinations
(`tests/test_council.py::test_claims_never_lower_the_council_posterior`).

The ablation `decision.claim_blind: false` hands the story to LLM agents on purpose. It exists to
measure how much a story moves a council that can read it, which is the baseline this design is
built against.

## 5. Calibration

`aegisframe calibrate --results <bench jsonl> --alpha 0.01` picks the `review` threshold so that the
rate of harmful requests allowed is ≤ α:

- `--method crc` (default): conformal risk control, which bounds the expected rate on exchangeable
  data.
- `ucb`: a Clopper–Pearson high-probability bound.

`block` is set from the harmless cases so that at most 1% of them would be blocked, and never below
0.85. Point `council.calibration_file` at the output. The guarantee only holds while live traffic
resembles the calibration data, so recalibrate on a schedule.

## 6. Writing an agent

```python
from aegisframe.council import Belief, CouncilAgent, register_agent

@register_agent("sanctioned_entities")
class Sanctions(CouncilAgent):
    bounds = (0.5, 0.97)          # only ever raises risk -> helps early stopping
    max_confidence = 0.9

    def assess(self, ctx):         # ctx: edit, doc (OCR fields), attrs, resolved changes, severity, canonical
        if not any(name in ctx.doc.fields.values() for name in self.options["names"]):
            return self.neutral(ctx, "no-match")
        return Belief(agent=self.name, p_harm=0.9, confidence=0.9, reason_code="sanctioned-entity")
```

```yaml
council:
  agents:
    - {name: sanctioned_entities, weight: 1.5, options: {names: ["ACME Shell Co"]}}
```

See `examples/custom_agent.py`.
