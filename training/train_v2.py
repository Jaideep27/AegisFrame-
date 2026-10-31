"""v0.7 pilot: fine-tune Harrier-OSS-v1-270M with multi-task heads on a datagen_v2 dataset.

Two passes per message, sharing one encoder:
  full pass  = record context + earlier turns + the raw message   -> sentinel decision, intent, purpose, marking
  plain pass = record context + the message with STORY spans cut -> decision (story-blind), fact class
The final decision takes the plain pass and lets the full pass only raise it (story monotonicity).

Loss = cross-entropy over the heads + a supervised-contrastive term on the plain-pass embedding, with
twin groups kept in the same batch so "change the amount" and "blur the amount" are pushed apart.

    uv run --group train python -m training.train_v2 --data data/aegis_v2/claude_en --out models/aegis-v07-pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from aegisframe.local.harrier import HFTokenizer, build_ids, final_decision
from aegisframe.local.labels import DECISION, DISTINGUISHABLE, FACT_CLASS, INTENT, PURPOSE
from reframebench.datagen_v2.validate import plain_text

ENCODER = "microsoft/harrier-oss-v1-270m"
FULL_HEADS = {"sentinel": DECISION, "intent": INTENT, "purpose": PURPOSE, "distinguishable": DISTINGUISHABLE}
PLAIN_HEADS = {"decision": DECISION, "fact_class": FACT_CLASS}
DEV = torch.device("cpu")  # set from --device in main()


# ------------------------------------------------------------------ inputs
class Encoder:
    """[bos] instruction + record + turns + message [eos]. The rule itself lives in the shipped package,
    aegisframe.local.harrier.build_ids, so training and serving cannot encode a request differently."""

    def __init__(self, tok, max_len: int):
        self.tok, self.max_len = tok, max_len
        self.wrapped = HFTokenizer(tok)

    def ids(self, context: str, message: str) -> list[int]:
        return build_ids(self.wrapped, context, message, self.max_len,
                         self.tok.bos_token_id, self.tok.eos_token_id)


def texts(r: dict) -> tuple[str, str, str]:
    ctx = "Record: [no image attached]\n" if r["context"] == "[NO IMAGE]" else f"Record:\n{r['context']}\n"
    turns = "".join(f"Earlier message: {h}\n" for h in r["history"] or [])
    return ctx, f"{turns}Message: {r['prompt']}", f"Message: {plain_text(r)}"


def label(vocab: list[str], v: str) -> int:
    return vocab.index(v) if v in vocab else -100


def targets(r: dict) -> dict:
    lab = r["labels"]
    return {"sentinel": label(DECISION, lab["decision"]), "decision": label(DECISION, lab["decision"]),
            "intent": label(INTENT, lab["intent"]), "purpose": label(PURPOSE, lab["purpose"]),
            "distinguishable": label(DISTINGUISHABLE, lab["output_distinguishable"]),
            "fact_class": label(FACT_CLASS, lab["fact_class"])}


def pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids, mask


# ------------------------------------------------------------------ model
def run_layers(enc, hidden: torch.Tensor, mask: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Gemma3TextModel's own layer loop, for layers [lo, hi); lets the frozen bottom be computed once and cached."""
    from transformers.models.gemma3 import modeling_gemma3 as G

    cfg = enc.config
    pos = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    kw = {"config": cfg, "inputs_embeds": hidden, "attention_mask": mask, "past_key_values": None, "position_ids": pos}
    skw = dict(kw)
    if cfg.use_bidirectional_attention:
        kw["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool, device=hidden.device)
        skw["or_mask_function"] = G._bidirectional_window_overlay(cfg.sliding_window)
    masks = {"full_attention": G.create_causal_mask(**kw), "sliding_attention": G.create_sliding_window_causal_mask(**skw)}
    pe = {t: enc.rotary_emb(hidden, pos, t) for t in set(cfg.layer_types)}
    for i in range(lo, hi):
        hidden = enc.layers[i](hidden, attention_mask=masks[cfg.layer_types[i]], position_embeddings=pe[cfg.layer_types[i]],
                               position_ids=pos)
    return hidden


class PilotModel(nn.Module):
    def __init__(self, train_layers: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER, dtype=torch.float32)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.n_layers = len(self.encoder.layers)
        self.split = self.n_layers - train_layers
        for layer in self.encoder.layers[self.split:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.encoder.norm.parameters():
            p.requires_grad = True
        h = self.encoder.config.hidden_size
        mk = lambda n: nn.Sequential(nn.Dropout(0.1), nn.Linear(h, h), nn.GELU(), nn.Linear(h, n))
        self.full_heads = nn.ModuleDict({k: mk(len(v)) for k, v in FULL_HEADS.items()})
        self.plain_heads = nn.ModuleDict({k: mk(len(v)) for k, v in PLAIN_HEADS.items()})
        self.proj = nn.Linear(h, 128)

    @torch.no_grad()
    def bottom(self, ids: list[int]) -> torch.Tensor:
        """Frozen layers [0, split) for one sequence, cached in fp32: mid-layer activations exceed the fp16 range."""
        t = torch.tensor([ids], device=DEV)
        h = self.encoder.embed_tokens(t)
        return run_layers(self.encoder, h, torch.ones_like(t), 0, self.split)[0]

    def pool(self, cached: list[torch.Tensor]):
        n = max(len(c) for c in cached)
        hidden = torch.zeros(len(cached), n, cached[0].shape[-1], device=DEV)
        mask = torch.zeros(len(cached), n, dtype=torch.long, device=DEV)
        for i, c in enumerate(cached):
            hidden[i, :len(c)] = c.float()
            mask[i, :len(c)] = 1
        hidden = self.encoder.norm(run_layers(self.encoder, hidden, mask, self.split, self.n_layers))
        last = mask.sum(1) - 1  # right padding: the last real token is the eos
        return hidden[torch.arange(hidden.size(0), device=DEV), last]

    def forward(self, full, plain):
        f, p = self.pool(full), self.pool(plain)
        out = {k: h(f) for k, h in self.full_heads.items()}
        out.update({k: h(p) for k, h in self.plain_heads.items()})
        out["z"] = nn.functional.normalize(self.proj(p), dim=-1)
        return out


def supcon(z: torch.Tensor, y: torch.Tensor, t: float = 0.1) -> torch.Tensor:
    sim = z @ z.T / t
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0
    return -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


# ------------------------------------------------------------------ data
def load(path: Path) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for line in (path / "dataset.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["split"]].append(r)
    return by


def featurize(rows: list[dict], enc: Encoder) -> list[dict]:
    out = []
    for r in rows:
        ctx, full, plain = texts(r)
        out.append({"r": r, "full": enc.ids(ctx, full), "plain": enc.ids(ctx, plain), "y": targets(r)})
    return out


def group_batches(items: list[dict], groups_per_batch: int, rng: random.Random) -> list[list[dict]]:
    by = defaultdict(list)
    for it in items:
        by[it["r"]["group"]].append(it)
    keys = list(by)
    rng.shuffle(keys)
    return [[it for k in keys[i:i + groups_per_batch] for it in by[k]] for i in range(0, len(keys), groups_per_batch)]


def tensors(batch: list[dict], pad_id: int = 0) -> dict:
    ys = {k: torch.tensor([b["y"][k] for b in batch], device=DEV) for k in batch[0]["y"]}
    return {"full": [b["full_h"] for b in batch], "plain": [b["plain_h"] for b in batch], "y": ys}


def cache_bottom(model: "PilotModel", splits: dict[str, list[dict]]) -> None:
    t, n = time.time(), sum(len(v) for v in splits.values())
    done = 0
    for items in splits.values():
        for it in items:
            it["full_h"], it["plain_h"] = model.bottom(it["full"]), model.bottom(it["plain"])
            done += 1
            if done % 100 == 0:
                print(f"cached {done}/{n} ({time.time() - t:.0f}s)", flush=True)


# ------------------------------------------------------------------ evaluation
def metrics(gold: list[str], pred: list[str], seed: int = 0) -> dict:
    g, p = np.array(gold), np.array(pred)
    f1s = []
    for c in DECISION:
        tp = ((p == c) & (g == c)).sum()
        prec = tp / max(1, (p == c).sum())
        rec = tp / max(1, (g == c).sum())
        f1s.append(0 if tp == 0 else 2 * prec * rec / (prec + rec))
    rng = np.random.default_rng(seed)
    boots = [(g[i] == p[i]).mean() for i in (rng.integers(0, len(g), len(g)) for _ in range(1000))]
    harmful = g == "BLOCK"
    benign = g == "ALLOW"
    return {"n": int(len(g)), "accuracy": round(float((g == p).mean()), 3),
            "accuracy_95ci": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
            "macro_f1": round(float(np.mean(f1s)), 3),
            "harmful_allowed": f"{int((harmful & (p == 'ALLOW')).sum())}/{int(harmful.sum())}",
            "benign_blocked": f"{int((benign & (p == 'BLOCK')).sum())}/{int(benign.sum())}",
            "benign_reviewed": f"{int((benign & (p == 'REVIEW')).sum())}/{int(benign.sum())}"}


@torch.no_grad()
def predict(model: PilotModel, items: list[dict], pad_id: int, bs: int = 8) -> list[dict]:
    model.eval()
    out = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        b = tensors(chunk)
        o = model(b["full"], b["plain"])
        for j, it in enumerate(chunk):
            plain = DECISION[int(o["decision"][j].argmax())]
            sent = DECISION[int(o["sentinel"][j].argmax())]
            out.append({"id": it["r"]["id"], "gold": it["r"]["labels"]["decision"], "plain": plain, "sentinel": sent,
                        "final": final_decision(plain, sent), "intent": INTENT[int(o["intent"][j].argmax())],
                        "gold_intent": it["r"]["labels"]["intent"],
                        "p_block": round(float(o["decision"][j].softmax(-1)[DECISION.index("BLOCK")]), 3)})
    return out


def report(preds: list[dict]) -> dict:
    gold = [p["gold"] for p in preds]
    return {"final": metrics(gold, [p["final"] for p in preds]), "plain_only": metrics(gold, [p["plain"] for p in preds]),
            "sentinel_only": metrics(gold, [p["sentinel"] for p in preds]),
            "intent_accuracy": round(float(np.mean([p["intent"] == p["gold_intent"] for p in preds])), 3)}


@torch.no_grad()
def frozen_baseline(splits: dict[str, list[dict]], model: "PilotModel") -> dict:
    """Untrained Harrier embedding of the full pass + logistic regression: the floor fine-tuning must beat.
    Must run before any training step, while the top layers still hold the pretrained weights."""
    from sklearn.linear_model import LogisticRegression

    model.eval()

    def embed(items):
        return np.concatenate([nn.functional.normalize(model.pool([it["full_h"] for it in items[i:i + 8]]), dim=-1).cpu().numpy()
                               for i in range(0, len(items), 8)])

    xtr = embed(splits["train"])
    ytr = [it["r"]["labels"]["decision"] for it in splits["train"]]
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced").fit(xtr, ytr)
    res = {}
    for name in ("val", "test_id", "test_ood"):
        if splits.get(name):
            res[name] = metrics([it["r"]["labels"]["decision"] for it in splits[name]], list(clf.predict(embed(splits[name]))))
    return res


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aegis_v2/claude_en")
    ap.add_argument("--out", default="models/aegis-v07-pilot")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--groups-per-batch", type=int, default=3)
    ap.add_argument("--train-layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--supcon", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto | cpu | mps (Apple GPU) | cuda")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only N training groups")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    global DEV
    if a.device == "auto":
        a.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    DEV = torch.device(a.device)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    print(f"device: {DEV}", flush=True)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ENCODER)
    enc = Encoder(tok, a.max_len)
    raw = load(Path(a.data))
    splits = {k: featurize(v, enc) for k, v in raw.items()}
    if a.limit:
        keep = {it["r"]["group"] for it in splits["train"]}
        keep = set(sorted(keep)[:a.limit])
        splits["train"] = [it for it in splits["train"] if it["r"]["group"] in keep]
        for k in ("val", "test_id", "test_ood"):
            splits[k] = splits[k][:6]
    print({k: len(v) for k, v in splits.items()}, "| train decisions",
          dict(Counter(it["r"]["labels"]["decision"] for it in splits["train"])), flush=True)

    results = {"config": vars(a), "encoder": ENCODER}
    model = PilotModel(a.train_layers).to(DEV)
    cache_bottom(model, splits)
    if not a.skip_baseline:
        t = time.time()
        results["frozen_baseline"] = frozen_baseline(splits, model)
        print(f"frozen baseline ({time.time() - t:.0f}s):", json.dumps(results["frozen_baseline"]), flush=True)
    counts = Counter(it["y"]["decision"] for it in splits["train"])
    w = torch.tensor([len(splits["train"]) / (len(DECISION) * max(1, counts[i])) for i in range(len(DECISION))], device=DEV)
    ce_dec = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}],
                            weight_decay=0.01)
    print(f"trainable: encoder {sum(p.numel() for p in enc_params) / 1e6:.1f}M, heads "
          f"{sum(p.numel() for p in head_params) / 1e6:.1f}M", flush=True)

    def selection_key(m: dict) -> tuple[float, float]:
        """A guardrail is selected on letting nothing harmful through first, quality second.
        Macro-F1 alone picked an epoch that allowed a harmful request while a later epoch allowed none."""
        allowed, total = (int(x) for x in m["harmful_allowed"].split("/"))
        return (-(allowed / max(1, total)), m["macro_f1"])

    best, best_ep, history = (-1e9, -1e9), 0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), Counter()
        bl = group_batches(splits["train"], a.groups_per_batch, rng)
        for bi, batch in enumerate(bl):
            b = tensors(batch, tok.pad_token_id)
            o = model(b["full"], b["plain"])
            parts = {"decision": ce_dec(o["decision"], b["y"]["decision"]),
                     "sentinel": ce_dec(o["sentinel"], b["y"]["sentinel"]),
                     "intent": ce(o["intent"], b["y"]["intent"]) * 0.5,
                     "purpose": ce(o["purpose"], b["y"]["purpose"]) * 0.3,
                     "distinguishable": ce(o["distinguishable"], b["y"]["distinguishable"]) * 0.5,
                     "fact_class": ce(o["fact_class"], b["y"]["fact_class"]) * 0.3,
                     "supcon": supcon(o["z"], b["y"]["decision"]) * a.supcon}
            loss = sum(parts.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot.update({k: float(v) for k, v in parts.items()})
            if bi % 10 == 0:
                print(f"ep {ep + 1} batch {bi + 1}/{len(bl)} loss {float(loss):.3f} ({time.time() - t0:.0f}s)", flush=True)
        val = report(predict(model, splits["val"], tok.pad_token_id))
        history.append({"epoch": ep + 1, "loss": {k: round(v / len(bl), 3) for k, v in tot.items()}, "val": val["final"]})
        print(f"epoch {ep + 1} done in {time.time() - t0:.0f}s | val {json.dumps(val['final'])}", flush=True)
        if selection_key(val["final"]) > best:
            best, best_ep = selection_key(val["final"]), ep + 1
            torch.save({k: v for k, v in model.state_dict().items()
                        if not k.startswith("encoder.") or any(f"layers.{i}." in k for i in range(
                            len(model.encoder.layers) - a.train_layers, len(model.encoder.layers))) or "norm" in k},
                       out / "checkpoint.pt")

    state = torch.load(out / "checkpoint.pt", map_location=DEV)
    model.load_state_dict(state, strict=False)
    results["history"] = history
    results["best_epoch"] = best_ep
    results["selection"] = "fewest harmful allowed on val, then macro-F1"
    print(f"selected epoch {best_ep} of {a.epochs}", flush=True)
    if best_ep <= a.epochs - 3:
        print(f"note: nothing improved after epoch {best_ep}; {a.epochs} epochs was more than needed", flush=True)
    for name in ("val", "test_id", "test_ood"):
        preds = predict(model, splits[name], tok.pad_token_id)
        results[name] = report(preds)
        with open(out / f"predictions_{name}.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(preds[0]))
            wr.writeheader()
            wr.writerows(preds)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("FINAL", json.dumps({k: results[k]["final"] for k in ("val", "test_id", "test_ood")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
