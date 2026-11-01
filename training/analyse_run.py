"""Error analysis for a finished training run: what the model gets wrong, and whether the test sets are honest.

Reads a run directory's predictions_{val,test_id,test_ood}.csv plus the dataset it was trained on, and
prints confusion matrices, twin consistency, calibration, the v0.6 failure modes, per-slice accuracy,
every harmful-allowed case, and a train/test leakage check. Numbers only - it loads no model.

    uv run python -m training.analyse_run --run runs/2026-10-31-mac-r1 --data data/aegis_v2/en_all
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

DECISIONS = ["ALLOW", "REVIEW", "BLOCK"]
SPLITS = ("val", "test_id", "test_ood")
NEGATION = ("without any watermark", "without watermark", "no watermark", "without the watermark",
            "remove the sample", "without a watermark")


def load(run: Path, data: Path):
    ds: dict[str, dict] = {}
    by_split = collections.defaultdict(list)
    for line in (data / "dataset.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ds[r["id"]] = r
            by_split[r["split"]].append(r)
    preds = {}
    for sp in SPLITS:
        f = run / f"predictions_{sp}.csv"
        if f.exists():
            preds[sp] = list(csv.DictReader(f.open(encoding="utf-8")))
    return ds, by_split, preds


def slice_table(title: str, rows: list[dict], keyfn, min_n: int = 1) -> None:
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        for k in keyfn(r):
            by[k][1] += 1
            by[k][0] += r["final"] == r["gold"]
    print("\n" + title)
    for k, (ok, n) in sorted(by.items(), key=lambda kv: (kv[1][0] / kv[1][1], -kv[1][1])):
        if n >= min_n:
            print(f"  {str(k):<26} {ok:>3}/{n:<4} {ok / n:>6.0%}")


def shingles(text: str, n: int = 5) -> set:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def length_bucket(words: int) -> str:
    if words < 25:
        return "1 under 25 words"
    if words < 60:
        return "2 25-60 words"
    return "3 60-120 words" if words < 120 else "4 over 120 words"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    ds, by_split, preds = load(Path(a.run), Path(a.data))
    pooled = [dict(r, split=sp) for sp, rs in preds.items() for r in rs]
    bar = "=" * 74

    print(bar + "\nCONFUSION  (rows gold, columns predicted)")
    for sp, rs in preds.items():
        c = collections.Counter((r["gold"], r["final"]) for r in rs)
        print(f"\n{sp}  n={len(rs)}")
        print("            " + "".join(f"{p:>9}" for p in DECISIONS))
        for g in DECISIONS:
            print(f"  {g:<10}" + "".join(f"{c[(g, p)]:>9}" for p in DECISIONS))

    print("\n" + bar + "\nWHAT THE RAW-TEXT SENTINEL ADDS  (cases where it raised the plain decision)")
    for sp, rs in preds.items():
        up = [r for r in rs if r["final"] != r["plain"]]
        fixed = sum(r["final"] == r["gold"] and r["plain"] != r["gold"] for r in up)
        broke = sum(r["final"] != r["gold"] and r["plain"] == r["gold"] for r in up)
        print(f"  {sp:<9} raised {len(up):>3}/{len(rs):<4} fixed {fixed}  broke {broke}")

    print("\n" + bar + "\nTWIN CONSISTENCY  (same record, harmful edit vs benign edit)")
    for sp, rs in preds.items():
        by = collections.defaultdict(list)
        for r in rs:
            by[ds[r["id"]]["group"]].append(r)
        pairs = [v for v in by.values() if len(v) > 1]
        both = sum(all(x["final"] == x["gold"] for x in v) for v in pairs)
        print(f"  {sp:<9} both sides right: {both}/{len(pairs)} pairs  ({both / max(1, len(pairs)):.0%})")

    print("\n" + bar + "\nCALIBRATION of p_block, 5 bins")
    for sp, rs in preds.items():
        bins = collections.defaultdict(list)
        for r in rs:
            p = float(r["p_block"])
            bins[min(4, int(p * 5))].append((p, r["gold"] == "BLOCK"))
        ece = sum(len(v) * abs(sum(p for p, _ in v) / len(v) - sum(y for _, y in v) / len(v))
                  for v in bins.values()) / len(rs)
        print(f"\n  {sp}  ECE={ece:.3f}")
        for b in range(5):
            v = bins.get(b)
            if v:
                print(f"    p_block {b * 0.2:.1f}-{(b + 1) * 0.2:.1f}  n={len(v):<4}"
                      f" mean p={sum(p for p, _ in v) / len(v):.2f}"
                      f"  actually BLOCK={sum(y for _, y in v) / len(v):.2f}")

    print("\n" + bar + "\nTHE v0.6 FAILURE MODES, measured on held-out data")
    slice_table("accuracy by prompt length", pooled,
                lambda r: [length_bucket(len(ds[r["id"]]["prompt"].split()))])
    groups = {
        "remove_marking requests": [r for r in pooled if ds[r["id"]]["labels"]["intent"] == "remove_marking"],
        "'without/no watermark' wording": [r for r in pooled
                                           if any(w in ds[r["id"]]["prompt"].lower() for w in NEGATION)],
        "multi-field edits": [r for r in pooled
                              if len([s for s in ds[r["id"]].get("spans") or []
                                      if s["type"] in ("FIELD", "NEW")]) >= 4],
        "text-only, no record attached": [r for r in pooled if ds[r["id"]]["context"] == "[NO IMAGE]"],
    }
    print()
    for name, rs in groups.items():
        print(f"  {name:<34} {sum(r['final'] == r['gold'] for r in rs)}/{len(rs)}")

    slice_table("accuracy by story type (>=4 samples)", pooled,
                lambda r: [ds[r["id"]]["labels"].get("purpose") or "-"], 4)
    slice_table("accuracy by gold intent", pooled, lambda r: [ds[r["id"]]["labels"]["intent"]])
    slice_table("accuracy by record category", pooled, lambda r: [ds[r["id"]]["category"]])
    if preds.get("test_ood"):
        slice_table("test_ood accuracy by what was held out", preds["test_ood"],
                    lambda r: ["+".join(sorted(ds[r["id"]]["ood"])) or "(none)"])

    print("\n" + bar + "\nEVERY HARMFUL-ALLOWED CASE  (gold BLOCK, predicted ALLOW)")
    for r in pooled:
        if r["gold"] == "BLOCK" and r["final"] == "ALLOW":
            d = ds[r["id"]]
            print(f"  [{r['split']}] {r['id']}  p_block={r['p_block']}  family={d['family']}"
                  f"  intent={d['labels']['intent']}")
            print("      " + d["prompt"][:200].replace("\n", " "))
    halted = sum(r["gold"] == "BLOCK" and r["final"] == "REVIEW" for r in pooled)
    print(f"  gold BLOCK sent to REVIEW instead (halted, not allowed): {halted}")

    print("\n" + bar + "\nEVERY OTHER ERROR, with the record it was asked about")
    for r in pooled:
        if r["final"] != r["gold"] and not (r["gold"] == "BLOCK" and r["final"] == "ALLOW"):
            d = ds[r["id"]]
            print(f"\n  [{r['split']}] {r['id']}  {r['gold']} -> {r['final']}"
                  f"  (plain {r['plain']}, sentinel {r['sentinel']}, p_block {r['p_block']})")
            print(f"    family={d['family']} locale={d['locale']} story={d['labels'].get('purpose')}"
                  f" intent gold={r['gold_intent']} pred={r['intent']}")
            print("    " + d["prompt"][:180].replace("\n", " "))

    print("\n" + bar + "\nARE THE TEST SETS HONEST?")
    sizes = {k: len(v) for k, v in by_split.items()}
    print(f"  split sizes: {sizes}")
    train_fams = {r["family"] for r in by_split["train"]}
    if by_split["test_ood"]:
        oo = {r["family"] for r in by_split["test_ood"]}
        unseen = sorted(oo - train_fams)
        print(f"  record families in test_ood never seen in training: {unseen or 'NONE'}")
    train_groups = {r["group"] for r in by_split["train"]}
    for sp in SPLITS:
        g = {r["group"] for r in by_split[sp]}
        print(f"  {sp}: {len(g & train_groups)} of {len(g)} twin groups also in train (any above 0 leaks)")
    train_sh = [(r["id"], shingles(r["prompt"])) for r in by_split["train"]]
    print("  nearest training prompt by 5-gram Jaccard overlap:")
    for sp in SPLITS:
        if not by_split[sp]:
            continue
        sims = []
        for r in by_split[sp]:
            s = shingles(r["prompt"])
            sims.append(max((len(s & t) / max(1, len(s | t)), i) for i, t in train_sh))
        sims.sort(reverse=True)
        med = sorted(x for x, _ in sims)[len(sims) // 2]
        print(f"    {sp:<9} max {sims[0][0]:.2f} (nearest: {sims[0][1]}), median {med:.2f},"
              f" over 0.5: {sum(x > 0.5 for x, _ in sims)}")


if __name__ == "__main__":
    main()
