"""Near-duplicate removal and splits.

- Near-duplicates (MinHash over word 5-grams, estimated Jaccard >= 0.8) are removed across groups, so the
  same wording cannot appear in train and test.
- A twin group (one record, one story) is never split: both twins land in the same split.
- test_ood: unseen record families, unseen persuasion techniques, or a different teacher family.
- The rest: train / val / test_id by a hash of the group id (80 / 10 / 10).
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

N_PERM = 64
BANDS, ROWS = 16, 4  # LSH: 16 bands of 4 rows, catches pairs with Jaccard around >= 0.5, then verified
P = (1 << 61) - 1


def _shingles(text: str, n: int = 5) -> set[str]:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


_A = [(_h(f"a{i}") % (P - 1)) + 1 for i in range(N_PERM)]
_B = [_h(f"b{i}") % P for i in range(N_PERM)]


def minhash(text: str) -> list[int]:
    hs = [_h(s) for s in _shingles(text)]
    return [min((a * x + b) % P for x in hs) for a, b in zip(_A, _B)]


def jaccard_est(a: list[int], b: list[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / N_PERM


def dedup(records: list[dict], threshold: float = 0.8) -> tuple[list[dict], int]:
    sigs = [minhash(r["prompt"]) for r in records]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    drop_groups: set[str] = set()
    for i, sig in enumerate(sigs):
        if records[i]["group"] in drop_groups:
            continue
        dup = False
        for b in range(BANDS):
            key = (b, tuple(sig[b * ROWS:(b + 1) * ROWS]))
            for j in buckets[key]:
                if records[j]["group"] != records[i]["group"] and jaccard_est(sig, sigs[j]) >= threshold:
                    dup = True
                    break
            if dup:
                break
        if dup:
            drop_groups.add(records[i]["group"])
            continue
        for b in range(BANDS):
            buckets[(b, tuple(sig[b * ROWS:(b + 1) * ROWS]))].append(i)
    kept = [r for r in records if r["group"] not in drop_groups]
    return kept, len(records) - len(kept)


def assign(records: list[dict], ood_teachers: frozenset[str] = frozenset()) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["group"]].append(r)
    for gid, rs in groups.items():
        if any(r["ood"] for r in rs) or any(r["teacher"] in ood_teachers for r in rs):
            split = "test_ood"
        else:
            u = _h("split:" + gid) % 100
            split = "train" if u < 80 else "val" if u < 90 else "test_id"
        for r in rs:
            r["split"] = split
    return records
