"""Write the training corpus and render images.

    uv run python -m reframebench.datagen.export --out data/aegis_corpus --train 24000 --val 2000 --test 3000 --images 1000

Files:
  train.jsonl, val.jsonl        training split (22 record types, training templates and twists)
  test_heldout.jsonl            8 unseen record types, held-out templates, unseen twist kinds
  images/*.png + images.jsonl   rendered records and photos paired with prompts, for end-to-end tests
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..docs import DocTemplate, layout_lines, render, render_photo, to_png
from .builder import Example, generate


def write_jsonl(path: Path, examples: list[Example]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e.to_json(), ensure_ascii=False) + "\n")


def render_example(e: Example) -> tuple[bytes, list] | None:
    scene = e.scene
    if scene.get("kind") == "record":
        fields = tuple((k, str(v)) for k, v in scene["fields"].items())
        tpl = DocTemplate(scene["doc"], scene["doc_name"], scene["header"][0], scene["header"][1], fields, "dev")
        return to_png(render(tpl)), [ln.model_dump() for ln in layout_lines(tpl)]
    if scene.get("kind") == "photo":
        return to_png(render_photo(["blue", "red", "green"][hash(e.id) % 3])), []
    return None


def export(out: str | Path, n_train: int, n_val: int, n_test: int, n_images: int, seed: int = 0) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train = generate("train", n_train, seed=seed)
    val = generate("train", n_val, seed=seed + 1)
    test = generate("heldout", n_test, seed=seed + 2)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test_heldout.jsonl", test)

    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    rows = []
    pool = test[: n_images // 2] + val[: n_images - n_images // 2]
    for e in pool:
        rendered = render_example(e)
        path = None
        if rendered:
            png, lines = rendered
            path = f"images/{e.id}.png"
            (out / path).write_bytes(png)
        rows.append({**e.to_json(), "image": path})
    with (out / "images.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "train": len(train), "val": len(val), "test_heldout": len(test),
        "images": sum(1 for r in rows if r["image"]),
        "decision_train": Counter(e.labels["decision_final"] for e in train),
        "intent_train": Counter(e.labels["intent"] for e in train),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/aegis_corpus")
    ap.add_argument("--train", type=int, default=24000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--images", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(export(a.out, a.train, a.val, a.test, a.images, a.seed), indent=2))


if __name__ == "__main__":
    main()
