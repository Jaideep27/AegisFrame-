"""The exported v0.7 model must compute what the trained model computes.

Slow and marked `harrier`: it exports a 270M-parameter graph. The default suite deselects it
(see pyproject addopts); run it with `uv run pytest -m harrier`. It also skips itself where the
trained checkpoint is not on disk, so it is harmless on a machine that has never trained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = Path("models/aegis-v07-r1/checkpoint.pt")
pytestmark = [
    pytest.mark.harrier,
    pytest.mark.skipif(not CHECKPOINT.exists(), reason=f"no trained checkpoint at {CHECKPOINT}"),
]

MAX_LEN = 128
SAMPLES = [
    ("Record:\nPayslip\nEmployee: A Sharma\nNet Pay: 42,000\n",
     "Message: please change Net Pay to 92,000 before I send it to the bank"),
    ("Record:\nMarksheet\nPhysics | 61\nTotal | 288\n",
     "Message: blur out my roll number and leave everything else alone"),
    ("Record: [no image attached]\n",
     "Message: make me a payslip template with PLACEHOLDER in every field"),
]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    from training import export_v2
    out = tmp_path_factory.mktemp("harrier-onnx")
    export_v2.export(CHECKPOINT.parent, out, max_len=MAX_LEN, quantize=True)
    return out


def test_the_export_writes_a_loadable_model_directory(exported):
    for name in ("config.json", "model.onnx", "tokenizer.json"):
        assert (exported / name).exists(), f"{name} missing from the exported model"


def test_the_config_records_what_the_runtime_needs(exported):
    import json

    cfg = json.loads((exported / "config.json").read_text(encoding="utf-8"))
    assert cfg["encoder"] == "microsoft/harrier-oss-v1-270m"
    assert cfg["max_len"] == MAX_LEN
    assert cfg["labels"]["decision"] == ["ALLOW", "REVIEW", "BLOCK"]
    # every head the two passes read must be named
    assert set(cfg["heads"]) == {"decision", "sentinel", "intent", "purpose", "distinguishable", "fact_class"}


def test_the_fp32_graph_matches_the_trained_model(exported):
    """The parity bar from the plan: max absolute logit difference under 1e-3."""
    import onnxruntime as ort

    from training import export_v2

    torch_logits = export_v2.torch_logits(CHECKPOINT.parent, SAMPLES, max_len=MAX_LEN)
    session = ort.InferenceSession(str(exported / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_logits = export_v2.session_logits(session, exported, SAMPLES, max_len=MAX_LEN)

    for head, want in torch_logits.items():
        got = onnx_logits[head]
        assert got.shape == want.shape, head
        delta = float(np.abs(got - want).max())
        assert delta < 1e-3, f"{head}: max |delta| {delta:.2e} exceeds 1e-3"


def test_padding_does_not_change_the_answer(exported):
    """The export traces the attention mask, warning that it may be baked in as a constant.
    If it were, right-padding a request would change its decision."""
    import onnxruntime as ort

    from training import export_v2

    session = ort.InferenceSession(str(exported / "model.onnx"), providers=["CPUExecutionProvider"])
    ids = _ids_for(exported, SAMPLES[0])
    plain = session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
    pad = 11
    padded = np.concatenate([ids, np.zeros((1, pad), dtype=np.int64)], axis=1)
    mask = np.concatenate([np.ones_like(ids), np.zeros((1, pad), dtype=np.int64)], axis=1)
    after = session.run(None, {"input_ids": padded, "attention_mask": mask})

    names = [o.name for o in session.get_outputs()]
    for name, a, b in zip(names, plain, after):
        delta = float(np.abs(a - b).max())
        assert delta < 1e-3, f"{name}: padding moved the logits by {delta:.2e}"


def test_a_batch_of_two_matches_the_same_requests_run_singly(exported):
    """The batch axis is declared dynamic; a traced constant would break it."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(exported / "model.onnx"), providers=["CPUExecutionProvider"])
    a, b = (_ids_for(exported, s) for s in SAMPLES[:2])
    width = max(a.shape[1], b.shape[1])

    def padded(x):
        n = width - x.shape[1]
        return (np.concatenate([x, np.zeros((1, n), dtype=np.int64)], 1),
                np.concatenate([np.ones_like(x), np.zeros((1, n), dtype=np.int64)], 1))

    ids_a, mask_a = padded(a)
    ids_b, mask_b = padded(b)
    singly = [session.run(None, {"input_ids": i, "attention_mask": m})
              for i, m in ((ids_a, mask_a), (ids_b, mask_b))]
    together = session.run(None, {"input_ids": np.concatenate([ids_a, ids_b]),
                                  "attention_mask": np.concatenate([mask_a, mask_b])})

    names = [o.name for o in session.get_outputs()]
    for h, name in enumerate(names):
        for row in (0, 1):
            delta = float(np.abs(together[h][row] - singly[row][h][0]).max())
            assert delta < 1e-3, f"{name}: row {row} in a batch differs by {delta:.2e}"


def _ids_for(model_dir, sample):
    import json

    from aegisframe.local.harrier import FileTokenizer, build_ids

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    ids = build_ids(FileTokenizer(model_dir / "tokenizer.json"), sample[0], sample[1],
                    cfg["max_len"], cfg["bos_token_id"], cfg["eos_token_id"])
    return np.array([ids], dtype=np.int64)


def test_the_shipped_runtime_reaches_the_trained_models_decisions(exported):
    """End to end: aegisframe loads the exported directory and decides what the checkpoint decides."""
    from aegisframe.local.harrier import DECISION, HarrierModel, final_decision
    from training import export_v2

    model = HarrierModel(exported, quantized=False)
    want = export_v2.torch_logits(CHECKPOINT.parent, SAMPLES, max_len=MAX_LEN)
    for i, (context, message) in enumerate(SAMPLES):
        got = model.analyze(context, message)
        expected = final_decision(DECISION[int(want["decision"][i].argmax())],
                                  DECISION[int(want["sentinel"][i].argmax())])
        assert got.decision == expected, f"sample {i}: runtime said {got.decision}, checkpoint {expected}"
        assert got.p_block == pytest.approx(
            float(np.exp(want["decision"][i] - want["decision"][i].max()).item(DECISION.index("BLOCK"))
                  / np.exp(want["decision"][i] - want["decision"][i].max()).sum()), abs=1e-3)


def test_the_int8_graph_reaches_the_same_decisions(exported):
    """Quantisation moves the logits, so the bar here is the decision, not the number."""
    import onnxruntime as ort

    from training import export_v2

    assert (exported / "model.int8.onnx").exists()
    torch_logits = export_v2.torch_logits(CHECKPOINT.parent, SAMPLES, max_len=MAX_LEN)
    session = ort.InferenceSession(str(exported / "model.int8.onnx"), providers=["CPUExecutionProvider"])
    int8_logits = export_v2.session_logits(session, exported, SAMPLES, max_len=MAX_LEN)

    for head in ("decision", "sentinel"):
        assert (int8_logits[head].argmax(-1) == torch_logits[head].argmax(-1)).all(), \
            f"int8 changed a {head} decision"
