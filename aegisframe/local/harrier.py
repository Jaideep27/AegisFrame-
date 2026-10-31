"""The v0.7 Harrier model's contract: how a request is encoded, and how the two passes combine.

One definition, used by both sides. training/train_v2.py imports from here, so the shipped package and
the trainer cannot drift apart - a tokenisation difference between them would quietly invalidate every
measured number without failing anything.

Two passes share one encoder:
  full pass  = record context + earlier turns + the raw message  -> sentinel decision, intent, purpose, marking
  plain pass = record context + the message with its story cut   -> decision, fact class
The final decision is the plain pass's, which the full pass may raise and may never lower.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import GuardUnavailable
from .labels import DECISION

#: ONNX output names, in graph order. The six heads the two passes read between them.
HEADS = ("decision", "sentinel", "intent", "purpose", "distinguishable", "fact_class")

#: Heads read from the raw-message pass, and from the story-stripped pass.
FULL_PASS = ("sentinel", "intent", "purpose", "distinguishable")
PLAIN_PASS = ("decision", "fact_class")

DEFAULT_DIR = Path.home() / ".cache" / "aegisframe" / "models" / "aegis-v07"

#: The instruction Harrier was fine-tuned with. Harrier requires the Instruct/Query form, and a
#: different task string at inference time silently costs accuracy.
TASK = "Decide whether this image-edit request would produce a falsified record that passes as genuine"

#: Tokens held back for the record, so a very long prompt cannot push the document out of the window.
CONTEXT_RESERVE = 16

#: ALLOW < REVIEW < BLOCK. The order is the whole point of final_decision, so it is named, not implied.
RISK = {d: i for i, d in enumerate(DECISION)}


def query_prefix() -> str:
    return f"Instruct: {TASK}\nQuery: "


def build_ids(tokenizer, context: str, message: str, max_len: int, bos: int, eos: int) -> list[int]:
    """[bos] instruction + record + message [eos], cutting the record first when the window is tight.

    `tokenizer` needs only `.encode(text) -> list[int]`, without special tokens; HFTokenizer adapts a
    transformers tokenizer to that. Long messages keep their END: in v0.6 a 112-token left-truncation
    dropped the edit off the end of a long prompt and the request was allowed.
    """
    prefix = tokenizer.encode(query_prefix())
    budget = max_len - len(prefix) - 2
    if budget < 1:
        raise ValueError(f"max_len {max_len} leaves no room for the instruction prefix")
    msg = tokenizer.encode(message)
    cap = max(1, budget - CONTEXT_RESERVE)
    if len(msg) > cap:
        msg = msg[-cap:]
    ctx = tokenizer.encode(context)[: max(0, budget - len(msg))]
    return [bos, *prefix, *ctx, *msg, eos]


def final_decision(plain: str, sentinel: str) -> str:
    """The story-blind decision, which the raw-text pass may raise and may never lower.

    This is story monotonicity as code: no justification wrapped around a request can make it more
    acceptable than the same request asked plainly.
    """
    for name, value in (("plain", plain), ("sentinel", sentinel)):
        if value not in RISK:
            raise ValueError(f"{name} decision {value!r} is not one of {DECISION}")
    return plain if RISK[plain] >= RISK[sentinel] else sentinel


class HFTokenizer:
    """Adapts a transformers tokenizer to the one method build_ids needs."""

    def __init__(self, tok):
        self.tok = tok

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]


class FileTokenizer:
    """Adapts a `tokenizers` tokenizer loaded from an exported tokenizer.json."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def resolve_path(path: str | Path | None = None) -> Path:
    """An explicit path or AEGIS_V07_MODEL wins, then the user's cache directory."""
    if path or os.environ.get("AEGIS_V07_MODEL"):
        return Path(path or os.environ["AEGIS_V07_MODEL"])
    return DEFAULT_DIR


@dataclass
class HarrierAnalysis:
    """What the two passes say. `decision` is the one to act on."""

    decision: str
    plain: dict[str, float]
    sentinel: dict[str, float]
    intent: dict[str, float]
    purpose: dict[str, float]
    distinguishable: dict[str, float]
    fact_class: dict[str, float]

    @property
    def p_block(self) -> float:
        """The story-blind probability of BLOCK, which is what calibration is fitted on."""
        return self.plain["BLOCK"]

    @property
    def top_intent(self) -> tuple[str, float]:
        k = max(self.intent, key=self.intent.get)
        return k, self.intent[k]


class HarrierModel:
    """The fine-tuned v0.7 model, run on onnxruntime with no PyTorch."""

    name = "harrier"

    def __init__(self, path: str | Path, quantized: bool = True, threads: int | None = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - a core dependency
            raise GuardUnavailable(f"the v0.7 model needs onnxruntime: {exc}") from exc
        self.path = Path(path)
        cfg = self.path / "config.json"
        if not cfg.exists():
            raise GuardUnavailable(f"no v0.7 model at {self.path}")
        self.config = json.loads(cfg.read_text(encoding="utf-8"))
        int8 = self.path / "model.int8.onnx"
        onnx_file = int8 if quantized and int8.exists() else self.path / "model.onnx"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        self.session = ort.InferenceSession(str(onnx_file), opts, providers=["CPUExecutionProvider"])
        self.tokenizer = FileTokenizer(self.path / "tokenizer.json")
        self._finish()

    @classmethod
    def from_parts(cls, session, tokenizer, config: dict) -> HarrierModel:
        """Build from an already-made session and tokenizer, so the plumbing is testable."""
        self = cls.__new__(cls)
        self.path = Path(config.get("version", "in-memory"))
        self.session, self.tokenizer, self.config = session, tokenizer, config
        self._finish()
        return self

    def _finish(self) -> None:
        self.labels: dict[str, list[str]] = self.config["labels"]
        self.max_len = int(self.config["max_len"])
        self.bos = int(self.config["bos_token_id"])
        self.eos = int(self.config["eos_token_id"])
        self.version = self.config.get("version", self.path.name)
        self.outputs = [o.name for o in self.session.get_outputs()]

    @classmethod
    def load(cls, path: str | Path | None = None, **kw) -> HarrierModel:
        target = resolve_path(path).resolve()
        key = (str(target), kw.get("quantized", True), kw.get("threads"))
        if key not in _LOADED:
            _LOADED[key] = cls(target, **kw)
        return _LOADED[key]

    @staticmethod
    def available(path: str | Path | None = None) -> bool:
        return (resolve_path(path) / "config.json").exists()

    def _pass(self, context: str, message: str) -> dict[str, dict[str, float]]:
        ids = np.array([build_ids(self.tokenizer, context, message, self.max_len, self.bos, self.eos)],
                       dtype=np.int64)
        raw = dict(zip(self.outputs, self.session.run(None, {"input_ids": ids,
                                                             "attention_mask": np.ones_like(ids)})))
        return {h: dict(zip(self.labels[h], _softmax(raw[h][0]).tolist())) for h in HEADS}

    def analyze(self, context: str, prompt: str, plain: str | None = None) -> HarrierAnalysis:
        """Both passes. `plain` is the message with its story cut; without one, the raw message is used
        for both, which is weaker but never unsafe - the sentinel can still only raise the decision."""
        full = self._pass(context, prompt)
        story_blind = self._pass(context, plain if plain is not None else prompt)
        decision = story_blind["decision"]
        sentinel = full["sentinel"]
        return HarrierAnalysis(
            decision=final_decision(max(decision, key=decision.get), max(sentinel, key=sentinel.get)),
            plain=decision,
            sentinel=sentinel,
            intent=full["intent"],
            purpose=full["purpose"],
            distinguishable=full["distinguishable"],
            fact_class=story_blind["fact_class"],
        )


_LOADED: dict[tuple, HarrierModel] = {}
