"""Run the local AegisFrame model with onnxruntime + tokenizers (no PyTorch, no API).

    model = LocalModel.load()               # a path, AEGIS_LOCAL_MODEL, ~/.cache/aegisframe/models/aegis-local,
                                            # or the model bundled with the package
    full  = model.analyze_full(context, prompt)        # separator spans, intent, twists, purpose
    plain = model.analyze_plain(context, canonical)    # decision, severity, pattern, misuse, benign need
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import GuardUnavailable

DEFAULT_DIR = Path.home() / ".cache" / "aegisframe" / "models" / "aegis-local"
BUNDLED_DIR = Path(__file__).resolve().parent / "model"
_LOADED: dict[tuple, "LocalModel"] = {}


def resolve_path(path: str | Path | None = None) -> Path:
    """An explicit path or AEGIS_LOCAL_MODEL wins; then a user-installed model; then the bundled one."""
    if path or os.environ.get("AEGIS_LOCAL_MODEL"):
        return Path(path or os.environ["AEGIS_LOCAL_MODEL"])
    return DEFAULT_DIR if (DEFAULT_DIR / "config.json").exists() else BUNDLED_DIR


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def decode_spans(tag_ids: list[int], offsets: list[tuple[int, int]], seq_ids: list[int | None],
                 labels: list[str], probs: list[float] | None = None) -> list[tuple[int, int, str, float]]:
    """BIO tags on the second sequence (the prompt) -> character spans (start, end, TAG, confidence)."""
    spans: list[list] = []
    for i, tid in enumerate(tag_ids):
        if seq_ids[i] != 1:
            continue
        s, e = offsets[i]
        if s == e:
            continue
        tag = labels[tid]
        conf = probs[i] if probs else 1.0
        if tag == "O":
            continue
        kind = tag[2:]
        if spans and spans[-1][2] == kind and (tag.startswith("I-") or s <= spans[-1][1] + 1) and s - spans[-1][1] <= 1:
            spans[-1][1] = e
            spans[-1][3] = min(spans[-1][3], conf)
        else:
            spans.append([s, e, kind, conf])
    return [tuple(x) for x in spans]


@dataclass
class FullAnalysis:
    spans: list[tuple[int, int, str, float]]
    intent: dict[str, float]
    twist: dict[str, float]
    purpose: float

    @property
    def top_intent(self) -> tuple[str, float]:
        k = max(self.intent, key=self.intent.get)
        return k, self.intent[k]


@dataclass
class PlainAnalysis:
    decision: dict[str, float]
    field_class: dict[str, float]
    pattern: dict[str, float]
    misuse: dict[str, float]
    benign_need: dict[str, float]


class LocalModel:
    name = "local"

    def __init__(self, path: str | Path, quantized: bool = True, threads: int | None = None):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - both are core dependencies
            raise GuardUnavailable(f"local model needs onnxruntime and tokenizers: {exc}") from exc
        self.path = Path(path)
        cfg_path = self.path / "config.json"
        if not cfg_path.exists():
            raise GuardUnavailable(f"no local model at {self.path} (run `aegisframe model install <dir>`)")
        self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.labels: dict[str, list[str]] = self.config["labels"]
        onnx_file = self.path / ("model.int8.onnx" if quantized and (self.path / "model.int8.onnx").exists() else "model.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        self.session = ort.InferenceSession(str(onnx_file), opts, providers=["CPUExecutionProvider"])
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self.tokenizer.no_padding()
        self._cache: OrderedDict[tuple, dict] = OrderedDict()
        self._lock = threading.Lock()
        self.version = self.config.get("version", self.path.name)

    @classmethod
    def load(cls, path: str | Path | None = None, **kw) -> LocalModel:
        """Loaded models are shared (thread-safe ONNX sessions): creating many Guards stays cheap."""
        target = resolve_path(path).resolve()
        key = (str(target), kw.get("quantized", True), kw.get("threads"))
        if key not in _LOADED:
            _LOADED[key] = cls(target, **kw)
        return _LOADED[key]

    @staticmethod
    def available(path: str | Path | None = None) -> bool:
        return (resolve_path(path) / "config.json").exists()

    def _run(self, context: str, text: str, max_len: int) -> tuple[dict, object]:
        # the cache and the tokenizer's truncation setting are shared state: one lock keeps
        # concurrent requests (e.g. the proxy's thread pool) from interleaving them
        with self._lock:
            return self._run_locked(context, text, max_len)

    def _run_locked(self, context: str, text: str, max_len: int) -> tuple[dict, object]:
        key = (context, text, max_len)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        self.tokenizer.enable_truncation(max_length=max_len, strategy="only_first")
        enc = self.tokenizer.encode(context, text)
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        result = dict(zip(self.outputs, self.session.run(None, feeds)))
        self._cache[key] = (result, enc)
        if len(self._cache) > 512:
            self._cache.popitem(last=False)
        return result, enc

    def _probs(self, out: dict, head: str) -> dict[str, float]:
        return dict(zip(self.labels[head], _softmax(out[head][0]).tolist()))

    def analyze_full(self, context: str, prompt: str) -> FullAnalysis:
        out, enc = self._run(context, prompt, self.config.get("max_full", 112))
        tag_probs = _softmax(out["tags"][0])
        spans = decode_spans(tag_probs.argmax(-1).tolist(), enc.offsets, enc.sequence_ids, self.labels["tags"],
                             tag_probs.max(-1).tolist())
        return FullAnalysis(
            spans=spans,
            intent=self._probs(out, "intent"),
            twist=dict(zip(self.labels["twist"], _sigmoid(out["twist"][0]).tolist())),
            purpose=float(_sigmoid(out["purpose"][0])[0]),
        )

    def analyze_plain(self, context: str, canonical: str) -> PlainAnalysis:
        out, _ = self._run(context, canonical, self.config.get("max_plain", 96))
        return PlainAnalysis(**{h: self._probs(out, h) for h in ("decision", "field_class", "pattern", "misuse", "benign_need")})
