"""HarrierGuard.load() with no model present must download it, not just error.

This is the published-package UX: `pip install aegisframe` then `HarrierGuard.load()`. A fake local
file source stands in for Hugging Face so no network is touched.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from aegisframe import harrier_guard
from aegisframe.local import harrier
from aegisframe.local.download import ModelManifest


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_model_source(tmp_path):
    """A directory of fake model files + a manifest pointing at it, as if it were the HF repo."""
    import io

    import numpy as np

    src = tmp_path / "remote"
    (src / "fusion").mkdir(parents=True)
    buf = io.BytesIO()
    np.savez(buf, harmful=np.zeros((1, 2)), benign=np.zeros((1, 2)), tau=np.array(0.1), k=np.array(1))
    files = {
        "config.json": b'{"ok": 1}',
        "model.int8.onnx": b"ONNX",
        "tokenizer.json": b"{}",
        "fusion/casebank.npz": buf.getvalue(),
        "fusion/fusion.json": b'{"f_default": 0, "prior": {"ALLOW": 0.45, "REVIEW": 0.1, "BLOCK": 0.45}}',
    }
    for name, data in files.items():
        (src / name).write_bytes(data)
    # only the large binaries are downloaded; config and fusion files ship inside the package
    big = ("model.int8.onnx", "tokenizer.json")
    manifest = ModelManifest(version="t", base_url=src.as_uri() + "/",
                             files={n: _sha(files[n]) for n in big})
    return manifest


def test_load_downloads_the_model_when_the_cache_is_empty(tmp_path, monkeypatch):
    manifest = _make_model_source(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setattr(harrier, "DEFAULT_DIR", cache)
    monkeypatch.setattr(harrier_guard, "_bundled_manifest", lambda: manifest)

    built = {}

    class FakeModel:
        name = "harrier"

        def __init__(self, path, **kw):
            built["path"] = str(path)
            self.path = path

        @classmethod
        def load(cls, path=None, **kw):
            return cls(harrier.resolve_path(path), **kw)

        @staticmethod
        def available(path=None):
            return (harrier.resolve_path(path) / "config.json").exists()

    monkeypatch.setattr(harrier_guard, "HarrierModel", FakeModel)

    harrier_guard.HarrierGuard.load()
    # the big files came from the (fake) remote ...
    assert (cache / "model.int8.onnx").read_bytes() == b"ONNX"
    assert (cache / "tokenizer.json").exists()
    # ... and the small ones from the package itself
    cfg = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    assert "heads" in cfg and cfg["max_len"] == 128
    assert (cache / "fusion" / "casebank.npz").exists()
    assert (cache / "fusion" / "fusion.json").exists()
    assert built["path"] == str(cache)


def test_an_explicit_path_is_not_auto_downloaded(tmp_path, monkeypatch):
    """Passing a path means 'use this exact model'; a missing one should error, not download."""
    monkeypatch.setattr(harrier_guard, "_bundled_manifest",
                        lambda: (_ for _ in ()).throw(AssertionError("must not download")))
    from aegisframe.errors import GuardUnavailable

    with pytest.raises(GuardUnavailable):
        harrier_guard.HarrierGuard.load(tmp_path / "nope")
