"""Download-on-first-use with a SHA-256 check: fetch the model files only if absent, verify every one."""

from __future__ import annotations

import hashlib
import json

import pytest

from aegisframe.local.download import ModelManifest, ensure_model


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    files = {"config.json": b'{"format": 2}', "model.int8.onnx": b"ONNXDATA", "tokenizer.json": b"{}"}
    for name, data in files.items():
        (src / name).write_bytes(data)
    manifest = ModelManifest(
        version="test-1",
        base_url=src.as_uri() + "/",
        files={name: _sha(data) for name, data in files.items()},
    )
    return src, manifest, files


def test_a_missing_model_is_downloaded_and_verified(tmp_path):
    _, manifest, files = write_source(tmp_path)
    dest = tmp_path / "dest"
    out = ensure_model(manifest, dest)
    for name, data in files.items():
        assert (out / name).read_bytes() == data


def test_an_already_present_model_is_not_refetched(tmp_path):
    src, manifest, files = write_source(tmp_path)
    dest = tmp_path / "dest"
    ensure_model(manifest, dest)
    # remove the source; a second call must succeed from the cache without fetching
    for f in src.iterdir():
        f.unlink()
    out = ensure_model(manifest, dest)
    assert (out / "config.json").exists()


def test_a_corrupted_download_is_rejected(tmp_path):
    src, manifest, files = write_source(tmp_path)
    # tamper with the manifest's expected hash so the real file fails verification
    manifest.files["model.int8.onnx"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        ensure_model(manifest, tmp_path / "dest")


def test_a_partial_cache_is_completed(tmp_path):
    src, manifest, files = write_source(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "config.json").write_bytes(files["config.json"])  # one file already there
    out = ensure_model(manifest, dest)
    assert (out / "model.int8.onnx").read_bytes() == files["model.int8.onnx"]


def test_a_manifest_round_trips_through_json(tmp_path):
    _, manifest, _ = write_source(tmp_path)
    path = tmp_path / "manifest.json"
    manifest.save(path)
    again = ModelManifest.load(path)
    assert again.version == manifest.version
    assert again.files == manifest.files
