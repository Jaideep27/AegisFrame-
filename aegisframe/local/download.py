"""Fetch a v0.7 model on first use and verify every file against a SHA-256 manifest.

The int8 ONNX model plus tokenizer is ~305 MB, too large to bundle in the wheel, so it is hosted (e.g. a
GitHub release) and downloaded once into the user's cache. Each file's hash is checked; a mismatch is a
hard error, never a silent accept. A file already present with the right hash is not re-fetched.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelManifest:
    version: str
    base_url: str  # files are fetched from base_url + filename
    files: dict[str, str] = field(default_factory=dict)  # filename -> expected sha256

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"version": self.version, "base_url": self.base_url,
                                          "files": self.files}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ModelManifest:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(version=d["version"], base_url=d["base_url"], files=d["files"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(manifest: ModelManifest, dest: str | Path) -> Path:
    """Make sure every manifest file is in `dest` with the right hash, downloading the missing ones.

    Returns the destination directory. Raises ValueError on any hash mismatch.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest.files.items():
        target = dest / name
        if target.exists() and _sha256(target) == expected:
            continue
        url = manifest.base_url + name
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - url comes from a trusted manifest
        got = _sha256(tmp)
        if got != expected:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"{name}: sha256 mismatch (expected {expected[:12]}…, got {got[:12]}…)")
        tmp.replace(target)
    return dest
