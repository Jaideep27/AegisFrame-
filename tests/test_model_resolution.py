"""Where the local model is loaded from: explicit path, env var, user install, then the bundled copy."""

from aegisframe.local import runtime
from aegisframe.local.runtime import BUNDLED_DIR, resolve_path


def test_the_package_ships_a_model():
    assert {"config.json", "tokenizer.json", "model.int8.onnx"} <= {p.name for p in BUNDLED_DIR.iterdir()}


def test_explicit_path_and_env_var_take_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("AEGIS_LOCAL_MODEL", str(tmp_path / "env"))
    assert resolve_path(tmp_path / "explicit") == tmp_path / "explicit"
    assert resolve_path() == tmp_path / "env"


def test_user_install_beats_the_bundled_model(monkeypatch, tmp_path):
    monkeypatch.delenv("AEGIS_LOCAL_MODEL", raising=False)
    monkeypatch.setattr(runtime, "DEFAULT_DIR", tmp_path)
    assert resolve_path() == BUNDLED_DIR
    (tmp_path / "config.json").write_text("{}")
    assert resolve_path() == tmp_path
