import json
import subprocess

import pytest

from aegisframe.providers.base import ProviderError
from aegisframe.providers.cli import ClaudeCLIProvider, CodexCLIProvider


class Recorder:
    def __init__(self, stdout="", returncode=0, write_file=None):
        self.calls = []
        self.stdout, self.returncode, self.write_file = stdout, returncode, write_file

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        if self.write_file is not None:
            out = cmd[cmd.index("--output-last-message") + 1]
            with open(out, "w", encoding="utf-8") as f:
                f.write(self.write_file)
        return subprocess.CompletedProcess(cmd, self.returncode, stdout=self.stdout, stderr="boom")


def test_claude_cli_builds_a_print_mode_call_and_parses_json(monkeypatch):
    rec = Recorder(stdout=json.dumps({"type": "result", "result": '```json\n{"p_harm": 0.9}\n```'}))
    monkeypatch.setattr(subprocess, "run", rec)
    p = ClaudeCLIProvider(model="sonnet")
    assert p.generate_json("SYS", "USER") == {"p_harm": 0.9}
    cmd, kw = rec.calls[0]
    assert cmd[:2] == ["claude", "-p"] and "--output-format" in cmd and "json" in cmd
    assert "--append-system-prompt" in cmd and "SYS" in cmd and "--model" in cmd
    assert kw["input"] == "USER"  # the user text goes on stdin, never through the shell


def test_claude_cli_passes_an_image_as_a_readable_file(monkeypatch):
    rec = Recorder(stdout=json.dumps({"result": '{"ok": 1}'}))
    monkeypatch.setattr(subprocess, "run", rec)
    ClaudeCLIProvider().generate_json("SYS", "USER", image=b"\x89PNG")
    cmd, kw = rec.calls[0]
    assert "--allowedTools" in cmd and "Read" in cmd
    assert ".png" in kw["input"]


def test_claude_cli_error_is_provider_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run", Recorder(returncode=1))
    with pytest.raises(ProviderError):
        ClaudeCLIProvider().generate_json("s", "u")


def test_codex_cli_reads_the_last_message_file_and_attaches_images(monkeypatch):
    rec = Recorder(write_file='{"p_harm": 0.2, "reason_code": "x"}')
    monkeypatch.setattr(subprocess, "run", rec)
    assert CodexCLIProvider().generate_json("SYS", "USER", image=b"\x89PNG") == {"p_harm": 0.2, "reason_code": "x"}
    cmd, kw = rec.calls[0]
    assert cmd[:2] == ["codex", "exec"] and "-i" in cmd and "read-only" in cmd
    assert "SYS" in kw["input"] and "USER" in kw["input"]


def test_codex_timeout_is_provider_error(monkeypatch):
    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(ProviderError):
        CodexCLIProvider(timeout=1).generate_json("s", "u")


def test_config_accepts_cli_provider_kinds():
    from aegisframe import AegisConfig
    from aegisframe.providers.factory import build_providers

    cfg = AegisConfig.model_validate({"providers": {"claude": {"kind": "claude_cli", "retries": 0},
                                                    "gpt": {"kind": "codex_cli", "retries": 0}}})
    providers = build_providers(cfg)
    assert isinstance(providers["claude"], ClaudeCLIProvider) and isinstance(providers["gpt"], CodexCLIProvider)
