"""Providers that drive the official coding CLIs, for local testing on a personal subscription.

    providers:
      claude: {kind: claude_cli, model: sonnet}     # `claude -p` (Claude Code, Pro/Max login)
      gpt:    {kind: codex_cli}                     # `codex exec` (Codex CLI, ChatGPT login)

These are for internal testing and research only: they run one CLI session per call (seconds),
are subject to your plan's limits, and a personal subscription must not back a product that
other people use. Use API keys (gemini / openai_compat providers) in production.
User text is passed on stdin, never through a shell.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .base import ProviderError, parse_json_reply


def _tmp_png(image: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".png", prefix="aegis_")
    with os.fdopen(fd, "wb") as f:
        f.write(image)
    return path


class ClaudeCLIProvider:
    def __init__(self, model: str | None = None, binary: str = "claude", timeout: float = 180.0):
        self.model, self.binary, self.timeout = model, binary, timeout
        self.name = f"claude-cli:{model or 'default'}"

    def generate_json(self, system: str, user: str, image: bytes | None = None) -> dict:
        cmd = [self.binary, "-p", "--output-format", "json", "--append-system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        img = None
        if image is not None:
            img = _tmp_png(image)
            cmd += ["--allowedTools", "Read"]
            user = f"{user}\n\nThe image is at: {img} (read it with the Read tool)."
        try:
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8",
                                  timeout=self.timeout)
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
        finally:
            if img:
                Path(img).unlink(missing_ok=True)
        if proc.returncode != 0:
            raise ProviderError(f"{self.name}: exit {proc.returncode}: {proc.stderr[:300]}")
        try:
            text = json.loads(proc.stdout).get("result", "")
        except json.JSONDecodeError:
            text = proc.stdout
        return parse_json_reply(text)


class CodexCLIProvider:
    def __init__(self, model: str | None = None, binary: str = "codex", timeout: float = 240.0):
        self.model, self.binary, self.timeout = model, binary, timeout
        self.name = f"codex-cli:{model or 'default'}"

    def generate_json(self, system: str, user: str, image: bytes | None = None) -> dict:
        fd, out = tempfile.mkstemp(suffix=".txt", prefix="aegis_codex_")
        os.close(fd)
        cmd = [self.binary, "exec", "--skip-git-repo-check", "-s", "read-only", "--output-last-message", out]
        if self.model:
            cmd += ["-m", self.model]
        img = _tmp_png(image) if image is not None else None
        if img:
            cmd += ["-i", img]
        cmd.append("-")  # read the prompt from stdin
        prompt = f"{system}\n\n{user}\n\nAnswer with the JSON object only. Do not run any commands."
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                  timeout=self.timeout)
            if proc.returncode != 0:
                raise ProviderError(f"{self.name}: exit {proc.returncode}: {proc.stderr[:300]}")
            return parse_json_reply(Path(out).read_text(encoding="utf-8"))
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
        finally:
            Path(out).unlink(missing_ok=True)
            if img:
                Path(img).unlink(missing_ok=True)
