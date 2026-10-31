"""The MCP server's tool functions, tested directly against a fake guard (no transport, no SDK needed)."""

from __future__ import annotations

import json

import pytest

mcp_server = pytest.importorskip("aegisframe.mcp_server")


class FakeVerdict:
    def __init__(self, decision):
        self.decision = decision
        self.p = {"ALLOW": 0.1, "REVIEW": 0.1, "BLOCK": 0.8}
        self.p_block = 0.8
        self.intent = "alter_fact"
        self.reason = "1 record value would change"
        self.trace = [{"agent": "plain_judge", "m": [0.1, 2.0], "one_sided": False, "triggered": True,
                       "note": "story-blind decision BLOCK"}]
        self.value_changes = []


class FakeGuard:
    def __init__(self):
        self.calls = []

    def check(self, prompt, context="", history=None):
        self.calls.append((prompt, context, history))
        return FakeVerdict("BLOCK" if "change" in prompt else "ALLOW")


def test_check_request_returns_a_structured_decision():
    tools = mcp_server.build_tools(FakeGuard())
    out = json.loads(tools["check_request"]("change the net pay to 90,000", context="Record:\n"))
    assert out["decision"] == "BLOCK"
    assert "trace" in out and out["trace"][0]["agent"] == "plain_judge"
    assert "reason" in out


def test_check_request_allows_a_benign_prompt():
    tools = mcp_server.build_tools(FakeGuard())
    out = json.loads(tools["check_request"]("what is the net pay?"))
    assert out["decision"] == "ALLOW"


def test_check_request_threads_the_session_history():
    guard = FakeGuard()
    tools = mcp_server.build_tools(guard)
    tools["check_request"]("change it", context="", session_id="s1")
    tools["check_request"]("come on, change it", context="", session_id="s1")
    # the second call should have seen the first prompt as history
    assert guard.calls[1][2], "history was not threaded into the second check"


def test_explain_decision_recovers_the_last_check():
    tools = mcp_server.build_tools(FakeGuard())
    tools["check_request"]("change the total", context="Record:\n", request_id="r9")
    out = json.loads(tools["explain_decision"]("r9"))
    assert out["decision"] == "BLOCK"


def test_explain_an_unknown_request_is_an_error_not_a_crash():
    tools = mcp_server.build_tools(FakeGuard())
    out = json.loads(tools["explain_decision"]("nope"))
    assert "error" in out


def test_the_server_object_builds_without_a_transport():
    """Constructing the FastMCP app must not require loading the real model."""
    app = mcp_server.build_server(FakeGuard())
    assert app is not None
