"""The LangGraph integration must give the same decision as HarrierGuard.check on every input."""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
from aegisframe.integrations.langgraph import build_guard_graph, guard_node  # noqa: E402


class FakeV:
    def __init__(self, decision):
        self.decision = decision
        self.p = {"ALLOW": 0.1, "REVIEW": 0.1, "BLOCK": 0.8}
        self.p_block = 0.8
        self.intent = "alter_fact"
        self.reason = "r"
        self.trace = []
        self.value_changes = []


class FakeGuard:
    def check(self, prompt, context="", history=None):
        return FakeV("BLOCK" if "change" in prompt else "ALLOW")


def test_the_node_writes_the_guard_decision_into_the_state():
    node = guard_node(FakeGuard())
    out = node({"prompt": "change the net pay", "context": "Record:\n"})
    assert out["decision"] == "BLOCK"
    assert out["guard"]["p_block"] == 0.8


def test_the_graph_matches_the_guard_on_every_prompt():
    guard = FakeGuard()
    graph = build_guard_graph(guard)
    for prompt in ("change the total", "what is the total?", "blur the account number", "change it again"):
        state = graph.invoke({"prompt": prompt, "context": "Record:\n"})
        assert state["decision"] == guard.check(prompt).decision


def test_a_blocked_request_routes_to_refuse():
    graph = build_guard_graph(FakeGuard())
    assert graph.invoke({"prompt": "change the net pay", "context": ""})["route"] == "refuse"
    assert graph.invoke({"prompt": "what is the net pay?", "context": ""})["route"] == "proceed"
