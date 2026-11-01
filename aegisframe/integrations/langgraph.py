"""LangGraph integration: drop the v0.7 guard in front of an image-editing node.

A LangGraph `StateGraph` is a blackboard already - shared typed state and conditional edges - so the guard
is one node plus a routing edge. `guard_node` is the node function; `build_guard_graph` wires it with a
conditional edge to `proceed` or `refuse`. The graph's decision is identical to `HarrierGuard.check` by
construction: the node calls `check` and copies its verdict into the state.

    from aegisframe import HarrierGuard
    from aegisframe.integrations.langgraph import build_guard_graph

    graph = build_guard_graph(HarrierGuard.load("models/aegis-v07-r1-onnx"))
    state = graph.invoke({"prompt": "...", "context": "Record:\n..."})
    state["decision"], state["route"]   # "BLOCK", "refuse"
"""

from __future__ import annotations

from typing import Any, Callable
from typing_extensions import TypedDict


def guard_node(guard) -> Callable[[dict], dict]:
    """A node that runs the guard on state['prompt']/['context'] and records the verdict."""

    def node(state: dict) -> dict:
        v = guard.check(state.get("prompt", ""), context=state.get("context", ""),
                        history=state.get("history"))
        return {
            "decision": v.decision,
            "route": "proceed" if v.decision == "ALLOW" else "refuse",
            "guard": {"decision": v.decision, "p": v.p, "p_block": v.p_block, "intent": v.intent,
                      "reason": v.reason, "trace": v.trace},
        }

    return node


def build_guard_graph(guard, proceed: Callable[[dict], dict] | None = None,
                      refuse: Callable[[dict], dict] | None = None):
    """A compiled graph: guard -> (proceed | refuse). Pass your own terminal nodes, or use the defaults.

    REVIEW and BLOCK both route to `refuse`; REVIEW means a human approval step belongs on that edge
    (LangGraph's `interrupt()` fits there).
    """
    from langgraph.graph import END, START, StateGraph

    class GuardState(TypedDict, total=False):
        prompt: str
        context: str
        history: list
        decision: str
        route: str
        guard: dict
        result: str

    def _default_proceed(state: dict) -> dict:
        return {"result": "edit allowed"}

    def _default_refuse(state: dict) -> dict:
        return {"result": f"edit {state['decision'].lower()}ed"}

    graph: Any = StateGraph(GuardState)
    graph.add_node("guard", guard_node(guard))
    graph.add_node("proceed", proceed or _default_proceed)
    graph.add_node("refuse", refuse or _default_refuse)
    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", lambda s: s["route"], {"proceed": "proceed", "refuse": "refuse"})
    graph.add_edge("proceed", END)
    graph.add_edge("refuse", END)
    return graph.compile()
