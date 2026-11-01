"""An MCP server that puts AegisFrame's v0.7 guard in front of an image-editing tool.

Tools:
  check_request(prompt, context?, session_id?, request_id?) -> the decision, reasons and the agent trace
  verify_output(prompt, context?)                           -> alias kept for symmetry with the plan
  explain_decision(request_id)                              -> recover a prior check by id

An agent is expected to call check_request before any document image edit and to refuse on BLOCK. The tool
logic lives in build_tools() as plain functions, fully testable without the MCP SDK; build_server() wraps
them in an MCPServer (the 2.x successor to FastMCP) for stdio or HTTP transport.

    aegisframe mcp                      # stdio, for Claude Desktop / Claude Code / Codex
    aegisframe mcp --http --port 8008   # streamable HTTP with a bearer token, for remote use
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from typing import Callable

INSTRUCTIONS = (
    "Always call check_request before editing any image of a document, and never proceed when the "
    "decision is BLOCK. REVIEW means a human must approve first. The trace explains which agents raised "
    "the risk."
)


def _verdict_json(v, request_id: str | None = None) -> str:
    return json.dumps({
        "decision": v.decision,
        "p": {k: round(x, 4) for k, x in v.p.items()},
        "p_block": round(v.p_block, 4),
        "intent": v.intent,
        "reason": v.reason,
        "value_changes": [{"old": c.old, "new": c.new, "kind": c.kind, "field": c.field}
                          for c in v.value_changes],
        "trace": v.trace,
        "request_id": request_id,
    })


def build_tools(guard) -> dict[str, Callable]:
    """Return the tool functions bound to a guard. Pure Python; no MCP SDK needed."""
    history: dict[str, list[str]] = defaultdict(list)
    last: "OrderedDict[str, object]" = OrderedDict()

    def check_request(prompt: str, context: str = "", session_id: str = "",
                      request_id: str = "", image_b64: str = "") -> str:
        hist = history[session_id] if session_id else None
        hist_arg = list(hist) if hist else None
        if image_b64:
            import base64

            v = guard.check_image(base64.b64decode(image_b64), prompt, history=hist_arg)
        else:
            v = guard.check(prompt, context=context, history=hist_arg)
        if session_id:
            history[session_id].append(prompt)
        if request_id:
            last[request_id] = v
            if len(last) > 256:
                last.popitem(last=False)
        return _verdict_json(v, request_id or None)

    def verify_output(prompt: str, context: str = "") -> str:
        return _verdict_json(guard.check(prompt, context=context))

    def explain_decision(request_id: str) -> str:
        v = last.get(request_id)
        if v is None:
            return json.dumps({"error": f"no decision recorded for request_id {request_id!r}"})
        return _verdict_json(v, request_id)

    return {"check_request": check_request, "verify_output": verify_output,
            "explain_decision": explain_decision}


def build_server(guard, name: str = "aegisframe"):
    """Wrap the tools in an MCPServer (lazily imported, so the SDK is only needed to serve)."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name, instructions=INSTRUCTIONS)
    tools = build_tools(guard)
    for tool_name, fn in tools.items():
        server.add_tool(fn, name=tool_name)
    return server


def serve(model_path: str | None = None, http: bool = False, port: int = 8008) -> None:  # pragma: no cover
    from .harrier_guard import HarrierGuard

    guard = HarrierGuard.load(model_path)
    server = build_server(guard)
    server.run(transport="streamable-http" if http else "stdio")
