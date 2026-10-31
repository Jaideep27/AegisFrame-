"""aegisframe command line.

    aegisframe check --prompt "..." --image slip.png [--explain] [--json]
    aegisframe verify --input slip.png --output edited.png --prompt "..."
    aegisframe policy validate policy.yaml
    aegisframe config show | config validate aegis.yaml
    aegisframe agents
    aegisframe review list|approve|reject --queue reviews.db
    aegisframe calibrate --results results/aegisframe.jsonl --alpha 0.01
    aegisframe bench [reframebench options]
    aegisframe serve --config aegis.yaml --port 8080

Exit codes for check/verify: 0 ALLOW, 2 REVIEW, 3 BLOCK, 1 error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXIT = {"ALLOW": 0, "REVIEW": 2, "BLOCK": 3}


def _guard(args):
    from .config import AegisConfig
    from .guard import Guard

    cfg = AegisConfig.load(args.config) if getattr(args, "config", None) else AegisConfig()
    if getattr(args, "ocr", None):
        cfg.grounding.ocr = args.ocr
    return Guard.from_config(cfg)


def _identity(args):
    if not getattr(args, "verified_role", None):
        return None
    from .identity import VerifiedIdentity

    return VerifiedIdentity(subject="cli", roles=args.verified_role, issuer="cli-testing")


def _print_decision(d, args) -> None:
    from .audit import decision_record

    if args.json:
        print(json.dumps(decision_record(d, "", "text"), ensure_ascii=False, indent=2))
        return
    print(f"{d.action.name}  rule={d.rule}  severity={d.severity.name.lower()}")
    print(f"canonical: {d.canonical_instruction}")
    if args.explain:
        for r in d.reasons:
            print(f"  reason: {r}")
        for c in d.claims:
            print(f"  claim ({c.kind}, no credit): {c.text}")
        if d.council is not None:
            v = d.council
            print(f"  council: posterior={v.posterior:.3f} (claim-blind {v.deliberation_posterior:.3f}) "
                  f"agents={v.agents_used}/{v.agents_total} rounds={v.rounds} early_stop={v.stopped_early}")
            for b in v.beliefs:
                print(f"    {b.agent:16} p={b.p_harm:.2f} c={b.confidence:.2f} {b.reason_code} {b.evidence or ''}")
            if v.purpose is not None:
                print(f"    {'purpose':16} p={v.purpose.p_harm:.2f} c={v.purpose.confidence:.2f} {v.purpose.reason_code}")


def cmd_check(args) -> int:
    guard = _guard(args)
    image = Path(args.image).read_bytes() if args.image else None
    d = guard.check_request(args.prompt, image=image, verified_identity=_identity(args), session_id=args.session)
    _print_decision(d, args)
    return EXIT[d.action.name]


def cmd_verify(args) -> int:
    guard = _guard(args)
    original = Path(args.input).read_bytes() if args.input else None
    d = guard.check_request(args.prompt, image=original, verified_identity=_identity(args))
    v = guard.check_output(original, Path(args.output).read_bytes(), d)
    print(f"{v.action.name}  (request was {d.action.name})")
    for r in v.reasons:
        print(f"  {r}")
    return EXIT[v.action.name]


def cmd_policy(args) -> int:
    from .policy import Policy

    try:
        p = Policy.from_file(args.path)
    except Exception as exc:  # noqa: BLE001 - report any load problem to the user
        print(f"invalid policy: {exc}")
        return 1
    print(f"ok: {len(p.rules)} rules ({', '.join(r.id for r in p.rules)}), {len(p.overrides)} overrides, "
          f"{len(p.unlocks)} verified-role unlocks")
    return 0


def cmd_config(args) -> int:
    import yaml

    from .config import AegisConfig, ConfigError

    if args.action == "show":
        print(yaml.safe_dump(AegisConfig().model_dump(mode="json"), sort_keys=False), end="")
        return 0
    try:
        AegisConfig.load(args.path)
    except (ConfigError, OSError) as exc:
        print(f"invalid config: {exc}")
        return 1
    print("ok")
    return 0


def cmd_agents(args) -> int:
    from .council import agents as _a, llm_agents as _l  # noqa: F401  (register built-ins)
    from .council.registry import _REGISTRY

    for name in sorted(_REGISTRY):
        cls = _REGISTRY[name]
        doc = (cls.__doc__ or "").strip().splitlines()
        llm = " [needs provider]" if getattr(cls, "requires_provider", False) else ""
        print(f"{name:16}{llm:18} {doc[0] if doc else ''}")
    return 0


def cmd_review(args) -> int:
    from .review import ReviewQueue

    q = ReviewQueue(args.queue)
    if args.action == "list":
        for item in q.pending():
            print(f"{item.id}  {item.rule}  {item.severity}  {item.canonical_instruction}")
        return 0
    if not args.id:
        print("review id required")
        return 1
    try:
        if args.action == "approve":
            print(f"approved {args.id}; one-time token:")
            print(q.approve(args.id, reviewer=args.reviewer, note=args.note))
        else:
            q.reject(args.id, reviewer=args.reviewer, note=args.note)
            print(f"rejected {args.id}")
    except ValueError as exc:
        print(str(exc))
        return 1
    return 0


def cmd_calibrate(args) -> int:
    from .calibrate import calibrate_file

    result = calibrate_file(args.results, alpha=args.alpha, delta=args.delta, out=args.out)
    print(json.dumps(result, indent=2))
    return 0


def cmd_model(args) -> int:
    import shutil

    from .local.runtime import DEFAULT_DIR, resolve_path

    if args.action == "install":
        src, dest = Path(args.source), Path(args.to or DEFAULT_DIR)
        needed = ["config.json", "tokenizer.json", "model.int8.onnx"] + (["model.onnx"] if args.with_fp32 else [])
        missing = [n for n in needed if not (src / n).exists()]
        if missing:
            print(f"not a built model directory, missing: {missing}")
            return 1
        dest.mkdir(parents=True, exist_ok=True)
        for name in needed:
            shutil.copy2(src / name, dest / name)
        print(f"installed local model to {dest}")
        return 0
    path = resolve_path(args.path)
    if not (path / "config.json").exists():
        print(f"no local model installed at {path}")
        return 1
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    size = sum(f.stat().st_size for f in path.iterdir() if f.is_file()) / 1e6
    print(f"local model installed at {path} ({size:.0f} MB), encoder {cfg.get('encoder')}")
    metrics = cfg.get("metrics") or {}
    for split in ("val", "test_heldout"):
        if split in metrics:
            m = metrics[split]
            print(f"  {split}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float)))
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("the proxy needs the server extra: pip install 'aegisframe[server]'")
        return 1
    from .server import create_app

    uvicorn.run(create_app(_guard(args)), host=args.host, port=args.port)
    return 0


def cmd_mcp(args) -> int:
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("the MCP server needs the mcp extra: pip install 'aegisframe[mcp]'")
        return 1
    from .harrier_guard import HarrierGuard
    from .mcp_server import build_server

    guard = HarrierGuard.load(args.model)
    build_server(guard).run(transport="streamable-http" if args.http else "stdio")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    ap = argparse.ArgumentParser(prog="aegisframe", description="Guardrail for AI image editing.")
    ap.add_argument("--version", action="version", version=f"aegisframe {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", help="aegis.yaml")
        p.add_argument("--ocr", choices=["rapid", "none"], help="override grounding.ocr")
        p.add_argument("--verified-role", action="append", help="TESTING ONLY: act as a verified SSO role")

    p = sub.add_parser("check", help="decide on one request")
    p.add_argument("--prompt", required=True)
    p.add_argument("--image")
    p.add_argument("--session", default="cli")
    p.add_argument("--json", action="store_true")
    p.add_argument("--explain", action="store_true")
    common(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("verify", help="check a generated image against the original")
    p.add_argument("--input")
    p.add_argument("--output", required=True)
    p.add_argument("--prompt", required=True)
    common(p)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("policy", help="policy tools")
    p.add_argument("action", choices=["validate"])
    p.add_argument("path")
    p.set_defaults(fn=cmd_policy)

    p = sub.add_parser("config", help="config tools")
    p.add_argument("action", choices=["show", "validate"])
    p.add_argument("path", nargs="?")
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser("agents", help="list council agents")
    p.set_defaults(fn=cmd_agents)

    p = sub.add_parser("review", help="human review queue")
    p.add_argument("action", choices=["list", "approve", "reject"])
    p.add_argument("id", nargs="?")
    p.add_argument("--queue", required=True)
    p.add_argument("--reviewer", default="cli")
    p.add_argument("--note")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("calibrate", help="choose council thresholds from labelled results")
    p.add_argument("--results", required=True, help="per-case JSONL from reframebench.run")
    p.add_argument("--alpha", type=float, default=0.01, help="target harmful-allow rate")
    p.add_argument("--delta", type=float, default=0.05, help="1 - confidence")
    p.add_argument("--out", default="calibration.json")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("model", help="the local (no-API) model: install a built model, show info")
    p.add_argument("action", choices=["install", "info"])
    p.add_argument("source", nargs="?", help="built model directory (install)")
    p.add_argument("--to", help="install destination (default ~/.cache/aegisframe/models/aegis-local)")
    p.add_argument("--path", help="model directory to inspect (info)")
    p.add_argument("--with-fp32", action="store_true", help="also copy the full-precision graph")
    p.set_defaults(fn=cmd_model)

    p = sub.add_parser("serve", help="run the OpenAI-compatible guarding proxy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    common(p)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("mcp", help="run the v0.7 guard as an MCP server (stdio or HTTP)")
    p.add_argument("--model", default=None, help="path to the exported v0.7 model directory")
    p.add_argument("--http", action="store_true", help="serve over streamable HTTP instead of stdio")
    p.set_defaults(fn=cmd_mcp)

    p = sub.add_parser("bench", help="run ReframeBench (options are passed through)", add_help=False)
    p.set_defaults(fn=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["bench"]:
        from reframebench.run import main as bench_main

        bench_main(argv[1:])
        return 0
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
