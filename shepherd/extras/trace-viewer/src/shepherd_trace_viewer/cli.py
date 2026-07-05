"""``shepherd-trace-viewer`` CLI for durable trace revisions."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from shepherd_trace_viewer.durable_reader import (
    DurableTraceReadError,
    read_trace_payload_file,
    read_trace_revision,
)
from shepherd_trace_viewer.serde import SchemaVersionError, from_json, to_json
from shepherd_trace_viewer.server import make_server
from shepherd_trace_viewer.trace_store_reader import TraceStoreReadError, read_trace_store_view

_BIND_WARNING_CATEGORIES = (
    "task ids and run refs",
    "trace event payloads, including file paths and model/tool metadata",
    "world head pointers in substrate.transition events",
    "record digests and identity-domain metadata",
)


def _load_view_json(args: argparse.Namespace) -> dict:
    if getattr(args, "trace_payload", None):
        path = Path(args.trace_payload)
        # Accept either a raw durable trace revision payload or an already-built
        # TraceView dataset. The latter keeps server/UI tests cheap.
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version"):
            return to_json(from_json(data))
        return to_json(read_trace_payload_file(path))
    if getattr(args, "trace_store", None):
        selector, selector_value = _trace_store_selector(args)
        return to_json(
            read_trace_store_view(
                args.trace_store,
                selector=selector,
                selector_value=selector_value,
                through=args.through,
                visibility=args.visibility,
                mode_filter=args.mode,
                actor=args.actor,
                trusted_internal=args.trusted_internal,
            )
        )
    if args.trace_head:
        return to_json(read_trace_revision(args.workspace))
    return to_json(read_trace_revision(args.workspace, args.trace_rev))


def _trace_store_selector(args: argparse.Namespace) -> tuple[str, str]:
    selectors = [
        ("cut", args.cut),
        ("owner", args.owner),
        ("causal_root", args.causal_root),
    ]
    selected = [(kind, value) for kind, value in selectors if value]
    if len(selected) != 1:
        raise ValueError("--trace-store requires exactly one of --cut, --owner, or --causal-root")
    return selected[0]


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        view_json = _load_view_json(args)
    except (DurableTraceReadError, TraceStoreReadError, SchemaVersionError, FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    if args.bind != "127.0.0.1":
        sys.stdout.write(
            f"!! --bind {args.bind}: the viewer exposes the following to anyone who can reach this port:\n"
        )
        for cat in _BIND_WARNING_CATEGORIES:
            sys.stdout.write(f"   - {cat}\n")

    httpd = make_server(view_json, bind=args.bind, port=args.port)
    host = "localhost" if args.bind in ("127.0.0.1", "0.0.0.0") else args.bind  # noqa: S104
    url = f"http://{host}:{args.port}/"
    sys.stdout.write(f"trace-viewer serving {url}\n")
    n = view_json.get("run", {}).get("summary", {})
    sys.stdout.write(f"  events={n.get('events', '?')} lanes={n.get('lanes', '?')} edges={n.get('edges', '?')}\n")
    sys.stdout.write("  Ctrl-C to stop.\n")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stdout.write("\nstopping.\n")
    finally:
        httpd.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the trace-viewer CLI parser."""
    parser = argparse.ArgumentParser(prog="shepherd-trace-viewer")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="serve the trace viewer over HTTP")
    source = serve.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace-payload", type=Path, help="raw durable trace JSON or v2 TraceView JSON")
    source.add_argument("--trace-rev", help="durable trace revision head to read from a VcsCore workspace")
    source.add_argument("--trace-head", action="store_true", help="read the selected trace head from a workspace")
    source.add_argument("--trace-store", type=Path, help="SQLite shepherd2 TraceStore file")
    selectors = serve.add_mutually_exclusive_group()
    selectors.add_argument("--cut", help="published TraceStore cut/frontier id")
    selectors.add_argument("--owner", help="TraceStore owner path id")
    selectors.add_argument("--causal-root", help="record id to use as a causal-closure root")
    serve.add_argument("--through", type=int, help="owner ordinal cutoff for --owner")
    serve.add_argument("--visibility", choices=("shape_only", "payload", "full_internal"), default="payload")
    serve.add_argument("--mode", choices=("declarations_only", "captures_only", "both"), default="both")
    serve.add_argument("--actor", default="trace-viewer", help="ReadContext actor_ref for TraceStore reads")
    serve.add_argument(
        "--trusted-internal", action="store_true", help="present trusted:internal for full_internal reads"
    )
    serve.add_argument("--workspace", type=Path, default=Path(), help="workspace holding .vcscore")
    serve.add_argument("--port", type=int, default=8767)
    serve.add_argument("--bind", default="127.0.0.1", help="network interface (default: 127.0.0.1)")
    serve.add_argument("--open", action="store_true", help="open the default browser")
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
