"""Stdlib HTTP server for the trace viewer.

Serves the vendored single-page app plus the current run's ViewModel at
``/api/trace``. Zero third-party dependencies.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

ASSETS_DIR = Path(__file__).with_name("assets")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".woff2": "font/woff2",
}


class _Handler(BaseHTTPRequestHandler):
    # Set by serve(): the ViewModel dict and the assets directory.
    view_json: ClassVar[dict[str, Any]] = {}
    assets_dir: ClassVar[Path] = ASSETS_DIR

    def log_message(self, *args: Any) -> None:  # quiet by default
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            self._serve_asset("index.html")
        elif path == "/api/trace":
            body = json.dumps(self.view_json).encode()
            self._send(200, body, "application/json")
        elif path.startswith("/assets/"):
            self._serve_asset(path[len("/assets/") :])
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def _serve_asset(self, rel: str) -> None:
        # Resolve safely under assets_dir; reject traversal.
        target = (self.assets_dir / rel).resolve()
        try:
            target.relative_to(self.assets_dir.resolve())
        except ValueError:
            self._send(403, b"forbidden", "text/plain; charset=utf-8")
            return
        if not target.is_file():
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ctype = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), ctype)


def make_server(view_json: dict[str, Any], *, bind: str, port: int) -> ThreadingHTTPServer:
    """Build (but do not start) a server bound to ``bind:port``."""
    handler = type(
        "BoundHandler",
        (_Handler,),
        {"view_json": view_json, "assets_dir": ASSETS_DIR},
    )
    return ThreadingHTTPServer((bind, port), handler)


def serve(view_json: dict[str, Any], *, bind: str = "127.0.0.1", port: int = 8767) -> ThreadingHTTPServer:
    """Start serving forever (blocking).

    Returns the server for tests that run it in a thread.
    """
    httpd = make_server(view_json, bind=bind, port=port)
    httpd.serve_forever()
    return httpd
