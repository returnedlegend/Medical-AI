"""Desktop resources and lifecycle for a private, embedded local API server.

Kept free of Qt imports so the server/resource contract can be tested directly.
"""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

logger = logging.getLogger(__name__)
TOKEN_HEADER = "x-retraction-desktop-token"


def resource_root() -> Path:
    """PyInstaller unpacks bundled read-only resources under _MEIPASS."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def frontend_directory() -> Path:
    path = resource_root() / "frontend" / "dist"
    if not (path / "index.html").is_file():
        raise RuntimeError("The application interface is missing. Build frontend/dist before packaging.")
    return path


class TokenGate:
    """Only the native WebEngine session can use this per-launch local server."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            supplied = headers.get(TOKEN_HEADER.encode(), b"")
            if not hmac.compare_digest(supplied, self.token.encode("ascii")):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    body = b"This private server is available only inside Retraction Lab."
                    await send({"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]})
                    await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


class LocalServer:
    """Own an ephemeral loopback socket and stop only the server we started."""

    def __init__(self, data_directory: Path, web_directory: Path | None = None):
        self.data_directory = data_directory
        self.web_directory = web_directory or frontend_directory()
        self.token = secrets.token_urlsafe(32)
        self._thread = None
        self._server = None
        self._socket = None
        self._error = None
        # A proxy is not used for this fixed, loopback-only in-process channel.
        self._opener = build_opener(ProxyHandler({}))

    @property
    def url(self) -> str:
        if self._socket is None:
            raise RuntimeError("Desktop API server has not started")
        return f"http://127.0.0.1:{self._socket.getsockname()[1]}"

    def start(self, timeout: float = 30) -> None:
        import uvicorn
        from retraction.main import create_app

        self.data_directory.mkdir(parents=True, exist_ok=True)
        app = create_app(data_dir=self.data_directory, web_dist=self.web_directory)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.set_inheritable(False)
        self._socket.listen(128)
        config = uvicorn.Config(TokenGate(app, self.token), host="127.0.0.1", port=0,
                                loop="asyncio", http="h11", ws="none", lifespan="on",
                                log_config=None, access_log=False)
        self._server = uvicorn.Server(config)

        def run():
            try:
                self._server.run(sockets=[self._socket])
            except BaseException as exc:
                self._error = exc
                logger.exception("Desktop API server failed")

        self._thread = threading.Thread(target=run, name="desktop-api", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._error is not None:
                self.stop()
                raise RuntimeError("The local research service could not start") from self._error
            try:
                if self.request("/api/health", timeout=0.5)["status"] == "ok":
                    return
            except (OSError, URLError, ValueError):
                time.sleep(0.05)
        self.stop()
        raise RuntimeError("The local research service did not become ready within 30 seconds")

    def request_bytes(self, path: str, data: dict | None = None, timeout: float = 30) -> bytes:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Only application-relative paths are permitted")
        headers = {TOKEN_HEADER: self.token}
        payload = None
        if data is not None:
            payload = json.dumps(data, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        request = Request(self.url + path, data=payload, headers=headers)
        with self._opener.open(request, timeout=timeout) as response:
            return response.read()

    def request(self, path: str, data: dict | None = None, timeout: float = 30):
        return json.loads(self.request_bytes(path, data, timeout))

    def stop(self, timeout: float | None = 15) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("A research job is still finishing during desktop shutdown")
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass

