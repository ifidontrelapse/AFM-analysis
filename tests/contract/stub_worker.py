"""A training worker on a socket, so the remote client has something to be wrong against (M8-T07).

**A fixture, not a product.** M8-T07 ships a client; no worker ships with this
project, and ADR-0087 says so in as many words. What this file is for is making
`RemoteTrainingProvider` **falsifiable**: the contract suite drives the real
client, over a real socket, against a process boundary it cannot see through.

The worker's training is `FakeTrainingProvider` — the same one the contract has
been checked against since M8-T01 — with `http.server` in front of it. Stdlib,
no dependency, and it means the only new thing under test is the client.

**Its root is a different directory from the client's project**, which is the
whole point. With one directory the upload and the download are no-ops, the
contract still passes, and it proves nothing: the assertion `(project_root /
run.weights_path).is_file()` is only about a transfer if the file was somewhere
else first.
"""

from __future__ import annotations

import json
import tarfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

from fake_provider import FakeTrainingProvider

from nanoscope.core.errors import NanoscopeError
from nanoscope.infrastructure.training.wire import decode_config, decode_dataset, encode_run


class StubWorker:
    """Serves the four endpoints ADR-0087 defines, and trains with the fake."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.provider = FakeTrainingProvider(root)
        #: Every dataset the client uploaded, so a test can assert the bytes
        #: really crossed rather than that the run finished.
        self.uploaded: list[str] = []
        #: Set by `vanish`. Checked at the top of every handler, because
        #: **closing the listening socket does not stop an established
        #: connection**: `protocol_version = "HTTP/1.1"` means keep-alive, the
        #: client's session holds one connection, and its handler thread went on
        #: answering for the whole ten seconds of a 500-epoch run after
        #: `shutdown()`. Found by the test that needed a worker to disappear.
        self.gone = False
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        #: Handler threads do not outlive the process, so `server_close` does
        #: not join them. It joined them before, and with keep-alive each one
        #: was waiting for a request that never came: **half a second of
        #: teardown per test**, 18 s across this file, for nothing.
        self._server.daemon_threads = True
        #: `serve_forever`'s default poll interval is **0.5 s**, and `shutdown`
        #: waits for the loop to notice — half a second of teardown per test,
        #: 18 s across this file. Nothing here is waiting on a network.
        self._thread = threading.Thread(
            target=lambda: self._server.serve_forever(poll_interval=0.01), daemon=True
        )

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> StubWorker:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.vanish()

    def vanish(self) -> None:
        """Stop answering, the way a machine that lost power stops answering.

        Both halves: no new connections, and the established one is dropped
        mid-conversation rather than answered with a tidy error — which is what
        a client on the other side of a network actually meets.
        """
        self.gone = True
        self._server.shutdown()
        self._server.server_close()

    # ── What each endpoint does ───────────────────────────────────────────────

    def create(self, archive: bytes, manifest: dict[str, Any]) -> dict[str, Any]:
        """Unpack the dataset where the spec says, then start training it.

        Under the **same relative root** the client named: that is what makes
        one string true under two roots, and it is why neither side translates
        a path (ADR-0003, ADR-0087 §3).
        """
        dataset = decode_dataset(manifest["dataset"])
        config = decode_config(manifest["config"])
        destination = self.root / dataset.root
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=BytesIO(archive)) as unpacked:
            unpacked.extractall(destination, filter="data")
        self.uploaded.append(dataset.root)
        #: The fake refuses a dataset directory that is not here, so a client
        #: that uploaded nothing fails at exactly this call.
        return encode_run(self.provider.start(dataset, config))

    def read(self, run_id: str) -> dict[str, Any]:
        return encode_run(self.provider.status(run_id))

    def weights(self, run_id: str) -> bytes:
        run = self.provider.status(run_id)
        if not run.weights_path:
            raise NanoscopeError(f"run {run_id} has produced no weights")
        return (self.root / run.weights_path).read_bytes()


def _handler(worker: StubWorker) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            """Silent: the suite's output is the assertions, not an access log."""

        def do_GET(self) -> None:
            if worker.gone:
                self.close_connection = True
                return
            parts = self.path.strip("/").split("/")
            try:
                if len(parts) == 2 and parts[0] == "runs":
                    self._json(worker.read(parts[1]))
                elif len(parts) == 3 and parts[0] == "runs" and parts[2] == "weights":
                    self._bytes(worker.weights(parts[1]))
                else:
                    self._fail(404, f"no such path: {self.path}")
            except NanoscopeError as refusal:
                self._fail(404, str(refusal))

        def do_POST(self) -> None:
            if worker.gone:
                self.close_connection = True
                return
            parts = self.path.strip("/").split("/")
            try:
                if parts == ["runs"]:
                    archive, manifest = _multipart(self)
                    self._json(worker.create(archive, manifest), status=201)
                elif len(parts) == 3 and parts[0] == "runs" and parts[2] == "cancel":
                    worker.provider.cancel(parts[1])
                    self._json({})
                else:
                    self._fail(404, f"no such path: {self.path}")
            except NanoscopeError as refusal:
                self._fail(400, str(refusal))

        # ── Replying ──────────────────────────────────────────────────────────

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _bytes(self, payload: bytes) -> None:
            self._send(200, payload, "application/octet-stream")

        def _fail(self, status: int, why: str) -> None:
            self._send(status, json.dumps({"error": why}).encode(), "application/json")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _multipart(request: BaseHTTPRequestHandler) -> tuple[bytes, dict[str, Any]]:
    """The tar and the manifest out of one `multipart/form-data` body.

    Hand-parsed because the stdlib's `cgi` module is gone in 3.13 and `email`'s
    parser wants the whole message with headers. Twenty lines against a
    dependency for a **test fixture** is the trade PROJECT_RULES §7 asks for.
    """
    boundary = request.headers["Content-Type"].split("boundary=")[1].encode()
    body = request.rfile.read(int(request.headers["Content-Length"]))

    archive = b""
    manifest: dict[str, Any] = {}
    for part in body.split(b"--" + boundary):
        head, _, payload = part.partition(b"\r\n\r\n")
        if not payload:
            continue
        #: Trailing CRLF belongs to the boundary, not to the field.
        payload = payload.rsplit(b"\r\n", 1)[0]
        if b'name="dataset"' in head:
            archive = payload
        elif b'name="manifest"' in head:
            manifest = json.loads(payload)
    return archive, manifest


def worker_at(root: Path) -> Iterator[StubWorker]:
    """A worker serving `root`, for the length of a fixture."""
    with StubWorker(root) as running:
        yield running
