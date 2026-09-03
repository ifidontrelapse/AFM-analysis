"""Training on a machine this application did not start (M8-T07, ADR-0087).

ADR-0006 named two implementations in M0 and said what the second was for: *"the
operator's laptop is not always the right GPU."* M8-T01 then wrote the port
before either existed and justified it in these words — *"let the port be wrong
now, cheaply, instead of in M8-T07 when a second implementation discovers it."*
This is that implementation, and the deliverable is the **fifteen existing
assertions passing unchanged** across a socket.

**A client, and only a client.** No worker ships with this project. ADR-0006
predicted this task's failure mode in its own Negative section — *"the remote
protocol risks being designed for an imagined deployment"* — so the protocol is
exactly what the contract forces and nothing else:

```
POST /runs              a tar of the dataset, plus the spec and config as JSON
GET  /runs/<id>         the snapshot
POST /runs/<id>/cancel  accepted, always
GET  /runs/<id>/weights the bytes a succeeded run produced
```

Three contract assertions are the whole of that list. `(project_root /
run.weights_path).is_file()` forces the weights **back**; a worker that cannot
read the dataset forces it **out**; and `run.device is not None` forces the
worker to say what it ran on. Everything a deployment would also want —
authentication, listing, resumption, log streaming — is absent because there is
no deployment to want it.

**Relative paths are the interoperability.** ADR-0003 made every path in a
project relative to the project so the project survives being moved; here that
is what lets one string be true under two different roots. The dataset is
uploaded under the same relative root the client knows it by, and the weights
land at the same relative path the worker reports. Neither side translates.

**The last transition is the client's.** The weights are downloaded *before*
`SUCCEEDED` is published, because a listener that heard it first would find the
file missing for as long as the transfer takes — the disagreement ADR-0080 §5
removed by refusing `collect_artifacts()`, arriving from the other direction.
"""

from __future__ import annotations

import io
import logging
import tarfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from nanoscope.core.entities.training import (
    DatasetSpec,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from nanoscope.core.errors import InvalidParameterError
from nanoscope.infrastructure.training.wire import decode_run, encode_config, encode_dataset

logger = logging.getLogger(__name__)

#: How often the watcher asks the worker how a run is going. Seconds, because a
#: training epoch is minutes and the contract's own comment asks for a provider
#: *"not hammered"* by a caller polling every 10 ms — `status()` answers from the
#: last snapshot, so the two rates are independent.
DEFAULT_POLL_S = 2.0

#: How long any one request may take. Long enough for a 5 MB dataset upload over
#: a slow link — measured: a 40-scan dataset is 5.2 MB — short enough that a
#: worker that has gone away is noticed rather than waited on for ever.
DEFAULT_TIMEOUT_S = 60.0


class RemoteTrainingProvider:
    """Submits to a worker over HTTP. Satisfies `TrainingProvider` structurally."""

    def __init__(
        self,
        base_url: str,
        project_root: Path | str,
        *,
        poll_seconds: float = DEFAULT_POLL_S,
        timeout_seconds: float = DEFAULT_TIMEOUT_S,
        session: requests.Session | None = None,
    ) -> None:
        """
        Args:
            base_url: where the worker is. A constructor argument and **not a
                stored preference**: a `training.remote_url` setting would be
                the imagined deployment ADR-0006 warned about, made permanent in
                a file (ADR-0041, eighth application).
            project_root: this project. The dataset is read from under it and
                the weights are written back into it, both by the relative paths
                the port already speaks (ADR-0003).
            poll_seconds: how often to ask. Independent of how often a caller
                asks *this*, which is what makes a 10 ms poll harmless.
            timeout_seconds: per request.
            session: handed in for a test, or to carry a proxy configuration.
        """
        self._base = base_url.rstrip("/")
        self._root = Path(project_root)
        self._poll_s = poll_seconds
        self._timeout_s = timeout_seconds
        self._http = session or requests.Session()
        self._lock = threading.Lock()
        self._runs: dict[str, TrainingRun] = {}
        #: Ids cancelled before the worker acknowledged the start. The same
        #: window `LocalTrainingProvider` had to close, for the same reason —
        #: the contract's own first test cancels immediately (ADR-0043).
        self._cancelled: set[str] = set()

    # ── The port ──────────────────────────────────────────────────────────────

    def start(
        self,
        dataset: DatasetSpec,
        config: TrainingConfig,
        listener: Callable[[TrainingRun], None] | None = None,
    ) -> TrainingRun:
        """Send the dataset and the configuration, and start watching."""
        source = self._root / dataset.root
        if not (source / "data.yaml").is_file():
            #: Refused before anything is uploaded, and in the same words the
            #: local provider uses: nothing was started, so this is not a run
            #: that failed (M8-T01's contract).
            raise InvalidParameterError(
                f"no dataset at {dataset.root}: data.yaml is not there. Build one "
                "from the project's annotations first (M8-T02)"
            )

        payload = {
            "dataset": encode_dataset(dataset),
            "config": encode_config(config),
        }
        response = self._post(
            "/runs",
            files={
                #: **`tar`, not `tar.gz`** — measured: a 40-scan dataset is
                #: 5.2 MB of PNG and short text, and gzip turns it into 5.3 MB.
                #: Compressing compressed data spends CPU to add bytes.
                "dataset": ("dataset.tar", _packed(source), "application/x-tar"),
                "manifest": (None, _json(payload), "application/json"),
            },
        )
        run = decode_run(response.json())
        with self._lock:
            self._runs[run.run_id] = run
            pending = run.run_id in self._cancelled

        if pending:
            self.cancel(run.run_id)
        threading.Thread(
            target=self._watch, args=(run.run_id, listener), daemon=True, name="nanoscope-remote"
        ).start()
        return run

    def status(self, run_id: str) -> TrainingRun:
        """The last snapshot the watcher fetched. **Local, and deliberately.**

        The contract polls this every 10 ms, and a provider that turned each
        call into a request would be one nobody could poll. The watcher owns the
        network at its own rate; this answers from what it last heard, which is
        what a snapshot is for (ADR-0080 §3).

        Raises:
            InvalidParameterError: no run by that id was started through this
                provider — its own answer, not the worker's, because an id from
                another client is not this object's to describe.
        """
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise InvalidParameterError(f"no training run {run_id!r} on this client")
        return run

    def cancel(self, run_id: str) -> None:
        """Ask the worker to stop. Never raises — the caller is a button.

        A cancel that arrives before `start` has a run id is remembered rather
        than dropped, which is the window ADR-0043 exists for and the one
        M8-T03 had to close on the local side.
        """
        with self._lock:
            known = self._runs.get(run_id)
            if known is None:
                self._cancelled.add(run_id)
                return
            if known.is_finished:
                return
        try:
            self._post(f"/runs/{run_id}/cancel")
        except Exception as unreachable:
            #: Logged and swallowed. `cancel` promises not to raise, and a
            #: worker that cannot be reached is what the watcher is about to
            #: report anyway.
            logger.warning("could not ask the worker to cancel %s: %s", run_id, unreachable)

    # ── Watching a run that is somewhere else ─────────────────────────────────

    def _watch(self, run_id: str, listener: Callable[[TrainingRun], None] | None) -> None:
        """Poll until the run is finished, publishing what changes.

        **A plain thread, and this is the one place this milestone departs from
        ADR-0043.** `LocalTrainingProvider` drives its run with `JobRunner`
        because there the run *is* the work; here the work is on another machine
        and what runs locally is a watcher that sleeps. `max_workers` is 2, and
        a sleeping loop holding one of them for six hours would leave this
        application one worker to import, analyse and export with — a worse
        version of the cost M8-T05 already had to state.
        """
        while True:
            try:
                fetched = decode_run(self._get(f"/runs/{run_id}").json())
            except Exception as lost:
                self._publish(_unreachable(self.status(run_id), lost), listener)
                return

            if fetched.status is TrainingStatus.SUCCEEDED and fetched.weights_path:
                try:
                    self._download(run_id, fetched.weights_path)
                except Exception as lost:
                    #: A run that says it succeeded and has no weights here is
                    #: the disagreement ADR-0080 §5 refused to allow. It failed,
                    #: from this side, and the sentence says which half.
                    self._publish(_unreachable(fetched, lost), listener)
                    return

            self._publish(fetched, listener)
            if fetched.is_finished:
                return
            time.sleep(self._poll_s)

    def _download(self, run_id: str, weights_path: str) -> None:
        """Bring the weights into this project, at the path the worker named.

        **Before `SUCCEEDED` is published**, so a listener that reacts to it
        finds the file — which is the ordering M8-T04's `start_training` depends
        on to register a model.
        """
        destination = self._root / weights_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = self._get(f"/runs/{run_id}/weights")
        destination.write_bytes(response.content)
        logger.info(
            "brought %s back from the worker (%d bytes)", weights_path, len(response.content)
        )

    def _publish(self, run: TrainingRun, listener: Callable[[TrainingRun], None] | None) -> None:
        """Replace the snapshot under the lock, then tell the listener outside it.

        Outside, because a listener that blocks — a Qt marshal, a repository
        write — must not hold the lock `status()` needs. The fake learned this
        by deadlocking on its own cancellation test (M8-T01).

        Silent when nothing moved: the port promises ordered snapshots, not one
        per poll, and a UI redrawing a table every two seconds because a number
        did not change is a UI that flickers.
        """
        with self._lock:
            previous = self._runs.get(run.run_id)
            self._runs[run.run_id] = run
        if listener is not None and run != previous:
            listener(run)

    # ── HTTP, in one place ────────────────────────────────────────────────────

    def _get(self, path: str) -> requests.Response:
        response = self._http.get(f"{self._base}{path}", timeout=self._timeout_s)
        response.raise_for_status()
        return response

    def _post(self, path: str, files: Any = None) -> requests.Response:
        response = self._http.post(f"{self._base}{path}", files=files, timeout=self._timeout_s)
        response.raise_for_status()
        return response


def _packed(source: Path) -> bytes:
    """The dataset directory as one uncompressed tar.

    `arcname="."` so the archive carries the directory's *contents* — the worker
    decides where to put them, under the same relative root the spec names, and
    an archive naming an absolute path is one that cannot be unpacked anywhere
    else.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.add(source, arcname=".")
    return buffer.getvalue()


def _json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload)


def _unreachable(run: TrainingRun, why: BaseException) -> TrainingRun:
    """The run as failed, saying what this side actually knows.

    Not `RUNNING` for ever, and not a silent stall: a watcher that has lost its
    subject knows one thing, which is that it cannot observe the run, and
    saying so is the only honest terminal state available.

    ADR-0084 §8 is not contradicted. That rule is about a **stored** run nobody
    is watching, whose status is what was true when a process died; this is a
    live watcher reporting its own failure, which is an observation rather than
    a substitution.
    """
    from dataclasses import replace
    from datetime import UTC, datetime

    return replace(
        run,
        status=TrainingStatus.FAILED,
        error=f"lost contact with the worker: {why}",
        finished_utc=run.finished_utc or datetime.now(UTC).isoformat(timespec="seconds"),
    )
