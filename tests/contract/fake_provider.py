"""A `TrainingProvider` that trains nothing, so the contract can be checked (M8-T01).

It exists for one reason: a contract suite with no implementation behind it is a
document, and this project deleted six ports in M2-T08 for exactly that (ADR-0041).
The fake is the second half of ADR-0080 §1 — the port ships without an adapter,
but not without something that has to satisfy it.

What it does honestly, because the contract asks: hands back a run before it has
finished, reports one epoch at a time in order, stops at an epoch boundary when
asked, writes a file where the configuration said the weights would go, and
records the device it "ran" on.

What it fakes: the numbers, and the CPU it claims. Nothing here is a substitute
for M8-T03 — it is the thing M8-T03's provider must agree with.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from nanoscope.core.entities.device import Device
from nanoscope.core.entities.training import (
    DatasetSpec,
    EpochMetrics,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.values import DeviceKind

#: Long enough that the contract's cancellation test can reach an epoch boundary
#: and stop the run before it ends, short enough that the whole suite is
#: milliseconds. The real providers get theirs from the work.
EPOCH_SECONDS = 0.02


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class FakeTrainingProvider:
    """Runs a loop that counts epochs, on a thread, and writes a file at the end."""

    def __init__(self, project_root: Path) -> None:
        self._root = project_root
        self._lock = threading.Lock()
        self._runs: dict[str, TrainingRun] = {}
        self._cancelled: set[str] = set()

    # ── The port ──────────────────────────────────────────────────────────────

    def start(
        self,
        dataset: DatasetSpec,
        config: TrainingConfig,
        listener: Callable[[TrainingRun], None] | None = None,
    ) -> TrainingRun:
        if not (self._root / dataset.root).is_dir():
            raise InvalidParameterError(
                f"dataset directory {dataset.root!r} is not in this project: refused "
                "here rather than reported as a run that failed a second in"
            )

        run = TrainingRun(
            run_id=str(uuid.uuid4()),
            status=TrainingStatus.RUNNING,
            dataset=dataset,
            config=config,
            device=Device(kind=DeviceKind.CPU, name="CPU", torch_name="cpu"),
            started_utc=_now(),
        )
        with self._lock:
            self._runs[run.run_id] = run
        threading.Thread(target=self._train, args=(run.run_id, listener), daemon=True).start()
        return run

    def status(self, run_id: str) -> TrainingRun:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise InvalidParameterError(f"no training run {run_id!r} in this provider")
        return run

    def cancel(self, run_id: str) -> None:
        # No raise for an unknown or finished id: the caller is a button.
        with self._lock:
            self._cancelled.add(run_id)

    # ── The loop that is not training ─────────────────────────────────────────

    def _train(self, run_id: str, listener: Callable[[TrainingRun], None] | None) -> None:
        run = self.status(run_id)
        validates = run.dataset.val_images > 0

        for epoch in range(1, run.config.epochs + 1):
            # The checkpoint is the epoch boundary, and only the boundary —
            # ADR-0043's "stop at the next checkpoint", where a trainer's is.
            #
            # The flag is read under the lock and published outside it: `_publish`
            # takes the same lock, and `threading.Lock` is not reentrant. Found by
            # this file deadlocking on its own cancellation test.
            with self._lock:
                stopping = run_id in self._cancelled
            if stopping:
                self._publish(run_id, TrainingStatus.CANCELLED, finished=True, listener=listener)
                return
            threading.Event().wait(EPOCH_SECONDS)
            run = self._publish(
                run_id,
                TrainingStatus.RUNNING,
                metrics=(*self.status(run_id).metrics, _metrics(epoch, validates)),
                listener=listener,
            )

        weights = Path(run.config.output_directory) / "best.pt"
        (self._root / weights).parent.mkdir(parents=True, exist_ok=True)
        (self._root / weights).write_bytes(b"not a model")
        self._publish(
            run_id,
            TrainingStatus.SUCCEEDED,
            finished=True,
            weights_path=weights.as_posix(),
            listener=listener,
        )

    def _publish(
        self,
        run_id: str,
        status: TrainingStatus,
        *,
        metrics: tuple[EpochMetrics, ...] | None = None,
        weights_path: str | None = None,
        finished: bool = False,
        listener: Callable[[TrainingRun], None] | None = None,
    ) -> TrainingRun:
        """Replace the snapshot under the lock, then tell the listener outside it.

        Outside, because a listener that blocks — a Qt marshal, a repaint —
        must not hold the lock `status()` needs.
        """
        from dataclasses import replace

        with self._lock:
            run = replace(
                self._runs[run_id],
                status=status,
                metrics=self._runs[run_id].metrics if metrics is None else metrics,
                weights_path=weights_path or self._runs[run_id].weights_path,
                finished_utc=_now() if finished else self._runs[run_id].finished_utc,
            )
            self._runs[run_id] = run
        if listener is not None:
            listener(run)
        return run


def _metrics(epoch: int, validates: bool) -> EpochMetrics:
    """Numbers that go the right way, so a chart drawn from them looks like training."""
    values = {"train_loss": 1.0 / epoch}
    if validates:
        values |= {
            "val_loss": 1.2 / epoch,
            "precision": 1.0 - 0.5 / epoch,
            "recall": 1.0 - 0.6 / epoch,
            "map50": 1.0 - 0.7 / epoch,
            "map50_95": 1.0 - 0.9 / epoch,
        }
    return EpochMetrics(epoch=epoch, values=values)
