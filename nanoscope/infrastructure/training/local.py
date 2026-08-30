"""Training on this machine, behind the port both providers satisfy (M8-T03).

The first thing in this project that produces a model. ADR-0006 named it in M0 —
*"trains on this machine (ultralytics), device resolved by the Device Manager"* —
and M8-T01 wrote the fourteen assertions it has to satisfy before any of this
existed. The deliverable is that suite passing with three new fixtures and no new
assertions (`tests/contract/test_local_training_provider.py`).

**The job runner is underneath, and there is no second thread policy.** ADR-0080
§2: this provider drives its run with `JobRunner`, so the checkpoints, the
listener and the honest limit on what cancel can promise are ADR-0043's, decided
once. Progress goes through `JobContext.report(epoch, epochs)`, which is why
M5-T07's job status widget will show a training run without knowing what one is.

**What was measured before this was written**, against ultralytics 8.4.41 rather
than assumed:

- `on_fit_epoch_end` is the epoch boundary, and `trainer.stop = True` inside it
  ends the run — the loop reads `if self.stop: break` immediately after firing
  it. Asked for 8 epochs, stopped after 2, and `best.pt` was still on disk.
- **That callback fires twice for the last epoch** (the final validation fires it
  again): three epochs reported `[0, 1, 2, 2]`. The port promises one entry per
  epoch, never sparse, so `_Reporter` deduplicates by number.
- Ultralytics epochs are 0-based; `EpochMetrics.epoch` is 1-based.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nanoscope.application.jobs import Job, JobContext, JobRunner
from nanoscope.core.entities.training import (
    DatasetSpec,
    EpochMetrics,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.ports.device import DeviceProvider

logger = logging.getLogger(__name__)

#: Where a run's artifacts go when the configuration names nowhere. Under
#: `models/`, which is where PROJECT_RULES §5 puts weights.
DEFAULT_OUTPUT = "models/training"

#: Ultralytics' names for the quantities ADR-0080 declared. The translation lives
#: here because the port declared the vocabulary once so that two providers could
#: not spell one quantity two ways (ADR-0031's rule) — which makes mapping the
#: adapter's job, not the reader's.
_VALIDATION = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}

#: The three components ultralytics reports for each of the two losses. Reported
#: as their **sum**: a total is what a chart plots, and the split is a
#: framework's internal rather than a quantity this project has named.
_LOSS_PARTS = ("box_loss", "cls_loss", "dfl_loss")


class LocalTrainingProvider:
    """Trains on this machine. Satisfies `TrainingProvider` structurally."""

    def __init__(
        self,
        project_root: Path | str,
        jobs: JobRunner,
        devices: DeviceProvider,
    ) -> None:
        """
        Args:
            project_root: the open project. Every path a run reports is relative
                to it (ADR-0003), so the project still opens after it is moved.
            jobs: the runner the work goes on. Handed in rather than made here —
                one thread pool per application, and the composition root owns it.
            devices: the one authority on where things run (ADR-0004). This class
                never asks torch, which is PROJECT_RULES §2.6.
        """
        self._root = Path(project_root)
        self._jobs = jobs
        self._devices = devices
        self._lock = threading.Lock()
        self._runs: dict[str, TrainingRun] = {}
        self._by_run: dict[str, Job] = {}
        #: Ids cancelled before their job was registered. `submit` returns after
        #: the work may already be running, so there is a window between "the
        #: run exists" and "there is a handle to stop it" — and the contract's
        #: own first test cancels immediately. A cancel that silently does
        #: nothing is the class of bug ADR-0043 was written about.
        self._cancel_pending: set[str] = set()

    # ── The port ──────────────────────────────────────────────────────────────

    def start(
        self,
        dataset: DatasetSpec,
        config: TrainingConfig,
        listener: Callable[[TrainingRun], None] | None = None,
    ) -> TrainingRun:
        """Begin a run and return it before it has trained anything."""
        manifest = self._root / dataset.root / "data.yaml"
        if not manifest.is_file():
            #: Refused here rather than reported as a run that failed four
            #: seconds in, because nothing was ever started (M8-T01's contract).
            raise InvalidParameterError(
                f"no dataset at {dataset.root}: {manifest.name} is not there. Build one "
                "from the project's annotations first (M8-T02)"
            )

        selection = self._devices.select(config.device)
        if selection.is_fallback:
            #: A fallback nobody is told about is a forty-fold slowdown that
            #: looks like the application being slow (ADR-0004, ADR-0049).
            logger.warning("training on %s: %s", selection.device, selection.reason)

        run = TrainingRun(
            run_id=str(uuid.uuid4()),
            status=TrainingStatus.PENDING,
            dataset=dataset,
            config=config,
            device=selection.device,
            started_utc=_now(),
        )
        with self._lock:
            self._runs[run.run_id] = run

        job = self._jobs.submit(
            f"Training {config.base_model} for {config.epochs} epoch(s)",
            lambda context: self._train(context, run.run_id, selection.device.torch_name, listener),
        )
        with self._lock:
            self._by_run[run.run_id] = job
            asked = run.run_id in self._cancel_pending
        if asked:
            job.cancel()
        return run

    def status(self, run_id: str) -> TrainingRun:
        """The run as it is now — a fresh snapshot, cheap enough to poll."""
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise InvalidParameterError(f"no training run {run_id!r} on this machine")
        return run

    def cancel(self, run_id: str) -> None:
        """Ask a run to stop at its next epoch boundary. Never raises."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.is_finished:
                #: An unknown or finished run: the caller is a button that can
                #: be pressed twice, and the second press is not an error dialog.
                return
            job = self._by_run.get(run_id)
            if job is None:
                self._cancel_pending.add(run_id)
                return
        #: The runner's flag, not a second one. ADR-0080 §2: the local provider
        #: drives its run with `JobRunner` underneath, and what must not happen
        #: is a second thread policy beside the one ADR-0043 settled.
        job.cancel()

    # ── The run ───────────────────────────────────────────────────────────────

    def _train(
        self,
        context: JobContext,
        run_id: str,
        torch_device: str,
        listener: Callable[[TrainingRun], None] | None,
    ) -> None:
        """The body, on the runner's thread.

        Every exception is the runner's to catch (ADR-0043 §6) — but a job that
        failed leaves the *run* saying `RUNNING`, so this catches too, records
        the reason on the snapshot, and re-raises for the job to hold the
        traceback.
        """
        from ultralytics import YOLO  # heavy, and function-local for that reason

        run = self._publish(run_id, TrainingStatus.RUNNING, listener=listener)
        reporter = _Reporter(self, run_id, context, listener)

        try:
            model = YOLO(run.config.base_model)
            model.add_callback("on_fit_epoch_end", reporter)
            output = self._root / (run.config.output_directory or DEFAULT_OUTPUT)
            model.train(
                data=str(self._root / run.dataset.root / "data.yaml"),
                epochs=run.config.epochs,
                imgsz=run.config.image_size_px,
                device=torch_device,
                #: `None` means the framework decides, which is what M8-T01 made
                #: `None` mean rather than a number this layer invented.
                **({} if run.config.batch_size is None else {"batch": run.config.batch_size}),
                **({} if run.config.seed is None else {"seed": run.config.seed}),
                #: No held-out set, no validation pass — and therefore no
                #: `validation` metric block, which is what ADR-0080 made
                #: `val_images == 0` mean.
                val=run.dataset.val_images > 0,
                project=str(output.parent),
                name=output.name,
                plots=False,
                verbose=False,
            )
        except BaseException as failure:
            self._publish(run_id, TrainingStatus.FAILED, error=str(failure), listener=listener)
            raise

        if context.cancelled:
            #: Cancelled runs keep their epochs and whatever checkpoint the
            #: framework last wrote, the way a cancelled import keeps the files
            #: it already copied (ADR-0043 §8, ADR-0080 §6).
            self._publish(run_id, TrainingStatus.CANCELLED, listener=listener)
            return

        #: `model.trainer` is typed `BaseTrainer | None` and is set by `train`;
        #: the assert is for mypy and for the day a future ultralytics stops
        #: setting it, which would otherwise be an `AttributeError` inside a
        #: worker thread rather than a failed run with a reason.
        trainer = model.trainer
        if trainer is None:  # pragma: no cover — ultralytics always sets it
            raise InvalidParameterError("ultralytics finished without a trainer to read")
        best = Path(trainer.best)
        self._publish(
            run_id,
            TrainingStatus.SUCCEEDED,
            #: Read back off the trainer rather than composed here: ultralytics
            #: increments the directory name on collision, so a path this
            #: adapter assembled would name a directory it did not use.
            weights_path=best.relative_to(self._root).as_posix(),
            listener=listener,
        )

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def _publish(
        self,
        run_id: str,
        status: TrainingStatus | None = None,
        *,
        metrics: tuple[EpochMetrics, ...] | None = None,
        weights_path: str | None = None,
        error: str | None = None,
        listener: Callable[[TrainingRun], None] | None = None,
    ) -> TrainingRun:
        """Replace the snapshot under the lock, then tell the listener outside it.

        Outside, because a listener blocks — a Qt marshal, a repaint — and
        holding the lock `status()` needs across it is the deadlock this
        project's own fake provider walked into on 2026-08-30.
        """
        with self._lock:
            current = self._runs[run_id]
            finishing = status is not None and status in _FINISHED
            run = replace(
                current,
                status=current.status if status is None else status,
                metrics=current.metrics if metrics is None else metrics,
                weights_path=weights_path or current.weights_path,
                error=current.error if error is None else error,
                finished_utc=_now() if finishing else current.finished_utc,
            )
            self._runs[run_id] = run
        if listener is not None:
            listener(run)
        return run


_FINISHED = (TrainingStatus.SUCCEEDED, TrainingStatus.FAILED, TrainingStatus.CANCELLED)


class _Reporter:
    """The `on_fit_epoch_end` callback: one epoch out, one cancellation in.

    A class rather than a closure because it holds the one piece of state that
    makes the port's promise true — the last epoch already reported.
    """

    def __init__(
        self,
        provider: LocalTrainingProvider,
        run_id: str,
        context: JobContext,
        listener: Callable[[TrainingRun], None] | None,
    ) -> None:
        self._provider = provider
        self._run_id = run_id
        self._context = context
        self._listener = listener
        self._last = 0

    def __call__(self, trainer: Any) -> None:
        epoch = int(trainer.epoch) + 1  # ultralytics counts from zero
        if epoch <= self._last:
            #: The final validation fires this callback again for the epoch that
            #: has already been reported — measured: three epochs give
            #: `[0, 1, 2, 2]`. The port promises one entry per epoch, in order
            #: and never sparse, so the second one is dropped.
            return
        self._last = epoch

        run = self._provider.status(self._run_id)
        values = _values(trainer, held_out=run.dataset.val_images > 0)
        self._provider._publish(
            self._run_id,
            metrics=(*run.metrics, EpochMetrics(epoch=epoch, values=values)),
            listener=self._listener,
        )
        self._context.report(epoch, run.config.epochs, f"epoch {epoch} of {run.config.epochs}")

        if self._context.cancelled:
            #: ADR-0043's *stop at the next checkpoint*, and for a trainer the
            #: checkpoint is here. Set rather than raised: raising out of a
            #: framework callback abandons the checkpoint the port promised a
            #: cancelled run would keep.
            logger.info("stopping run %s at epoch %d", self._run_id, epoch)
            trainer.stop = True


def _values(trainer: Any, *, held_out: bool) -> Mapping[str, float]:
    """One epoch's numbers, in the vocabulary the port declared (ADR-0080).

    `held_out` is the whole of why this is not a straight forwarding of
    `trainer.metrics`. **Ultralytics validates the final epoch whether or not it
    was asked to** — `if self.args.val or final_epoch: self.metrics = self.validate()`
    — and when nothing was held out, `data.yaml`'s `val` points at the training
    split because the trainer refuses a manifest where it does not resolve
    (ADR-0081). So numbers appear, and they are the model scored on what it
    trained on.

    Reporting those as `validation` would be the self-confirmation ADR-0044
    named one level down, dressed as a metric. ADR-0080's block means *a held-out
    set existed*, not *a validation pass ran*, and this is the line where the
    difference is kept. Found by the contract suite: the last epoch of a
    no-validation run arrived carrying a precision.
    """
    reported: dict[str, float] = dict(trainer.metrics or {})
    reported.update(trainer.label_loss_items(trainer.tloss) or {})

    values: dict[str, float] = {}
    train_loss = _total(reported, "train")
    if train_loss is not None:
        values["train_loss"] = train_loss
    if not held_out:
        return values

    val_loss = _total(reported, "val")
    quality = {name: reported[key] for name, key in _VALIDATION.items() if key in reported}
    if val_loss is not None and len(quality) == len(_VALIDATION):
        #: All of it or none of it. Half a block says a quantity was measured
        #: and lost, when it was never measured (ADR-0031, ADR-0080 §4) — and
        #: `EpochMetrics` refuses a partial one anyway.
        values["val_loss"] = val_loss
        values.update({name: float(value) for name, value in quality.items()})
    return values


def _total(reported: Mapping[str, Any], prefix: str) -> float | None:
    """`box + cls + dfl`, or `None` when that pass did not run."""
    parts = [reported[f"{prefix}/{part}"] for part in _LOSS_PARTS if f"{prefix}/{part}" in reported]
    return float(sum(parts)) if len(parts) == len(_LOSS_PARTS) else None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
