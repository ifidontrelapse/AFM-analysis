"""A run cancelled before it trains anything still ends (M8-T04, ADR-0084).

The contract suite asserts this for every provider, but its local subclass is
`slow` and skipped where ultralytics is absent — which is CI. This file pins the
same promise **deterministically and without a trainer**: the runner is given one
worker and that worker is busy, so the training job is queued, the cancel drops
it, and `_train` is never called.

That is exactly the path the defect lived on. Measured before the fix, with this
arrangement: `pending` immediately, `pending` a second later, and `pending`
after a restart, because nothing publishes when the body does not run.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from nanoscope.application.jobs import JobRunner
from nanoscope.core.entities.device import Device, DeviceSelection
from nanoscope.core.entities.training import (
    DatasetSpec,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from nanoscope.core.values import DeviceKind
from nanoscope.infrastructure.training import LocalTrainingProvider


class _CpuOnly:
    """A `DeviceProvider` that answers CPU, so nothing here imports torch."""

    def available(self) -> list[Device]:
        return [Device(kind=DeviceKind.CPU, name="CPU", torch_name="cpu")]

    def select(self, preferred: DeviceKind | None = None) -> DeviceSelection:
        return DeviceSelection(device=self.available()[0], requested=preferred)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project root with the one file `start` insists on seeing."""
    (tmp_path / "cache" / "ds").mkdir(parents=True)
    (tmp_path / "cache" / "ds" / "data.yaml").write_text("path: .\ntrain: images\nval: images\n")
    return tmp_path


def test_a_queued_run_that_is_cancelled_reaches_a_terminal_state(project: Path) -> None:
    seen: list[TrainingRun] = []
    release = threading.Event()

    with JobRunner(max_workers=1) as jobs:
        # The one worker, occupied: the training job below is queued, and a
        # queued job that is cancelled is dropped rather than run (ADR-0043).
        jobs.submit("something else", lambda context: release.wait(5))

        provider = LocalTrainingProvider(project, jobs, _CpuOnly())
        run = provider.start(
            DatasetSpec(root="cache/ds", classes=("particle",), train_images=2, val_images=1),
            TrainingConfig(base_model="yolo11n.yaml", epochs=3, image_size_px=32),
            listener=seen.append,
        )
        provider.cancel(run.run_id)

        stopped = provider.status(run.run_id)
        assert stopped.status is TrainingStatus.CANCELLED
        assert stopped.is_finished
        assert stopped.finished_utc, "a terminal run says when it ended"
        assert stopped.metrics == (), "nothing trained, so there is nothing to keep"

        # The listener is what a record and a UI are built on: a terminal state
        # nobody was told about is one nothing can store or stop showing.
        assert seen and seen[-1].status is TrainingStatus.CANCELLED
        release.set()
