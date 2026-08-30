"""The contract suite, run against the provider that really trains (M8-T03).

**Three fixtures and no new assertions.** That is the deliverable M8-T01 promised
when it wrote a port before its adapter and argued the case in ADR-0080 §1: the
objection to an unimplemented port is that it is *unfalsifiable*, and the answer
was fourteen assertions a second implementation would have to satisfy unchanged.
This file is where that claim is settled — every test in
`TrainingProviderContract` runs here, and not one of them was edited to make it.

**`slow`, and in the gate.** Measured before it was written: three epochs on two
32-px images with one held out cost **2.7 s** on CPU, so the whole subclass is
seconds rather than minutes. An environment variable guarding it would make it a
test nobody runs, and a test nobody runs is one that rots — PROJECT_RULES §6
keeps *inference* out of the gate because it is not reproducible, not because it
is slow, and nothing here asserts a number a model produced.

It skips where ultralytics is not installed, which is CI (the `ci` dependency
group has no torch on purpose, M1-T08). The skip names what is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from training_provider import TrainingProviderContract

from nanoscope.application.jobs import JobRunner
from nanoscope.core.entities.training import DatasetSpec, TrainingConfig
from nanoscope.core.ports import TrainingProvider
from nanoscope.infrastructure.device import DeviceManager

pytestmark = pytest.mark.slow

#: Built from a YAML that ships inside ultralytics, so a contract run needs no
#: download and no checkpoint — `yolo11n.yaml` resolves to the package's own
#: `cfg/models/11/yolo11.yaml` with the `n` scale. Training from scratch on four
#: images produces a useless model, which is the point: the contract is about the
#: port's behaviour, and nothing in it asserts a number.
FROM_SCRATCH = "yolo11n.yaml"

#: Small enough that the suite is seconds, more than one so the cancellation test
#: has an epoch boundary to stop at — which the contract asks for by name.
EPOCHS = 6
IMAGE_PX = 32


def _dataset(project_root: Path, *, val_images: int) -> DatasetSpec:
    """A dataset on disk in the shape M8-T02 builds, written here directly.

    Not through `build_dataset`: this file is testing the *provider*, and a
    fixture that also exercises the builder makes a failure two suspects wide.
    """
    import cv2

    root = project_root / "cache" / f"ds{val_images}"
    halves = {"train": 4, "val": val_images}
    for half, count in halves.items():
        (root / "images" / half).mkdir(parents=True, exist_ok=True)
        (root / "labels" / half).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            picture = np.full((IMAGE_PX, IMAGE_PX), 200, np.uint8)
            cv2.circle(picture, (IMAGE_PX // 2, IMAGE_PX // 2), IMAGE_PX // 6, 40, -1)
            cv2.imwrite(str(root / "images" / half / f"{index}.png"), picture)
            (root / "labels" / half / f"{index}.txt").write_text("0 0.5 0.5 0.3 0.3\n")

    #: `val` points at the training split when nothing is held out, which is
    #: what M8-T02 writes and why: ultralytics refuses a manifest whose `val`
    #: does not resolve, and validates the final epoch regardless (ADR-0081).
    #: The provider is what refuses to call those numbers validation.
    half = "val" if val_images else "train"
    manifest = f"path: {root}\ntrain: images/train\nval: images/{half}\n"
    manifest += "nc: 1\nnames:\n  0: 'particle'\n"
    (root / "data.yaml").write_text(manifest)
    return DatasetSpec(
        root=root.relative_to(project_root).as_posix(),
        classes=("particle",),
        train_images=4,
        val_images=val_images,
    )


class TestLocalTrainingProvider(TrainingProviderContract):
    """A held-out set, which is the ordinary case."""

    val_images = 2

    @pytest.fixture
    def provider(self, project_root: Path) -> TrainingProvider:
        pytest.importorskip(
            "ultralytics",
            reason="training needs ultralytics and torch; CI installs neither (M1-T08)",
        )
        from nanoscope.infrastructure.training import LocalTrainingProvider

        with JobRunner(max_workers=2) as jobs:
            #: CPU by name rather than by policy: a contract run must not queue
            #: behind whatever else is using the GPU, and nothing here measures
            #: how good the model is.
            yield LocalTrainingProvider(project_root, jobs, DeviceManager())

    @pytest.fixture
    def trainable(self, project_root: Path) -> tuple[DatasetSpec, TrainingConfig]:
        from nanoscope.core.values import DeviceKind

        return _dataset(project_root, val_images=self.val_images), TrainingConfig(
            base_model=FROM_SCRATCH,
            epochs=EPOCHS,
            image_size_px=IMAGE_PX,
            batch_size=2,
            device=DeviceKind.CPU,
            seed=0,
            output_directory="models/contract",
        )


class TestLocalTrainingProviderWithoutValidation(TestLocalTrainingProvider):
    """Nothing held out — ADR-0031's absent block, against a real trainer.

    Every assertion inherited, including the one that says the `validation`
    block is absent for every epoch when `val_images == 0`. Here that is not a
    fake choosing not to report: the run is started with `val=False` and
    ultralytics never computes them.
    """

    val_images = 0
