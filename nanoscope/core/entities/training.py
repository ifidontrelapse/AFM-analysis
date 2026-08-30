"""What a training run is, before anything trains (M8-T01).

ADR-0006 chose this seam in M0 and said why: training runs for hours instead of
seconds, consumes a dataset rather than an image, produces artifacts and metrics
rather than detections, must be cancellable, and may run on another machine. So
it is a module behind a port, and these are the records that port speaks in.

They are written **before** the first provider on purpose, and ADR-0080 argues
the case: the surface here is only what M8-T02…T05 will actually call, and the
thing that keeps that honest is a contract suite two implementations must pass
(`tests/contract/`) rather than a docstring promising they will.

**Nothing here imports the job runner.** `application.jobs` is one layer out, and
the arrow points inward — but the duplication of five state names is a decision,
not an oversight: a `Job` is in-process and dies with the process, and a training
run has to be findable after a restart (M8-T04) and may be executing on a machine
this application did not start (M8-T07). ADR-0080 §2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from nanoscope.core.entities.device import Device
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.values import DeviceKind

#: Every metric a provider may report, grouped the way ADR-0031 grouped
#: measurements: **a block is present in full or absent in full**, so two epochs
#: of one run cannot disagree about what was measured, and a reader can ask
#: "did this run validate?" instead of "is `map50` in this dict?".
#:
#: The split is the one the work actually has: a trainer always has a training
#: loss, and everything else exists only if a validation pass ran — which is
#: exactly the case `DatasetSpec.val_images == 0` describes.
METRIC_BLOCKS: Mapping[str, tuple[str, ...]] = {
    "loss": ("train_loss",),
    "validation": ("val_loss", "precision", "recall", "map50", "map50_95"),
}

#: Flattened once, so the check below is a lookup and not a nested loop.
METRIC_NAMES: frozenset[str] = frozenset(name for names in METRIC_BLOCKS.values() for name in names)


class TrainingStatus(StrEnum):
    """Where a run is. The last three are terminal.

    Deliberately the same five words as `JobState`, and deliberately a different
    enum: `core` may not import `application`, and the two answer different
    questions — a job's state is about a callable on a thread pool, a run's is
    about work that outlives the process (ADR-0080 §2).
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL = (TrainingStatus.SUCCEEDED, TrainingStatus.FAILED, TrainingStatus.CANCELLED)


@dataclass(frozen=True)
class DatasetSpec:
    """The dataset a run trains on, as the provider needs to talk about it.

    What is *inside* `root` is the builder's business (M8-T02) and the provider's
    to read — no framework's file name appears here, because `core` naming one
    would decide for the second provider what the first happened to use
    (PROJECT_RULES §2.5, in spirit).

    `classes` is here rather than only on disk because a run has to be readable
    a month later, when the directory may be gone: a `ModelDescriptor.class_map`
    is built from this (ADR-0005).
    """

    #: The dataset directory, relative to the project root (ADR-0003: paths in
    #: the project are relative).
    root: str
    #: Class names in index order — index 0 is `classes[0]`, the same rule
    #: `classes.txt` carries in M7-T09's export.
    classes: tuple[str, ...]
    #: How the builder split it. `val_images == 0` is legal and means the
    #: `validation` metric block will be absent for every epoch.
    train_images: int
    val_images: int = 0

    def __post_init__(self) -> None:
        if not self.root:
            raise InvalidParameterError("dataset root is empty: a run must name what it trained on")
        if not self.classes:
            raise InvalidParameterError(
                "dataset has no classes: a detector with nothing to detect is not a training run"
            )
        if self.train_images < 1:
            raise InvalidParameterError(
                f"train_images={self.train_images}: a dataset with no training image trains nothing"
            )
        if self.val_images < 0:
            raise InvalidParameterError(f"val_images={self.val_images} is negative")


@dataclass(frozen=True)
class TrainingConfig:
    """What to train, for how long, and where the result goes.

    Every field is something a provider needs and an operator can be asked for.
    Anything a framework can decide for itself is `None` here rather than a
    guess this layer invents — the same rule ADR-0019 wrote for an unknown pixel
    scale, applied to a hyperparameter.
    """

    #: Weights to start from. A `ModelDescriptor.path` for fine-tuning what this
    #: project already has, or whatever name the provider's framework resolves
    #: for a fresh start. Passed through, never interpreted here.
    base_model: str
    epochs: int
    image_size_px: int
    #: `None` lets the framework choose — the honest value on a machine whose
    #: memory this layer cannot see.
    batch_size: int | None = None
    #: A preference, not a decision. The provider resolves it through the device
    #: manager and reports what it got on the run (ADR-0004, PROJECT_RULES §2.6).
    device: DeviceKind | None = None
    #: `None` means the framework's own seeding. Set it and a run is repeatable
    #: to the extent the framework allows, which is the only claim this layer
    #: can make for it.
    seed: int | None = None
    #: Where artifacts land, relative to the project root — under `models/` by
    #: PROJECT_RULES §5. Empty lets the provider choose a name under it.
    #:
    #: There is no `collect_artifacts()` on the port, and this field is why: the
    #: provider puts the weights where the project can see them, so "the run
    #: succeeded" and "the file is here" are one fact rather than two (ADR-0080
    #: §5, and ADR-0006's *no silent artifacts on disk*).
    output_directory: str = ""

    def __post_init__(self) -> None:
        if not self.base_model:
            raise InvalidParameterError(
                "base_model is empty: a run must say what it started from, or its "
                "result has no provenance (ADR-0005)"
            )
        if self.epochs < 1:
            raise InvalidParameterError(f"epochs={self.epochs}: a run of no epochs is not a run")
        if self.image_size_px < 1:
            raise InvalidParameterError(f"image_size_px={self.image_size_px} is not a size")
        if self.batch_size is not None and self.batch_size < 1:
            raise InvalidParameterError(
                f"batch_size={self.batch_size}: use None to let the framework decide, not 0"
            )


@dataclass(frozen=True)
class EpochMetrics:
    """What one epoch reported: its number, and named scalars.

    ADR-0031's shape, one milestone on and for a different producer — *one
    quantity, one name, and a block present in full or absent in full*. The
    alternative is what it was there: two providers, two vocabularies, and a
    chart that has to guess whether `mAP50` and `map50` are the same column.

    The check runs in the constructor rather than in a validator a caller must
    remember, because the caller here is a framework callback firing once a
    minute for six hours, and the first wrong name should fail on epoch 1.
    """

    #: One-based, the way an operator reads it: "epoch 3 of 50".
    epoch: int
    values: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise InvalidParameterError(f"epoch={self.epoch}: epochs are numbered from 1")

        unknown = sorted(set(self.values) - METRIC_NAMES)
        if unknown:
            raise InvalidParameterError(
                f"unknown metric(s) {unknown}: the vocabulary is declared once in "
                f"METRIC_BLOCKS, so two providers cannot name one quantity twice "
                f"(ADR-0031's rule). Known: {sorted(METRIC_NAMES)}"
            )

        for block, names in METRIC_BLOCKS.items():
            present = [name for name in names if name in self.values]
            if present and len(present) != len(names):
                missing = sorted(set(names) - set(present))
                raise InvalidParameterError(
                    f"metric block {block!r} is partial: {missing} missing. A block is "
                    f"present in full or absent in full — half a block says a quantity "
                    f"was measured and lost, when it was never measured (ADR-0031)"
                )

    def has(self, block: str) -> bool:
        """Whether this epoch reported that block. Raises for a name that is not one."""
        return all(name in self.values for name in METRIC_BLOCKS[block])


@dataclass(frozen=True)
class TrainingRun:
    """One training run, at one moment: its identity, its state, what it produced.

    **This is the handle.** `TrainingProvider.start` returns one, `status`
    returns a fresh one, and every field is a snapshot rather than a live view —
    which is what lets the same record describe a local run on a thread and a
    remote run polled over a socket, and what lets M8-T04 store it as it is.
    """

    #: Unique in the project, chosen by the provider that started the run, and
    #: the only thing `status` and `cancel` are given.
    run_id: str
    status: TrainingStatus
    dataset: DatasetSpec
    config: TrainingConfig
    #: One entry per completed epoch, in order. Never sparse: an epoch missing
    #: from here is an epoch that did not finish.
    metrics: tuple[EpochMetrics, ...] = ()
    #: The weights this run produced, relative to the project root, or `None`
    #: until it has succeeded. A path here is a file that exists — that is the
    #: whole of ADR-0006's *no silent artifacts on disk*.
    weights_path: str | None = None
    #: What it actually ran on, as the device manager resolved it — not what the
    #: config asked for. `None` for a run that never started.
    device: Device | None = None
    started_utc: str = ""
    finished_utc: str = ""
    #: Why it failed, in a sentence an operator can read. Empty otherwise.
    error: str = ""

    @property
    def is_finished(self) -> bool:
        return self.status in _TERMINAL

    @property
    def epochs_done(self) -> int:
        """How many epochs have finished — the numerator of the progress bar.

        Derived from the last report rather than counted separately, so the
        number on screen and the metrics behind it cannot disagree.
        """
        return self.metrics[-1].epoch if self.metrics else 0
