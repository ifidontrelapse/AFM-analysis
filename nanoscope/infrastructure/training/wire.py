"""A training run, as JSON, in both directions (M8-T07, ADR-0087).

The encoding the remote protocol speaks. One module rather than one per side,
because a codec written twice is two codecs that agree until they do not — and
what this one carries is the thing the contract compares field for field.

**Written, not derived**, and that is a measurement rather than a preference.
`dataclasses.asdict` followed by `json.dumps` does not fail on a `TrainingRun`:
it produces 501 valid characters, and `TrainingRun(**json.loads(text))` hands
back a snapshot whose `dataset` is a `dict` and whose `status` is a `str`. It
compares unequal to what was sent, `is_finished` reads wrong on it, and nothing
anywhere raises. A wire format this project can be silently wrong about is one
to write on purpose.

**Decoding reconstructs the entities**, so their constructors still run: an
`EpochMetrics` naming a metric this application does not know is refused at the
boundary rather than becoming a chart, which is ADR-0080 §4's guard reaching the
network without being written twice.

Nothing here is versioned. There is no second version and no deployed worker to
be compatible with; when there is one, the version goes in the envelope and this
comment is what says it was left out on purpose.
"""

from __future__ import annotations

from typing import Any

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


def encode_run(run: TrainingRun) -> dict[str, Any]:
    """A snapshot as JSON-safe data. Every field, because a snapshot is whole."""
    return {
        "run_id": run.run_id,
        "status": str(run.status),
        "dataset": encode_dataset(run.dataset),
        "config": encode_config(run.config),
        "metrics": [{"epoch": one.epoch, "values": dict(one.values)} for one in run.metrics],
        "weights_path": run.weights_path,
        "device": None if run.device is None else encode_device(run.device),
        "started_utc": run.started_utc,
        "finished_utc": run.finished_utc,
        "error": run.error,
    }


def decode_run(data: Any) -> TrainingRun:
    """A snapshot back, as the entities rather than as dictionaries.

    Raises:
        InvalidParameterError: the payload is not a run this application can
            read — a missing field, a status it does not know, or a metric name
            that is not in `METRIC_BLOCKS`. Refused **here**, at the edge,
            because the alternative is a `TrainingRun` whose `status` is a
            string that no comparison matches and whose `is_finished` is always
            false (ADR-0087).
    """
    fields = _mapping(data, "run")
    try:
        return TrainingRun(
            run_id=str(fields["run_id"]),
            status=TrainingStatus(fields["status"]),
            dataset=decode_dataset(fields["dataset"]),
            config=decode_config(fields["config"]),
            metrics=tuple(
                #: The constructor runs, so an unknown metric fails on the
                #: epoch it arrives in rather than in a chart six hours later.
                EpochMetrics(epoch=int(one["epoch"]), values=dict(one["values"]))
                for one in fields.get("metrics", ())
            ),
            weights_path=fields.get("weights_path"),
            device=None if fields.get("device") is None else decode_device(fields["device"]),
            started_utc=str(fields.get("started_utc", "")),
            finished_utc=str(fields.get("finished_utc", "")),
            error=str(fields.get("error", "")),
        )
    except (KeyError, TypeError, ValueError) as broken:
        raise InvalidParameterError(f"this is not a training run: {broken}") from broken


def encode_dataset(dataset: DatasetSpec) -> dict[str, Any]:
    """The spec, with `root` **relative** — which is what makes it portable.

    ADR-0003 made every path in a project relative to the project, for a reason
    that had nothing to do with a network: so the project survives being moved.
    It is what lets one string be true under two different roots here, and it is
    why neither side of this protocol translates a path.
    """
    return {
        "root": dataset.root,
        "classes": list(dataset.classes),
        "train_images": dataset.train_images,
        "val_images": dataset.val_images,
    }


def decode_dataset(data: Any) -> DatasetSpec:
    fields = _mapping(data, "dataset")
    return DatasetSpec(
        root=str(fields["root"]),
        classes=tuple(str(name) for name in fields["classes"]),
        train_images=int(fields["train_images"]),
        val_images=int(fields.get("val_images", 0)),
    )


def encode_config(config: TrainingConfig) -> dict[str, Any]:
    return {
        "base_model": config.base_model,
        "epochs": config.epochs,
        "image_size_px": config.image_size_px,
        "batch_size": config.batch_size,
        "device": None if config.device is None else str(config.device),
        "seed": config.seed,
        "output_directory": config.output_directory,
    }


def decode_config(data: Any) -> TrainingConfig:
    fields = _mapping(data, "config")
    return TrainingConfig(
        base_model=str(fields["base_model"]),
        epochs=int(fields["epochs"]),
        image_size_px=int(fields["image_size_px"]),
        batch_size=None if fields.get("batch_size") is None else int(fields["batch_size"]),
        #: The *preference*, which is a different fact from what the worker
        #: resolved — ADR-0084 §3 stored both for the same reason.
        device=None if fields.get("device") is None else DeviceKind(fields["device"]),
        seed=None if fields.get("seed") is None else int(fields["seed"]),
        output_directory=str(fields.get("output_directory", "")),
    )


def encode_device(device: Device) -> dict[str, Any]:
    """A resolved device: three fields that are present or absent together."""
    return {"kind": str(device.kind), "name": device.name, "torch_name": device.torch_name}


def decode_device(data: Any) -> Device:
    fields = _mapping(data, "device")
    return Device(
        kind=DeviceKind(fields["kind"]),
        name=str(fields["name"]),
        torch_name=str(fields["torch_name"]),
    )


def _mapping(data: Any, what: str) -> dict[str, Any]:
    """A JSON object, or a refusal naming what was expected.

    A list or a string reaching a `[...]` lookup raises `TypeError` two frames
    down with a message about indices; this is the sentence PROJECT_RULES §3
    asks for instead.
    """
    if not isinstance(data, dict):
        raise InvalidParameterError(f"expected a {what} object, got {type(data).__name__}")
    return data
