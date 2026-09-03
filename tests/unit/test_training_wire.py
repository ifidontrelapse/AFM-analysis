"""What crosses the wire, and what is refused at the edge (M8-T07, ADR-0087).

The contract suite drives this codec through a stub worker that always sends
well-formed payloads, so the half it cannot reach is the half that matters when
a real worker is on the other end: **a message this application cannot read**.

The reason the codec is written at all is a measurement. `dataclasses.asdict`
followed by `json.dumps` does not fail on a `TrainingRun` — it produces 501
valid characters, and `TrainingRun(**json.loads(text))` hands back a snapshot
whose `dataset` is a `dict` and whose `status` is a `str`, comparing unequal to
what was sent. The first test here is that measurement, kept as an assertion so
that nobody reaches for the derived version again.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

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
from nanoscope.infrastructure.training.wire import decode_run, encode_run


def a_run(**overrides: object) -> TrainingRun:
    fields: dict[str, object] = {
        "run_id": "r",
        "status": TrainingStatus.RUNNING,
        "dataset": DatasetSpec(
            root="cache/ds", classes=("particle", "fibre"), train_images=8, val_images=2
        ),
        "config": TrainingConfig(
            base_model="n.pt",
            epochs=3,
            image_size_px=32,
            batch_size=4,
            device=DeviceKind.CPU,
            seed=7,
            output_directory="models/remote",
        ),
        "metrics": (EpochMetrics(epoch=1, values={"train_loss": 0.5}),),
        "device": Device(kind=DeviceKind.CUDA, name="RTX 4090", torch_name="cuda:0"),
        "started_utc": "2026-09-03T10:00:00+00:00",
    }
    fields.update(overrides)
    return TrainingRun(**fields)  # type: ignore[arg-type]


class TestARunSurvivesTheWire:
    def test_it_comes_back_equal(self) -> None:
        run = a_run(
            status=TrainingStatus.SUCCEEDED,
            weights_path="models/remote/best.pt",
            finished_utc="2026-09-03T11:00:00+00:00",
        )

        assert decode_run(json.loads(json.dumps(encode_run(run)))) == run

    def test_the_derived_version_is_silently_wrong(self) -> None:
        """The measurement this module exists because of. It does not raise."""
        run = a_run()

        text = json.dumps(dataclasses.asdict(run))
        derived = TrainingRun(**json.loads(text))

        assert derived != run
        assert isinstance(derived.dataset, dict)
        assert isinstance(derived.status, str)

        #: And the consequences, which are what would actually ship. Not
        #: `is_finished` — `TrainingStatus` is a `StrEnum`, so the string still
        #: compares equal and that check keeps working, which is precisely what
        #: makes the rest of it quiet.
        assert derived.is_finished == run.is_finished
        with pytest.raises(AttributeError):
            _ = derived.dataset.classes
        with pytest.raises(AttributeError):
            _ = derived.metrics[0].epoch
        with pytest.raises(AttributeError):
            _ = derived.config.epochs

    def test_a_run_with_no_device_and_no_weights_round_trips_too(self) -> None:
        """A run that never started ran nowhere, and the absence is the fact."""
        run = a_run(status=TrainingStatus.PENDING, device=None, metrics=())

        assert decode_run(encode_run(run)) == run

    def test_the_enums_travel_as_their_values(self) -> None:
        """Readable on a wire, and stable across a worker written in something
        else — which is the only reason a protocol has a text encoding at all."""
        encoded = encode_run(a_run())

        assert encoded["status"] == "running"
        assert encoded["config"]["device"] == "cpu"
        assert encoded["device"]["kind"] == "cuda"


class TestWhatIsRefusedAtTheEdge:
    def test_a_metric_this_application_cannot_name(self) -> None:
        """`EpochMetrics` refuses an unknown name in its constructor, and
        decoding **calls the constructor** — so ADR-0080 §4's guard reaches the
        network without being written a second time."""
        payload = encode_run(a_run())
        payload["metrics"] = [{"epoch": 1, "values": {"perplexity": 3.0}}]

        with pytest.raises(InvalidParameterError):
            decode_run(payload)

    def test_a_status_no_version_of_this_application_knows(self) -> None:
        payload = encode_run(a_run())
        payload["status"] = "paused"

        with pytest.raises(InvalidParameterError, match="not a training run"):
            decode_run(payload)

    def test_a_missing_field(self) -> None:
        payload = encode_run(a_run())
        del payload["dataset"]

        with pytest.raises(InvalidParameterError, match="not a training run"):
            decode_run(payload)

    def test_something_that_is_not_an_object_at_all(self) -> None:
        """A list or a string reaching a `[...]` lookup raises `TypeError` two
        frames down, about indices. PROJECT_RULES §3 asks for a sentence."""
        with pytest.raises(InvalidParameterError, match="expected a run object, got list"):
            decode_run([1, 2, 3])

    def test_an_epoch_numbered_from_zero(self) -> None:
        """Ultralytics counts from 0 and `EpochMetrics` from 1 — a difference
        M8-T03 had to translate, and one a second worker could get wrong."""
        payload = encode_run(a_run())
        payload["metrics"] = [{"epoch": 0, "values": {"train_loss": 1.0}}]

        with pytest.raises(InvalidParameterError):
            decode_run(payload)
