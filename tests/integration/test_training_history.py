"""A run the project remembers, and the model it produced (M8-T04, ADR-0084).

M8-T01 wrote what this file has to prove into the entity it defined: *a `Job`
dies with the process; a training run has to be findable after a restart.* So
the assertions here close the project and open it again, which is the only form
of "it persists" that is not a claim about a dict.

The provider below is a **synchronous** stub rather than
`tests/contract/fake_provider.py`: what is under test is the use case's policy —
what gets written, and when a model is registered — and a thread in the middle
of that turns an assertion about policy into an assertion about timing. The
threaded provider is exercised by the contract suite, which is where it belongs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from nanoscope.application.use_cases.training import descriptor_for, start_training
from nanoscope.core.entities.device import Device
from nanoscope.core.entities.model import ModelFramework, ModelTask
from nanoscope.core.entities.training import (
    DatasetSpec,
    EpochMetrics,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.ports import TrainingProvider
from nanoscope.core.values import DeviceKind
from nanoscope.infrastructure.storage import SqliteProjectRepository

DATASET = DatasetSpec(
    root="cache/dataset-2026-09-03",
    classes=("particle", "aggregate"),
    train_images=8,
    val_images=2,
)
CONFIG = TrainingConfig(
    base_model="yolo11n.pt",
    epochs=3,
    image_size_px=640,
    batch_size=4,
    device=DeviceKind.CUDA,
    seed=7,
    output_directory="models/run-1",
)
CPU = Device(kind=DeviceKind.CPU, name="CPU", torch_name="cpu")


def _epoch(number: int, *, validates: bool) -> EpochMetrics:
    values = {"train_loss": 1.0 / number}
    if validates:
        values |= {
            "val_loss": 2.0 / number,
            "precision": 0.5,
            "recall": 0.6,
            "map50": 0.7,
            "map50_95": 0.4,
        }
    return EpochMetrics(epoch=number, values=values)


def _run(
    *,
    status: TrainingStatus = TrainingStatus.SUCCEEDED,
    dataset: DatasetSpec = DATASET,
    epochs: int = 3,
    weights_path: str | None = "models/run-1/weights/best.pt",
    error: str = "",
) -> TrainingRun:
    return TrainingRun(
        run_id="11111111-2222-3333-4444-555555555555",
        status=status,
        dataset=dataset,
        config=CONFIG,
        metrics=tuple(
            _epoch(number, validates=dataset.val_images > 0) for number in range(1, epochs + 1)
        ),
        weights_path=weights_path,
        device=CPU,
        started_utc="2026-09-03T10:00:00+00:00",
        finished_utc="2026-09-03T10:42:00+00:00",
        error=error,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SqliteProjectRepository]:
    with SqliteProjectRepository.create(tmp_path / "P", "P") as repository:
        yield repository


class TestARunSurvivesTheProcess:
    def test_it_round_trips_whole(self, repo: SqliteProjectRepository) -> None:
        run = _run()
        repo.save_training_run(run)

        assert repo.get_training_run(run.run_id) == run

    def test_it_is_there_after_the_project_is_closed(self, tmp_path: Path) -> None:
        """The assertion the whole task exists for (ADR-0080 §2)."""
        run = _run()
        with SqliteProjectRepository.create(tmp_path / "Q", "Q") as repository:
            repository.save_training_run(run)

        with SqliteProjectRepository.open(tmp_path / "Q") as reopened:
            assert reopened.get_training_run(run.run_id) == run

    def test_the_configuration_comes_back_as_it_was_asked_for(
        self, repo: SqliteProjectRepository
    ) -> None:
        """`config.device` is the *request*; `run.device` is what it got (ADR-0049)."""
        repo.save_training_run(_run())
        stored = repo.get_training_run(_run().run_id)

        assert stored.config == CONFIG
        assert stored.config.device is DeviceKind.CUDA
        assert stored.device == CPU

    def test_a_run_that_never_started_ran_nowhere(self, repo: SqliteProjectRepository) -> None:
        run = replace(_run(status=TrainingStatus.CANCELLED, weights_path=None), device=None)
        repo.save_training_run(run)

        assert repo.get_training_run(run.run_id).device is None

    def test_a_failed_run_keeps_its_reason(self, repo: SqliteProjectRepository) -> None:
        run = _run(status=TrainingStatus.FAILED, weights_path=None, error="CUDA out of memory")
        repo.save_training_run(run)

        assert repo.get_training_run(run.run_id).error == "CUDA out of memory"

    def test_an_unrecorded_run_is_refused_rather_than_answered(
        self, repo: SqliteProjectRepository
    ) -> None:
        with pytest.raises(InvalidParameterError, match="no training run"):
            repo.get_training_run("no-such-run")

    def test_runs_come_back_oldest_first(self, repo: SqliteProjectRepository) -> None:
        first = _run()
        second = replace(first, run_id="second", started_utc="2026-09-03T11:00:00+00:00")
        repo.save_training_run(second)
        repo.save_training_run(first)

        assert [one.run_id for one in repo.list_training_runs()] == [first.run_id, "second"]


class TestEveryEpochIsKept:
    def test_they_come_back_in_order_and_complete(self, repo: SqliteProjectRepository) -> None:
        run = _run(epochs=5)
        repo.save_training_run(run)

        stored = repo.get_training_run(run.run_id)
        assert [one.epoch for one in stored.metrics] == [1, 2, 3, 4, 5]
        assert stored.epochs_done == 5

    def test_the_numbers_survive(self, repo: SqliteProjectRepository) -> None:
        repo.save_training_run(_run())

        assert repo.get_training_run(_run().run_id).metrics[0].values == {
            "train_loss": 1.0,
            "val_loss": 2.0,
            "precision": 0.5,
            "recall": 0.6,
            "map50": 0.7,
            "map50_95": 0.4,
        }

    def test_a_run_with_nothing_held_out_stores_no_validation_block(
        self, repo: SqliteProjectRepository
    ) -> None:
        """ADR-0082's line, one layer down: absent stays absent, and does not
        come back as a NaN or a zero (ADR-0031, ADR-0025)."""
        run = _run(dataset=replace(DATASET, val_images=0))
        repo.save_training_run(run)

        stored = repo.get_training_run(run.run_id)
        assert stored.metrics
        for one in stored.metrics:
            assert one.has("loss")
            assert not one.has("validation")

    def test_saving_the_same_run_again_advances_it(self, repo: SqliteProjectRepository) -> None:
        """What a listener does once an epoch: the whole snapshot, every time."""
        running = _run(status=TrainingStatus.RUNNING, epochs=1, weights_path=None)
        repo.save_training_run(running)
        repo.save_training_run(_run(epochs=3))

        stored = repo.get_training_run(running.run_id)
        assert stored.status is TrainingStatus.SUCCEEDED
        assert stored.epochs_done == 3
        assert len(repo.list_training_runs()) == 1


class _StubProvider:
    """A `TrainingProvider` that publishes a prepared list of snapshots, in order.

    Synchronous on purpose: `start` returns after the listener has seen every one
    of them, so a test asserts what was recorded rather than when.
    """

    def __init__(self, snapshots: list[TrainingRun]) -> None:
        self._snapshots = snapshots
        self.cancelled: list[str] = []

    def start(
        self,
        dataset: DatasetSpec,
        config: TrainingConfig,
        listener: Callable[[TrainingRun], None] | None = None,
    ) -> TrainingRun:
        for snapshot in self._snapshots:
            if listener is not None:
                listener(snapshot)
        return replace(self._snapshots[0], status=TrainingStatus.RUNNING, metrics=())

    def status(self, run_id: str) -> TrainingRun:
        return self._snapshots[-1]

    def cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)


def _weights(repo: SqliteProjectRepository, run: TrainingRun) -> Path:
    """Put the file a succeeded run says it produced where it says it is."""
    assert run.weights_path is not None
    path = repo.root / run.weights_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a model, but a file")
    return path


class TestTheUseCaseKeepsTheRecord:
    def test_it_satisfies_the_port(self) -> None:
        provider: TrainingProvider = _StubProvider([_run()])
        assert isinstance(provider, TrainingProvider)

    def test_every_snapshot_is_written(self, repo: SqliteProjectRepository) -> None:
        finished = _run()
        _weights(repo, finished)
        provider = _StubProvider(
            [
                _run(status=TrainingStatus.RUNNING, epochs=1, weights_path=None),
                _run(status=TrainingStatus.RUNNING, epochs=2, weights_path=None),
                finished,
            ]
        )
        start_training(repo, provider, DATASET, CONFIG, model_id="particles-v13")

        assert repo.get_training_run(finished.run_id) == finished

    def test_a_succeeded_run_registers_what_it_produced(
        self, repo: SqliteProjectRepository
    ) -> None:
        """ADR-0006's compliance clause, and M8-T03's stated remainder."""
        finished = _run()
        weights = _weights(repo, finished)
        start_training(
            repo,
            _StubProvider([finished]),
            DATASET,
            CONFIG,
            model_id="particles-v13",
        )

        model = repo.get_model("particles-v13")
        assert model.task is ModelTask.DETECT
        assert model.framework is ModelFramework.ULTRALYTICS
        assert repo.path_of_model(model) == weights
        assert model.input_size_px == 640
        assert model.class_map == {0: "particle", 1: "aggregate"}
        assert model.sha256 is not None, "the checksum is computed from the file (ADR-0040)"
        assert DATASET.root in model.provenance
        assert finished.run_id in model.provenance

    def test_a_cancelled_run_registers_nothing(self, repo: SqliteProjectRepository) -> None:
        """A model row pointing at weights nobody wrote is the disagreement
        ADR-0080 §5 removed by refusing `collect_artifacts()`."""
        cancelled = _run(status=TrainingStatus.CANCELLED, epochs=1, weights_path=None)
        start_training(repo, _StubProvider([cancelled]), DATASET, CONFIG, model_id="particles-v13")

        assert repo.list_models() == []
        assert repo.get_training_run(cancelled.run_id).status is TrainingStatus.CANCELLED

    def test_a_failed_run_registers_nothing_and_is_still_recorded(
        self, repo: SqliteProjectRepository
    ) -> None:
        failed = _run(status=TrainingStatus.FAILED, epochs=1, weights_path=None, error="no disk")
        start_training(repo, _StubProvider([failed]), DATASET, CONFIG, model_id="particles-v13")

        assert repo.list_models() == []
        assert repo.get_training_run(failed.run_id).error == "no disk"

    def test_the_caller_hears_every_snapshot_after_it_is_recorded(
        self, repo: SqliteProjectRepository
    ) -> None:
        finished = _run()
        _weights(repo, finished)
        seen: list[TrainingRun] = []
        start_training(
            repo,
            _StubProvider(
                [_run(status=TrainingStatus.RUNNING, epochs=1, weights_path=None), finished]
            ),
            DATASET,
            CONFIG,
            model_id="particles-v13",
            listener=seen.append,
        )

        assert [one.status for one in seen] == [TrainingStatus.RUNNING, TrainingStatus.SUCCEEDED]
        # The listener is a UI: by the time it hears "succeeded", the model it
        # will offer to select is already in the project.
        assert repo.get_model("particles-v13")

    def test_an_unnamed_model_is_refused_before_anything_starts(
        self, repo: SqliteProjectRepository
    ) -> None:
        provider = _StubProvider([_run()])
        with pytest.raises(InvalidParameterError, match="model_id"):
            start_training(repo, provider, DATASET, CONFIG, model_id="")

        assert repo.list_training_runs() == []


class TestWhatARunSaysAboutItsModel:
    def test_a_run_with_no_weights_has_no_model_to_register(self) -> None:
        with pytest.raises(InvalidParameterError, match="no weights"):
            descriptor_for(_run(status=TrainingStatus.CANCELLED, weights_path=None), model_id="x")

    def test_the_provenance_names_what_a_reader_asks_first(self) -> None:
        provenance = descriptor_for(_run(), model_id="x").provenance

        assert "yolo11n.pt" in provenance
        assert "3 of 3 epoch(s)" in provenance
        assert "8 training image(s)" in provenance
        assert "2 held out" in provenance
        assert "CPU" in provenance
