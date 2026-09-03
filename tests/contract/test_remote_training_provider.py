"""The contract suite, run across a socket (M8-T07, ADR-0087).

**M8's fourth exit criterion**, and the moment ADR-0080 §1 was written for:

> *"Let the port be wrong now, cheaply, instead of in M8-T07 when a second
> implementation discovers it."*

Fifteen assertions, written in M8-T01 against a fake in this process, satisfied
by `LocalTrainingProvider` in M8-T03 with three fixtures and no edits, and now by
a client that can see none of the run it is describing. **Not one assertion was
changed to make this file pass**, which is the deliverable — the same claim
M8-T03 settled, made once more against a process boundary instead of a thread.

The worker is `stub_worker.StubWorker`: `FakeTrainingProvider` behind
`http.server`, on a real port. **Its root is a different directory**, so the
dataset genuinely travels out and the weights genuinely come back — with one
directory the transfers are no-ops and the suite would prove nothing.

Fast, and in the gate: the fake's epochs are 20 ms and the socket is loopback,
so the whole file is under a second. Nothing here needs torch.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from stub_worker import StubWorker
from training_provider import TrainingProviderContract, await_terminal

from nanoscope.core.entities.training import DatasetSpec, TrainingConfig, TrainingStatus
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.ports import TrainingProvider
from nanoscope.infrastructure.training import RemoteTrainingProvider

#: More than one, so the cancellation test has an epoch boundary to stop at.
EPOCHS = 8

#: Fast enough that the suite is milliseconds. The client's default is seconds,
#: which is right against a worker whose epochs are minutes.
POLL_S = 0.01


def a_dataset(project_root: Path, *, val_images: int) -> DatasetSpec:
    """A dataset directory in the client's project, with files worth moving.

    The manifest is what `start` checks before uploading anything, and the two
    label files are what proves the archive carried more than one entry.
    """
    root = project_root / "cache" / "ds"
    for half in ("train", "val"):
        (root / "images" / half).mkdir(parents=True, exist_ok=True)
        (root / "labels" / half).mkdir(parents=True, exist_ok=True)
        (root / "images" / half / "0.png").write_bytes(b"\x89PNG\r\n\x1a\n not really")
        (root / "labels" / half / "0.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\n")
    return DatasetSpec(
        root=root.relative_to(project_root).as_posix(),
        classes=("particle",),
        train_images=4,
        val_images=val_images,
    )


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """This machine's project. A **subdirectory**, so the worker's root beside
    it is a genuinely different tree rather than the same one twice."""
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def worker(tmp_path: Path) -> Iterator[StubWorker]:
    """The other machine. **A different root** — that is the whole fixture."""
    elsewhere = tmp_path / "worker"
    elsewhere.mkdir()
    with StubWorker(elsewhere) as running:
        yield running


class TestRemoteTrainingProvider(TrainingProviderContract):
    """A held-out set, which is the ordinary case."""

    val_images = 2

    @pytest.fixture
    def provider(self, project_root: Path, worker: StubWorker) -> TrainingProvider:
        return RemoteTrainingProvider(worker.url, project_root, poll_seconds=POLL_S)

    @pytest.fixture
    def trainable(self, project_root: Path) -> tuple[DatasetSpec, TrainingConfig]:
        return a_dataset(project_root, val_images=self.val_images), TrainingConfig(
            base_model="whatever-the-worker-resolves.pt",
            epochs=EPOCHS,
            image_size_px=32,
            output_directory="models/remote",
        )


class TestRemoteTrainingProviderWithNothingHeldOut(TestRemoteTrainingProvider):
    """The same fifteen assertions, with no validation split.

    Which changes exactly one of them — `the validation block follows the split`
    — and it is the assertion ADR-0082 was written for, checked here for the
    third provider.
    """

    val_images = 0


class TestWhatOnlyCrossingAMachineCanGetWrong:
    """The assertions the contract cannot make, because it cannot see a socket."""

    @pytest.fixture
    def provider(self, project_root: Path, worker: StubWorker) -> RemoteTrainingProvider:
        return RemoteTrainingProvider(worker.url, project_root, poll_seconds=POLL_S)

    @pytest.fixture
    def trainable(self, project_root: Path) -> tuple[DatasetSpec, TrainingConfig]:
        return a_dataset(project_root, val_images=2), TrainingConfig(
            base_model="n.pt", epochs=2, image_size_px=32, output_directory="models/remote"
        )

    def test_the_dataset_reaches_the_worker(
        self,
        provider: RemoteTrainingProvider,
        worker: StubWorker,
        trainable: tuple[DatasetSpec, TrainingConfig],
    ) -> None:
        """The upload, asserted as files on the other root rather than as a run
        that happened to finish."""
        dataset, config = trainable
        run = await_terminal(provider, provider.start(dataset, config).run_id)

        assert run.status is TrainingStatus.SUCCEEDED, run.error
        assert worker.uploaded == [dataset.root]
        unpacked = worker.root / dataset.root
        assert (unpacked / "data.yaml").is_file()
        assert (unpacked / "labels" / "train" / "0.txt").read_text().startswith("0 ")

    def test_the_weights_come_back_into_this_project(
        self,
        provider: RemoteTrainingProvider,
        project_root: Path,
        worker: StubWorker,
        trainable: tuple[DatasetSpec, TrainingConfig],
    ) -> None:
        """Two roots, one relative path, and the bytes are equal on both sides.

        The contract asserts the file is *here*; this asserts it was *there*
        first, which is the half a single directory would hide."""
        dataset, config = trainable
        run = await_terminal(provider, provider.start(dataset, config).run_id)

        assert run.weights_path
        here = project_root / run.weights_path
        there = worker.root / run.weights_path
        assert here.is_file() and there.is_file()
        assert here.read_bytes() == there.read_bytes()
        assert here != there

    def test_a_run_survives_the_wire_field_for_field(
        self,
        provider: RemoteTrainingProvider,
        trainable: tuple[DatasetSpec, TrainingConfig],
    ) -> None:
        """The measurement this codec was written for.

        `dataclasses.asdict` + `json.dumps` does **not** raise on a
        `TrainingRun` — it produces 501 valid characters and hands back a
        snapshot whose `dataset` is a `dict` and whose `status` is a `str`,
        comparing unequal to what was sent. Silently.
        """
        dataset, config = trainable
        run = await_terminal(provider, provider.start(dataset, config).run_id)

        assert run.dataset == dataset
        assert run.config == config
        assert run.device is not None
        assert [one.epoch for one in run.metrics] == list(range(1, config.epochs + 1))
        assert all(one.has("loss") for one in run.metrics)

    def test_a_worker_that_stops_answering_ends_the_run(
        self,
        provider: RemoteTrainingProvider,
        worker: StubWorker,
        trainable: tuple[DatasetSpec, TrainingConfig],
    ) -> None:
        """Not `RUNNING` for ever, and not a silent stall.

        A watcher that has lost its subject knows one thing — that it cannot
        observe the run — and saying so is the only honest terminal state it
        has. ADR-0084 §8 is about a *stored* run nobody is watching; this is an
        observation, not a substitution.
        """
        dataset, _config = trainable
        #: Long enough that the run is still going when the worker disappears —
        #: at 20 ms an epoch, the two-epoch default finishes first and the test
        #: passes for the wrong reason.
        started = provider.start(
            dataset,
            TrainingConfig(
                base_model="n.pt", epochs=500, image_size_px=32, output_directory="models/remote"
            ),
        )
        worker.vanish()

        run = await_terminal(provider, started.run_id)
        assert run.status is TrainingStatus.FAILED
        assert "lost contact with the worker" in run.error
        assert run.finished_utc

    def test_a_dataset_that_is_not_here_is_refused_before_anything_is_uploaded(
        self, provider: RemoteTrainingProvider, worker: StubWorker
    ) -> None:
        """Nothing was started, so this is not a run that failed — the same
        answer, and the same sentence, the local provider gives."""
        with pytest.raises(InvalidParameterError, match=r"data\.yaml is not there"):
            provider.start(
                DatasetSpec(root="cache/nothing", classes=("particle",), train_images=1),
                TrainingConfig(base_model="n.pt", epochs=1, image_size_px=32),
            )

        assert worker.uploaded == []

    def test_status_does_not_ask_the_worker(
        self,
        provider: RemoteTrainingProvider,
        worker: StubWorker,
        trainable: tuple[DatasetSpec, TrainingConfig],
    ) -> None:
        """The contract polls `status()` every 10 ms. A provider that turned
        each call into a request is one nobody could poll — the watcher owns the
        network at its own rate, and this answers from the last snapshot."""
        dataset, config = trainable
        run = await_terminal(provider, provider.start(dataset, config).run_id)
        worker.vanish()

        #: The worker is gone and this still answers, which is the assertion.
        assert provider.status(run.run_id).status is TrainingStatus.SUCCEEDED

    def test_an_id_from_another_client_is_not_this_ones_to_describe(
        self, provider: RemoteTrainingProvider
    ) -> None:
        with pytest.raises(InvalidParameterError, match="on this client"):
            provider.status("a-run-this-client-never-started")

    def test_a_cancel_that_arrives_before_the_worker_answered_is_not_dropped(
        self, provider: RemoteTrainingProvider
    ) -> None:
        """The window ADR-0043 exists for, on the other side of a socket: a
        cancel for an id `start` has not returned yet is remembered."""
        provider.cancel("not-yet-started")

        assert "not-yet-started" in provider._cancelled

    def test_cancelling_when_the_worker_is_gone_does_not_raise(
        self,
        provider: RemoteTrainingProvider,
        worker: StubWorker,
        trainable: tuple[DatasetSpec, TrainingConfig],
    ) -> None:
        """The caller is a button, and the port says `cancel` never raises."""
        dataset, config = trainable
        started = provider.start(dataset, config)
        worker.vanish()

        provider.cancel(started.run_id)
