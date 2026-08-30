"""The suite every `TrainingProvider` passes, whoever wrote it (M8-T01).

ADR-0006's compliance clause is one line — *both providers pass the same contract
test suite* — and this is it, written in the task that defined the port rather
than in the task that wrote the first implementation. That ordering is the whole
point: a port justified by its first adapter is that adapter's shape, and the
suite is what lets the port be **wrong** before there is code invested in it
being right (ADR-0080 §1).

**How to use it.** Subclass `TrainingProviderContract`, override the three
fixtures, and every test below runs against your provider:

```python
class TestLocalTrainingProvider(TrainingProviderContract):
    @pytest.fixture
    def project_root(self, tmp_path): ...
    @pytest.fixture
    def provider(self, project_root): ...
    @pytest.fixture
    def trainable(self, project_root): ...
```

`trainable` returns a `(DatasetSpec, TrainingConfig)` your provider can really
train — for the local one that means real files and a real, tiny model, which is
why M8-T03 will mark its subclass `slow` and keep it out of CI (PROJECT_RULES §6:
model inference is not reproducible enough for the gate). The *contract* is in
the gate today because the fake satisfies it in milliseconds.

The suite polls `status()` rather than waiting on a listener, deliberately: that
is what a caller on the other side of a network has to do anyway, so a provider
that only worked through its callback would pass a test it should not.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from nanoscope.core.entities.training import (
    DatasetSpec,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.ports import TrainingProvider

#: How long any single run in this suite may take before the test gives up.
#: Generous, because a real trainer's first epoch includes loading weights; the
#: poll below returns the moment the run is terminal, so a fast provider does not
#: pay for it.
TIMEOUT_S = 300.0

#: The poll interval. Short enough that the fake finishes in milliseconds, long
#: enough that a remote provider's implementation is not hammered.
POLL_S = 0.01


def await_terminal(
    provider: TrainingProvider, run_id: str, timeout: float = TIMEOUT_S
) -> TrainingRun:
    """Poll until the run stops, and fail the test rather than hang if it does not."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = provider.status(run_id)
        if run.is_finished:
            return run
        time.sleep(POLL_S)
    raise AssertionError(f"run {run_id} did not reach a terminal state in {timeout}s")


def await_condition(
    provider: TrainingProvider,
    run_id: str,
    predicate: Callable[[TrainingRun], bool],
    timeout: float = TIMEOUT_S,
) -> TrainingRun:
    """Poll until `predicate` holds, or the run ends, or the clock runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = provider.status(run_id)
        if predicate(run) or run.is_finished:
            return run
        time.sleep(POLL_S)
    raise AssertionError(f"run {run_id} never satisfied the condition in {timeout}s")


class TrainingProviderContract:
    """Every assertion the port makes. Subclass and supply the three fixtures."""

    # ── What a subclass provides ──────────────────────────────────────────────

    @pytest.fixture
    def project_root(self, tmp_path: Path) -> Path:
        """The directory every path in a run is relative to (ADR-0003)."""
        return tmp_path

    @pytest.fixture
    def provider(self, project_root: Path) -> TrainingProvider:
        raise NotImplementedError("a subclass supplies the provider under test")

    @pytest.fixture
    def trainable(self, project_root: Path) -> tuple[DatasetSpec, TrainingConfig]:
        """A dataset and a configuration this provider can actually train.

        Give it more than one epoch: the cancellation test needs a boundary to
        stop at, and a single-epoch run has none.
        """
        raise NotImplementedError("a subclass supplies something trainable")

    # ── The port is the port ──────────────────────────────────────────────────

    def test_it_satisfies_the_port(self, provider: TrainingProvider) -> None:
        # Structural, and only about the method names — mypy checks the
        # signatures when `make types` runs, which is the half `isinstance`
        # cannot do (the same limit `tests/unit/test_ports.py` states for
        # `Detector`).
        assert isinstance(provider, TrainingProvider)

    # ── Starting ──────────────────────────────────────────────────────────────

    def test_start_returns_before_the_training_is_over(
        self, provider: TrainingProvider, trainable: tuple[DatasetSpec, TrainingConfig]
    ) -> None:
        """A `start` that blocked for six hours is the thing this port exists to avoid."""
        dataset, config = trainable
        run = provider.start(dataset, config)

        assert run.run_id
        assert run.status in (TrainingStatus.PENDING, TrainingStatus.RUNNING)
        assert not run.is_finished
        assert run.weights_path is None
        assert run.dataset == dataset
        assert run.config == config

        provider.cancel(run.run_id)

    def test_two_runs_do_not_share_an_id(
        self, provider: TrainingProvider, trainable: tuple[DatasetSpec, TrainingConfig]
    ) -> None:
        dataset, config = trainable
        first = provider.start(dataset, config)
        second = provider.start(dataset, config)
        assert first.run_id != second.run_id

        provider.cancel(first.run_id)
        provider.cancel(second.run_id)

    # ── Asking ────────────────────────────────────────────────────────────────

    def test_status_finds_a_run_by_its_id(
        self, provider: TrainingProvider, trainable: tuple[DatasetSpec, TrainingConfig]
    ) -> None:
        dataset, config = trainable
        started = provider.start(dataset, config)
        assert provider.status(started.run_id).run_id == started.run_id

        provider.cancel(started.run_id)

    def test_an_unknown_id_is_refused_rather_than_answered(
        self, provider: TrainingProvider
    ) -> None:
        """An invented empty run would make a typo look like a run that had not started."""
        with pytest.raises(InvalidParameterError):
            provider.status("no-such-run")

    # ── Finishing ─────────────────────────────────────────────────────────────

    def test_a_finished_run_succeeded_and_its_weights_are_on_disk(
        self,
        provider: TrainingProvider,
        trainable: tuple[DatasetSpec, TrainingConfig],
        project_root: Path,
    ) -> None:
        """ADR-0006's clause, as an assertion: no silent artifacts on disk.

        A run that says it succeeded and points at a file that is not there is
        the failure mode the clause was written against — the operator finds out
        in M8-T04, when registering the model fails for a run reported green.
        """
        dataset, config = trainable
        run = await_terminal(provider, provider.start(dataset, config).run_id)

        assert run.status is TrainingStatus.SUCCEEDED, run.error
        assert run.error == ""
        assert run.weights_path is not None
        assert (project_root / run.weights_path).is_file()
        assert not Path(run.weights_path).is_absolute(), (
            "a run's artifacts are recorded relative to the project (ADR-0003), so the "
            "project still opens after it is moved or copied"
        )
        assert run.started_utc and run.finished_utc
        assert run.device is not None, "a run records what it actually ran on (ADR-0004)"

    def test_every_epoch_reported_once_and_in_order(
        self, provider: TrainingProvider, trainable: tuple[DatasetSpec, TrainingConfig]
    ) -> None:
        dataset, config = trainable
        run = await_terminal(provider, provider.start(dataset, config).run_id)

        epochs = [one.epoch for one in run.metrics]
        assert epochs == list(range(1, config.epochs + 1)), (
            "metrics are one entry per completed epoch, in order and never sparse — "
            "a gap here is a progress bar that jumps"
        )
        assert run.epochs_done == config.epochs

    def test_a_run_reports_a_training_loss_every_epoch(
        self, provider: TrainingProvider, trainable: tuple[DatasetSpec, TrainingConfig]
    ) -> None:
        """The one block a trainer always has. The rest are conditional; this is not."""
        dataset, config = trainable
        run = await_terminal(provider, provider.start(dataset, config).run_id)

        assert all(one.has("loss") for one in run.metrics)

    def test_the_validation_block_follows_the_split(
        self, provider: TrainingProvider, trainable: tuple[DatasetSpec, TrainingConfig]
    ) -> None:
        """ADR-0031's rule where it bites: a held-out set, or no validation numbers.

        Not "NaN where there was no validation pass" — a dataset with nothing
        held out has no precision, and a column of NaN is a substitution with
        better manners (ADR-0031, and ADR-0025 before it).
        """
        dataset, config = trainable
        run = await_terminal(provider, provider.start(dataset, config).run_id)

        expected = dataset.val_images > 0
        for one in run.metrics:
            assert one.has("validation") is expected, one.values

    # ── Stopping ──────────────────────────────────────────────────────────────

    def test_cancel_stops_the_run_and_keeps_what_it_finished(
        self, provider: TrainingProvider, trainable: tuple[DatasetSpec, TrainingConfig]
    ) -> None:
        """ADR-0043's promise, at a trainer's checkpoint: the epoch boundary.

        A cancelled run keeps the epochs it completed, the way a cancelled import
        keeps the files it already copied — pretending nothing happened would
        throw away hours of real work.
        """
        dataset, config = trainable
        started = provider.start(dataset, config)
        await_condition(provider, started.run_id, lambda run: run.epochs_done >= 1)
        provider.cancel(started.run_id)
        run = await_terminal(provider, started.run_id)

        assert run.status is TrainingStatus.CANCELLED, (
            "a provider whose epochs are too short to interrupt should give this suite a "
            "longer configuration — the contract is that cancel is honoured, not that it "
            "is instant"
        )
        assert run.epochs_done >= 1
        assert run.epochs_done < config.epochs
        assert run.finished_utc

    def test_cancelling_an_unknown_run_is_not_an_error(self, provider: TrainingProvider) -> None:
        """The caller is a button that can be pressed twice."""
        provider.cancel("no-such-run")

    def test_cancelling_a_finished_run_does_nothing(
        self, provider: TrainingProvider, trainable: tuple[DatasetSpec, TrainingConfig]
    ) -> None:
        dataset, config = trainable
        run = await_terminal(provider, provider.start(dataset, config).run_id)
        provider.cancel(run.run_id)

        assert provider.status(run.run_id).status is TrainingStatus.SUCCEEDED

    # ── Telling ───────────────────────────────────────────────────────────────

    def test_the_listener_hears_the_run_reach_a_terminal_state(
        self, provider: TrainingProvider, trainable: tuple[DatasetSpec, TrainingConfig]
    ) -> None:
        """A UI that only ever saw `RUNNING` would show a spinner forever."""
        seen: list[TrainingRun] = []
        dataset, config = trainable
        started = provider.start(dataset, config, listener=seen.append)
        await_terminal(provider, started.run_id)
        # The last callback may land just after `status` turns terminal — it is
        # fired from the provider's own thread. Poll for it rather than sleeping
        # a fixed amount, which is a flake on a slow machine.
        deadline = time.monotonic() + TIMEOUT_S
        while time.monotonic() < deadline and not (seen and seen[-1].is_finished):
            time.sleep(POLL_S)

        assert seen, "the listener was never called"
        assert seen[-1].is_finished
        assert all(one.run_id == started.run_id for one in seen)
        assert [one.epochs_done for one in seen] == sorted(one.epochs_done for one in seen), (
            "snapshots arrive in order; a listener that can go backwards makes a "
            "progress bar move backwards"
        )
