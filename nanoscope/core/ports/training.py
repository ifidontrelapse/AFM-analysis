"""The contract every trainer satisfies (M8-T01).

The fifth port to pay out the table in `core/ports/__init__.py`, and the first
one written **before** its adapter. That is a departure from the rule this
package wrote for itself — *the rest ship with their first adapter* — and
ADR-0080 §1 argues why this one is the exception rather than the erosion:
ADR-0006 committed to two implementations of it in M0, the second of them on
another machine, and a port discovered from the first implementation is that
implementation's shape with an `abstract` keyword on it.

What keeps it from being a guess is not this file. It is
`tests/contract/training_provider.py`: the suite the fake here passes today and
`LocalTrainingProvider` passes in M8-T03, which is ADR-0006's own compliance
clause — *both providers pass the same contract test suite* — as a test rather
than a review note.

**Three methods, and no fourth.** Start a run, ask about one, ask one to stop.
There is no `collect_artifacts`: `TrainingConfig.output_directory` says where the
weights go and a succeeded run carries the path, so the provider that knows how
to move a file — a copy locally, a download remotely — is the one that does it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from nanoscope.core.entities.training import DatasetSpec, TrainingConfig, TrainingRun


@runtime_checkable
class TrainingProvider(Protocol):
    """Turns a dataset into a model, here or somewhere else.

    `runtime_checkable` like `Detector`, and with the same limit: it proves the
    methods exist, never their signatures. Signatures are mypy's job, asserted
    structurally in the contract suite.
    """

    def start(
        self,
        dataset: DatasetSpec,
        config: TrainingConfig,
        listener: Callable[[TrainingRun], None] | None = None,
    ) -> TrainingRun:
        """Begin a run and return it immediately, before it has trained anything.

        Args:
            dataset: what to train on, as M8-T02's builder produced it.
            config: what to train, for how long, and where the result goes.
            listener: called with a fresh snapshot on every status change and
                every completed epoch. **It does not run on the caller's
                thread** — the local provider fires it from the worker that
                ADR-0043 put the work on, and a remote one from whatever polls.
                A Qt caller marshals, exactly as it does for a `Job` (ADR-0058).

        Returns:
            The run, `PENDING` or `RUNNING`, carrying the `run_id` the other two
            methods take. Never a finished one: a `start` that returned only when
            the training was over would be the multi-hour blocking call this port
            exists to avoid.

        Raises:
            InvalidParameterError: the dataset or the configuration cannot be
                trained on — a missing dataset directory, weights that are not
                there. Refused here rather than reported as a run that failed
                four seconds in, because nothing was ever started.
        """
        ...

    def status(self, run_id: str) -> TrainingRun:
        """The run as it is now.

        A fresh snapshot every call. Cheap enough to poll, which is what a
        provider on the other side of a network is doing anyway, and what a UI
        falls back to when it has missed a listener callback.

        Raises:
            InvalidParameterError: no run by that id. A provider that invented an
                empty run for an unknown id would let a typo look like a run that
                had not started yet.
        """
        ...

    def cancel(self, run_id: str) -> None:
        """Ask a run to stop. **Ask**, because it cannot be made to.

        ADR-0043 settled what a cancel button may promise — *stop at the next
        checkpoint* — and for a trainer the checkpoint is an **epoch boundary**.
        A run cancelled two minutes into a forty-minute epoch keeps training for
        thirty-eight, and the UI says so (M8-T05), because the alternative is a
        button that appears to do nothing and an operator who concludes the
        application has hung. M5-T07 learned this once already.

        Cancelling a finished or unknown run does nothing, and does not raise:
        the caller is a button that can be pressed twice, and the second press
        should not be an error dialog.

        A cancelled run keeps the epochs it completed and whatever checkpoint the
        framework last wrote — the same honesty ADR-0043 gave a cancelled import,
        which keeps the files it had already copied.
        """
        ...
