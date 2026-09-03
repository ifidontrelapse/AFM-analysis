"""A run the project remembers, and the model it produced (M8-T04).

M8-T01 wrote the reason for this module into the entity it defined: *"a `Job` is
in-process and dies with the process; a training run has to be findable after a
restart."* M8-T03 then produced the first model this project has ever made and
left the other half of ADR-0006's compliance clause — *the trained model is
registered as a `ModelDescriptor`* — in its own out-of-scope list, addressed
here.

So this is the module where a run stops being a dict entry. It starts one through
the port, records **every snapshot the provider publishes**, and registers what a
succeeded run left on disk.

**The provider is not handed a repository.** `infrastructure/training/` may not
name storage and does not; the listener the port already defines is the seam, and
what to keep is a policy, which is what a use case is for (ADR-0041). The
snapshot `start` *returns* is deliberately not written: that would be a write
from the calling thread racing the worker's first callback, and the loser is
whichever lands last — a `pending` row over a `succeeded` one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from nanoscope.core.entities.model import ModelDescriptor, ModelFramework, ModelTask
from nanoscope.core.entities.training import (
    DatasetSpec,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.ports import ProjectRepository, TrainingProvider

logger = logging.getLogger(__name__)


def start_training(
    repository: ProjectRepository,
    provider: TrainingProvider,
    dataset: DatasetSpec,
    config: TrainingConfig,
    *,
    model_id: str,
    listener: Callable[[TrainingRun], None] | None = None,
) -> TrainingRun:
    """Train, keep the record, and register what the run produced.

    Args:
        repository: the open project. It is the memory: `TrainingProvider.status`
            knows only the runs this process started, and a run has to be
            findable tomorrow.
        provider: who does the training — here or on another machine.
        dataset: what to train on, as `build_dataset` produced it.
        config: what to train, for how long, and where the weights go.
        model_id: what the produced model will be called in this project. An
            operator names their model (ADR-0050); it is not derived from the
            run's id, because a configuration naming a UUID is one nobody can
            read. Registering under an id that already exists replaces it,
            which is what retraining means.
        listener: called with every snapshot, **on the provider's thread**, after
            the record has been written — so a UI that reacts to `SUCCEEDED`
            finds the model already registered. A Qt caller marshals (ADR-0058).

    Returns:
        The run as `start` returned it: `PENDING` or `RUNNING`, never finished.

    Raises:
        InvalidParameterError: `model_id` is empty, or the provider refused the
            dataset or the configuration outright — nothing was started, so
            nothing is recorded.
    """
    if not model_id:
        raise InvalidParameterError(
            "model_id is empty: a run's result is registered under a name a "
            "configuration can use, and an unnamed model cannot be selected"
        )

    def record(run: TrainingRun) -> None:
        repository.save_training_run(run)
        if run.status is TrainingStatus.SUCCEEDED and run.weights_path:
            repository.register_model(descriptor_for(run, model_id=model_id))
            logger.info("registered %r from training run %s", model_id, run.run_id)
        if listener is not None:
            listener(run)

    return provider.start(dataset, config, listener=record)


def descriptor_for(run: TrainingRun, *, model_id: str) -> ModelDescriptor:
    """What a finished run says about the model it produced.

    Everything except the name comes off the run, because everything except the
    name *is* on the run — which is what ADR-0080 §5 bought by refusing a
    `collect_artifacts()` that could disagree with it, and what `DatasetSpec`
    carrying its classes and counts was for
    (ADR-0081: the dataset directory is in `cache/` and may be gone by the time
    somebody reads this row).

    Raises:
        InvalidParameterError: the run produced no weights. A model row pointing
            at a file a cancelled run never wrote is exactly the disagreement
            ADR-0006's *no silent artifacts on disk* rules out.
    """
    if not run.weights_path:
        raise InvalidParameterError(
            f"training run {run.run_id} produced no weights ({run.status}): "
            "there is no model to register"
        )
    return ModelDescriptor(
        model_id=model_id,
        # The only task this project trains. A segmentation model is imported,
        # not trained here (M8-T06), and inventing a second value from a run
        # that cannot produce one would be a guess with a field to live in.
        task=ModelTask.DETECT,
        # The dataset this port consumes is the one M8-T02 builds — a `data.yaml`
        # and normalised label files — so whatever trains it, here or on another
        # machine (M8-T07), produces weights ultralytics loads. A parameter for a
        # value with one possible answer is a question nobody can answer better.
        framework=ModelFramework.ULTRALYTICS,
        path=run.weights_path,
        input_size_px=run.config.image_size_px,
        class_map=dict(enumerate(run.dataset.classes)),
        provenance=_provenance(run),
    )


def _provenance(run: TrainingRun) -> str:
    """Where a model came from, in a sentence (ADR-0005: free text, on purpose).

    Free text *because* provenance that must fit a schema stops being recorded —
    but the facts in it are the ones a reader asks for first: what it trained on,
    how much of it, for how long, from what, and on which machine's device.
    """
    device = f" on {run.device}" if run.device is not None else ""
    return (
        f"trained {run.finished_utc or run.started_utc} from {run.config.base_model}"
        f"{device}: {run.epochs_done} of {run.config.epochs} epoch(s) on "
        f"{run.dataset.train_images} training image(s), {run.dataset.val_images} held out "
        f"(dataset {run.dataset.root}, run {run.run_id})"
    )
