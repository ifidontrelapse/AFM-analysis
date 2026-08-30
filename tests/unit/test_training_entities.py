"""What a training run refuses to be (M8-T01).

The port is checked by `tests/contract/`; this file checks the records it speaks
in, and the assertions worth having are the refusals. Two of them carry a
decision each: **the metric vocabulary is declared once** (ADR-0031's rule, one
milestone on and for a second kind of producer), and **a block is present in full
or absent in full** — because a provider that reports precision without recall
has measured half a thing and lost the other half, which is a different statement
from having measured neither.
"""

from __future__ import annotations

import pytest

from nanoscope.core.entities.training import (
    METRIC_BLOCKS,
    METRIC_NAMES,
    DatasetSpec,
    EpochMetrics,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from nanoscope.core.errors import InvalidParameterError


def _dataset(**overrides: object) -> DatasetSpec:
    return DatasetSpec(
        **{"root": "cache/ds", "classes": ("particle",), "train_images": 8, **overrides}
    )  # type: ignore[arg-type]


def _config(**overrides: object) -> TrainingConfig:
    return TrainingConfig(**{"base_model": "n.pt", "epochs": 10, "image_size_px": 640, **overrides})  # type: ignore[arg-type]


# ── The dataset ───────────────────────────────────────────────────────────────


def test_a_dataset_names_where_it_is_and_what_is_in_it() -> None:
    spec = _dataset(val_images=2)
    assert spec.root == "cache/ds"
    assert spec.classes == ("particle",)


@pytest.mark.parametrize(
    ("overrides", "because"),
    [
        ({"root": ""}, "a run must say what it trained on"),
        ({"classes": ()}, "a detector with nothing to detect"),
        ({"train_images": 0}, "no training image trains nothing"),
        ({"val_images": -1}, "a negative count is not a split"),
    ],
)
def test_a_dataset_that_cannot_be_trained_on_is_refused(
    overrides: dict[str, object], because: str
) -> None:
    with pytest.raises(InvalidParameterError):
        _dataset(**overrides)


def test_holding_nothing_out_is_legal() -> None:
    """`val_images == 0` is a choice, not an error — and it is what makes the
    `validation` metric block conditional rather than always-there."""
    assert _dataset(val_images=0).val_images == 0


# ── The configuration ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_model": ""},
        {"epochs": 0},
        {"epochs": -1},
        {"image_size_px": 0},
        {"batch_size": 0},
    ],
)
def test_a_configuration_that_describes_no_run_is_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(InvalidParameterError):
        _config(**overrides)


def test_what_the_framework_may_decide_is_none_and_not_a_guess() -> None:
    """ADR-0019's rule, applied to a hyperparameter: unknown is `None`, never a
    number this layer invented to fill the field."""
    config = _config()
    assert config.batch_size is None
    assert config.device is None
    assert config.seed is None


# ── The metrics ───────────────────────────────────────────────────────────────


def test_an_epoch_may_report_only_its_loss() -> None:
    one = EpochMetrics(epoch=1, values={"train_loss": 0.4})
    assert one.has("loss")
    assert not one.has("validation")


def test_an_epoch_may_report_nothing_at_all() -> None:
    assert EpochMetrics(epoch=1).values == {}


def test_a_name_outside_the_vocabulary_is_refused() -> None:
    """The failure ADR-0031 was written about: two producers, two spellings of
    one quantity, and a chart that cannot tell they are the same column."""
    with pytest.raises(InvalidParameterError, match="unknown metric"):
        EpochMetrics(epoch=1, values={"mAP50": 0.8})


def test_half_a_block_is_refused() -> None:
    with pytest.raises(InvalidParameterError, match="partial"):
        EpochMetrics(epoch=1, values={"precision": 0.9, "recall": 0.8})


def test_a_whole_block_is_accepted() -> None:
    values = dict.fromkeys(METRIC_BLOCKS["validation"], 0.5) | {"train_loss": 0.4}
    assert EpochMetrics(epoch=3, values=values).has("validation")


def test_epochs_are_numbered_from_one() -> None:
    with pytest.raises(InvalidParameterError, match="numbered from 1"):
        EpochMetrics(epoch=0, values={"train_loss": 0.4})


def test_the_vocabulary_has_no_name_in_two_blocks() -> None:
    """One quantity, one name — and one home. Two blocks sharing a name would
    make "is this block present?" answerable two ways."""
    total = sum(len(names) for names in METRIC_BLOCKS.values())
    assert total == len(METRIC_NAMES)


# ── The run ───────────────────────────────────────────────────────────────────


def _run(**overrides: object) -> TrainingRun:
    return TrainingRun(
        **{
            "run_id": "r1",
            "status": TrainingStatus.RUNNING,
            "dataset": _dataset(),
            "config": _config(),
            **overrides,
        }  # type: ignore[arg-type]
    )


def test_a_fresh_run_has_finished_nothing() -> None:
    run = _run()
    assert run.epochs_done == 0
    assert not run.is_finished
    assert run.weights_path is None


def test_progress_is_derived_from_the_last_report() -> None:
    """Not counted separately, so the number on screen and the metrics behind it
    cannot disagree."""
    run = _run(
        metrics=tuple(EpochMetrics(epoch=n, values={"train_loss": 1 / n}) for n in (1, 2, 3))
    )
    assert run.epochs_done == 3


@pytest.mark.parametrize(
    ("status", "finished"),
    [
        (TrainingStatus.PENDING, False),
        (TrainingStatus.RUNNING, False),
        (TrainingStatus.SUCCEEDED, True),
        (TrainingStatus.FAILED, True),
        (TrainingStatus.CANCELLED, True),
    ],
)
def test_which_states_are_terminal(status: TrainingStatus, finished: bool) -> None:
    assert _run(status=status).is_finished is finished
