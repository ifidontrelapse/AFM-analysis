"""The contract suite, run against the fake (M8-T01).

The file that makes `tests/contract/training_provider.py` a test rather than a
specification. M8-T03 adds one beside it with three different fixtures and not a
line of new assertions — which is the deliverable this task actually promised.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fake_provider import FakeTrainingProvider
from training_provider import TrainingProviderContract

from nanoscope.core.entities.training import DatasetSpec, TrainingConfig
from nanoscope.core.ports import TrainingProvider


def _dataset(project_root: Path, *, val_images: int) -> DatasetSpec:
    (project_root / "cache" / "dataset").mkdir(parents=True, exist_ok=True)
    return DatasetSpec(
        root="cache/dataset",
        classes=("particle",),
        train_images=8,
        val_images=val_images,
    )


class TestFakeTrainingProvider(TrainingProviderContract):
    """A held-out set, which is the ordinary case."""

    @pytest.fixture
    def provider(self, project_root: Path) -> TrainingProvider:
        return FakeTrainingProvider(project_root)

    @pytest.fixture
    def trainable(self, project_root: Path) -> tuple[DatasetSpec, TrainingConfig]:
        return _dataset(project_root, val_images=2), TrainingConfig(
            base_model="fake.pt",
            epochs=6,
            image_size_px=640,
            output_directory="models/run",
        )


class TestFakeTrainingProviderWithoutValidation(TestFakeTrainingProvider):
    """The same provider with nothing held out — ADR-0031's absent block, live.

    Inherits every assertion, which is the point: the split changes what the
    metrics contain and changes nothing else about the contract.
    """

    @pytest.fixture
    def trainable(self, project_root: Path) -> tuple[DatasetSpec, TrainingConfig]:
        return _dataset(project_root, val_images=0), TrainingConfig(
            base_model="fake.pt",
            epochs=6,
            image_size_px=640,
            output_directory="models/run",
        )


def test_the_contract_can_fail(project_root: Path) -> None:
    """A suite nothing can fail is decoration. This proves it detects a liar.

    The failure chosen is the one ADR-0006's compliance clause names: a run that
    reports success and leaves no file behind.
    """

    class SilentProvider(FakeTrainingProvider):
        def _train(self, run_id: str, listener: object = None) -> None:  # type: ignore[override]
            from nanoscope.core.entities.training import TrainingStatus

            self._publish(
                run_id, TrainingStatus.SUCCEEDED, finished=True, weights_path="models/gone.pt"
            )

    provider = SilentProvider(project_root)
    dataset = _dataset(project_root, val_images=2)
    config = TrainingConfig(base_model="fake.pt", epochs=2, image_size_px=640)

    contract = TestFakeTrainingProvider()
    with pytest.raises(AssertionError):
        contract.test_a_finished_run_succeeded_and_its_weights_are_on_disk(
            provider, (dataset, config), project_root
        )


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path
