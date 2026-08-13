"""What is still there tomorrow (M6-T09, ADR-0069).

M6's fourth exit criterion: *"results persist across application restart"*. Most
of it is true by construction — `run_analysis` stores a run, its detections and
its measurement table (ADR-0042) — so what this file does is **prove it**, and
prove it the only way that means anything: the container is closed, the
repository connection with it, and a **new** container and a **new** window open
the same directory. Anything less proves the cache.

The other half is what deliberately does not come back, and says so.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from nanoscope.app.container import Nanoscope
from nanoscope.application.jobs import Job
from nanoscope.core.entities import PipelineConfig
from nanoscope.core.entities.model import ModelDescriptor, ModelFramework, ModelTask
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.models import registry
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def phantom(size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    height = rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    for cy, cx in ((14, 14), (32, 30)):
        height += 5.0 * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / 10.0))
    return height.astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        source = tmp_path / "afm.npy"
        np.save(source, phantom())
        repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
    return root


def settle(job: Job | None) -> None:
    assert job is not None
    assert job.wait(60.0)
    QApplication.processEvents()


def session_for(tmp_path: Path, project: Path) -> Iterator[SessionViewModel]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        model = SessionViewModel(container)
        model.open_project(project)
        assert model.project is not None
        model.select_image(model.project.images[0].id)
        yield model


class TestARunSurvivesTheProcess:
    def test_the_window_shows_it_again_after_a_restart(self, tmp_path: Path, project: Path) -> None:
        """A new container **and** a new window: the first proves the storage,
        the second proves that what reads it is the application and not a
        variable somebody kept."""
        for session in session_for(tmp_path, project):
            settle(session.detect(PipelineConfig(detector="log", mode="baseline")))
            assert session.run is not None
            before = (session.run.id, len(session.run.detections))
            rows = len(session.measurements())  # type: ignore[arg-type]

        with Nanoscope(settings_path=tmp_path / "settings.json") as container:
            window = MainWindow(container)
            opened = window.open_project(project)
            assert opened is not None
            window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

            assert window.session.run is not None
            assert (window.session.run.id, len(window.session.run.detections)) == before
            assert window.measurements.table.rowCount() == rows
            assert len(window.viewer.view.overlay) == before[1]
            assert window.statistics.column.count() > 0

    def test_the_exported_file_is_still_there_too(self, tmp_path: Path, project: Path) -> None:
        for session in session_for(tmp_path, project):
            settle(session.detect(PipelineConfig(detector="log", mode="baseline")))
            settle(session.export(everything=True))

        assert list((project / "exports").glob("*.csv"))


class TestOlderRunsAreReachable:
    def test_every_run_is_offered_and_can_be_shown(self, tmp_path: Path, project: Path) -> None:
        """Three analyses of one scan leave three rows, and reaching only the
        newest is "results persist" satisfied on a technicality."""
        for session in session_for(tmp_path, project):
            settle(session.detect(PipelineConfig(detector="log", mode="detect")))
            first = session.run
            settle(session.detect(PipelineConfig(detector="log", mode="baseline")))
            second = session.run
            assert first is not None and second is not None

            assert [run.id for run in session.runs()] == [first.id, second.id]
            assert session.run.id == second.id  # type: ignore[union-attr]

            assert session.select_run(first.id) is True
            assert session.run.id == first.id  # type: ignore[union-attr]

    def test_the_panel_offers_them_after_a_restart(self, tmp_path: Path, project: Path) -> None:
        for session in session_for(tmp_path, project):
            settle(session.detect(PipelineConfig(detector="log", mode="detect")))
            settle(session.detect(PipelineConfig(detector="log", mode="baseline")))

        with Nanoscope(settings_path=tmp_path / "settings.json") as container:
            window = MainWindow(container)
            window.open_project(project)
            window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

            assert window.measurements.run.count() == 2
            window.measurements.run.setCurrentIndex(0)

            assert window.session.run is not None
            assert window.session.run.mode == "detect"
            assert "measured nothing" in window.measurements.note.text()

    def test_a_run_of_another_image_is_refused(self, tmp_path: Path, project: Path) -> None:
        for session in session_for(tmp_path, project):
            assert session.select_run(9_999) is False


class TestWhatDoesNotComeBack:
    def test_a_restored_segmentation_run_says_its_masks_were_not_stored(
        self, tmp_path: Path, project: Path
    ) -> None:
        """An empty overlay reads as "segmentation found nothing". It is not
        nothing — it is not stored (ADR-0042, ADR-0064)."""
        before = registry._REGISTRY.get(ModelFramework.SAM2)
        registry.register(ModelFramework.SAM2, lambda path, device: _StubPredictor())
        try:
            with SqliteProjectRepository.open(project) as repo:
                weights = repo.root / "models" / "sam.pt"
                weights.parent.mkdir(exist_ok=True)
                weights.write_bytes(b"not really weights")
                repo.register_model(
                    ModelDescriptor(
                        model_id="sam",
                        task=ModelTask.SEGMENT,
                        framework=ModelFramework.SAM2,
                        path="models/sam.pt",
                    )
                )

            for session in session_for(tmp_path, project):
                settle(session.detect(PipelineConfig(detector="log", mode="segment")))
                assert session.run is not None and session.run.masks

            with Nanoscope(settings_path=tmp_path / "settings.json") as container:
                window = MainWindow(container)
                window.open_project(project)
                window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

                assert window.session.run is not None
                assert window.session.run.masks == ()
                assert "not stored" in window.measurements.note.text()
                assert window.viewer.view.mask_overlay == []
        finally:
            if before is not None:
                registry.register(ModelFramework.SAM2, before)


class _StubPredictor:
    """A disc where it is prompted — enough shape for the pipeline (M6-T04)."""

    def __init__(self) -> None:
        self._shape = (48, 48)

    def set_image(self, image: np.ndarray) -> None:
        self._shape = image.shape[:2]

    def predict(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        point = kwargs.get("point_coords")
        cy, cx = (24, 24) if point is None else (float(point[0][1]), float(point[0][0]))
        y, x = np.mgrid[0 : self._shape[0], 0 : self._shape[1]]
        return (
            (((y - cy) ** 2 + (x - cx) ** 2) <= 9)[None, ...],
            np.array([0.9]),
            np.zeros((1, 4, 4)),
        )
