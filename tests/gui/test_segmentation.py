"""Segmentation, and the masks it produces (M6-T04, ADR-0064).

M6-T02 disables every `segment` row with *"segmentation needs a loaded
predictor, which arrives in M6-T04"*. This is the file that makes the sentence
true — driven by a **stub predictor registered through the registry**, which is
the only way this path can be tested at all: there are no SAM2 weights here or in
CI, and M3-T14 set the precedent.

The other half is what the masks are allowed to claim. They are **not stored**
(ADR-0042), so the run carries them in memory and a run read back from the
project has none.
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
from nanoscope.gui.panels import DetectionPanel, ImageViewer
from nanoscope.gui.panels.viewer import _outline
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.models import registry
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


class StubPredictor:
    """What SAM2's predictor looks like from the pipeline's side.

    Enough of the shape for `run_sam2_from_blobs` to run: set an image, then
    answer a prompt with a mask, a score and logits. It returns a disc around
    the point it was given, which is what a real predictor would do on a phantom
    and what makes the assertions below about *shape* rather than about scores.
    """

    def __init__(self) -> None:
        self.loaded: list[Path] = []
        self._shape = (48, 48)

    def set_image(self, image: np.ndarray) -> None:
        self._shape = image.shape[:2]

    def predict(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        point = kwargs.get("point_coords")
        cy, cx = (24, 24) if point is None else (float(point[0][1]), float(point[0][0]))
        y, x = np.mgrid[0 : self._shape[0], 0 : self._shape[1]]
        mask = ((y - cy) ** 2 + (x - cx) ** 2) <= 9
        return mask[None, ...], np.array([0.9]), np.zeros((1, 4, 4))


@pytest.fixture(autouse=True)
def stub_registered() -> Iterator[None]:
    """A factory for the segmentation framework, and the real one back after."""
    before = registry._REGISTRY.get(ModelFramework.SAM2)
    built: list[Path] = []

    def factory(path: Path, device: object) -> StubPredictor:
        built.append(path)
        predictor = StubPredictor()
        predictor.loaded = built
        return predictor

    registry.register(ModelFramework.SAM2, factory)
    yield
    if before is not None:
        registry.register(ModelFramework.SAM2, before)


def phantom(size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    height = rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    for cy, cx in ((14, 14), (32, 30)):
        height += 4.0 * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / 12.0))
    return height.astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        source = tmp_path / "afm.npy"
        np.save(source, phantom())
        repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
    return root


def register_segmentation_model(session: SessionViewModel) -> None:
    repository = session._app.repository
    assert repository is not None
    weights = repository.root / "models" / "sam.pt"
    weights.parent.mkdir(exist_ok=True)
    weights.write_bytes(b"not really weights")
    repository.register_model(
        ModelDescriptor(
            model_id="sam",
            task=ModelTask.SEGMENT,
            framework=ModelFramework.SAM2,
            path="models/sam.pt",
        )
    )


@pytest.fixture
def session(tmp_path: Path, project: Path) -> Iterator[SessionViewModel]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        model = SessionViewModel(container)
        model.open_project(project)
        yield model


def image_ids(session: SessionViewModel) -> list[int]:
    assert session.project is not None
    return [image.id for image in session.project.images]


def settle(job: Job | None) -> None:
    assert job is not None
    assert job.wait(60.0)
    QApplication.processEvents()


class TestTheModeBecomesAvailable:
    def test_without_a_model_it_is_refused(self, session: SessionViewModel) -> None:
        session.select_image(image_ids(session)[0])
        panel = DetectionPanel(session)

        modes = {
            panel.mode.itemText(i): panel.mode.model().item(i).isEnabled()
            for i in range(panel.mode.count())
        }

        assert modes["segment"] is False

    def test_registering_one_makes_it_selectable(self, session: SessionViewModel) -> None:
        """M6-T02's promise, with a date on it."""
        register_segmentation_model(session)
        session.select_image(image_ids(session)[0])

        panel = DetectionPanel(session)
        modes = {
            panel.mode.itemText(i): panel.mode.model().item(i).isEnabled()
            for i in range(panel.mode.count())
        }

        assert modes["segment"] is True

    def test_asking_whether_it_can_segment_loads_nothing(self, session: SessionViewModel) -> None:
        """ADR-0050 made the registry cheap so that this question costs no disk.
        The weights are read inside the job, not to fill a combo box."""
        register_segmentation_model(session)
        session.select_image(image_ids(session)[0])

        session.detector_options()

        assert session._app._predictor is None


class TestTheRun:
    def test_the_predictor_is_built_once_and_produces_masks(
        self, session: SessionViewModel
    ) -> None:
        register_segmentation_model(session)
        session.select_image(image_ids(session)[0])

        settle(session.detect(PipelineConfig(detector="log", mode="segment")))
        run = session.run
        assert run is not None
        assert run.masks

        settle(session.detect(PipelineConfig(detector="log", mode="segment")))

        predictor = session._app.segmentation_predictor()
        assert isinstance(predictor, StubPredictor)
        assert len(predictor.loaded) == 1, "weights are loaded once per project"

    def test_a_stored_run_has_no_masks(self, session: SessionViewModel) -> None:
        """They are not persisted (ADR-0042), so a run read back from the
        project carries none — and the overlay says so by having nothing to
        draw rather than by drawing something stale."""
        register_segmentation_model(session)
        session.select_image(image_ids(session)[0])
        settle(session.detect(PipelineConfig(detector="log", mode="segment")))
        assert session.run is not None and session.run.masks

        repository = session._app.repository
        assert repository is not None
        reread = repository.get_run(session.run.id)

        assert reread.masks == ()
        assert reread.detections

    def test_closing_the_project_lets_the_predictor_go(self, session: SessionViewModel) -> None:
        register_segmentation_model(session)
        assert session._app.segmentation_predictor() is not None

        session.close_project()

        assert session._app._predictor is None

    def test_a_project_with_no_model_has_no_predictor(self, session: SessionViewModel) -> None:
        assert session._app.segmentation_predictor() is None


class TestTheMaskOverlay:
    def test_a_mask_is_drawn_as_an_outline(self) -> None:
        """Filled would hide the pixels it describes, and those pixels are the
        measurement."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 3:8] = True

        item = _outline(mask)

        assert item.brush().style() == item.brush().style().NoBrush
        assert item.pen().isCosmetic()
        rect = item.path().boundingRect()
        assert (rect.x(), rect.y(), rect.width(), rect.height()) == (3, 2, 5, 4)

    def test_the_run_puts_them_on_the_scene(self, session: SessionViewModel) -> None:
        register_segmentation_model(session)
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])

        settle(session.detect(PipelineConfig(detector="log", mode="segment")))

        assert session.run is not None
        assert len(viewer.view.mask_overlay) == len(session.run.masks)
        assert viewer.show_masks.isVisible() or viewer.show_masks.isVisibleTo(viewer)

    def test_they_can_be_turned_off(self, session: SessionViewModel) -> None:
        register_segmentation_model(session)
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])
        settle(session.detect(PipelineConfig(detector="log", mode="segment")))

        viewer.show_masks.setChecked(False)

        assert viewer.view.mask_overlay == []

    def test_the_toggle_is_hidden_when_there_are_none(self, session: SessionViewModel) -> None:
        """A control for something that does not exist teaches an operator to
        ignore the row it sits in."""
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])

        settle(session.detect(PipelineConfig(detector="log", mode="detect")))

        assert not viewer.show_masks.isVisibleTo(viewer)
