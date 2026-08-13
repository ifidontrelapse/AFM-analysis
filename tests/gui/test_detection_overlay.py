"""The detections, drawn where they were found (M6-T03, ADR-0063).

What is asserted is the *shape* and the *place*: a box detector gets a box, a
blob detector gets a circle, and both sit in scene coordinates so the view's own
transform keeps them on their particles at every zoom.

The other half is which run is on screen — the newest stored one for the selected
image, which is the first time `runs_for` has had a reader.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsRectItem

from nanoscope.app.container import Nanoscope
from nanoscope.application.jobs import Job
from nanoscope.core.entities import Detection, PipelineConfig
from nanoscope.core.values import Modality
from nanoscope.gui.panels import DetectionPanel, ImageViewer
from nanoscope.gui.panels.viewer import _shape
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def phantom(size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(2)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    height = rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    for cy, cx in ((14, 14), (30, 34)):
        height += 4.0 * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / 12.0))
    return height.astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name in ("one.npy", "two.npy"):
            source = tmp_path / name
            np.save(source, phantom())
            repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
    return root


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
    assert job.wait(30.0)
    QApplication.processEvents()


def analyse(session: SessionViewModel) -> None:
    settle(session.detect(PipelineConfig(detector="log", mode="detect")))


class TestTheShapeIsWhatWasFound:
    def test_a_box_detection_draws_a_box(self) -> None:
        item = _shape(
            Detection(x_px=10, y_px=20, radius_px=4, radius_nm=None, bbox=(5, 15, 15, 25))
        )

        assert isinstance(item, QGraphicsRectItem)
        assert item.rect().width() == 10

    def test_a_detection_without_one_draws_a_circle(self) -> None:
        """`bbox` is `None` on the blob path (ADR-0031); an invented box around
        a circle is a shape nothing found."""
        item = _shape(Detection(x_px=10, y_px=20, radius_px=4, radius_nm=None))

        assert isinstance(item, QGraphicsEllipseItem)
        assert item.rect().center().x() == pytest.approx(10)
        assert item.rect().width() == pytest.approx(8)

    def test_the_outline_is_cosmetic(self) -> None:
        """A pen measured in scene units turns a circle into a filled blob at
        32x — the zoom this viewer allows."""
        item = _shape(Detection(x_px=1, y_px=1, radius_px=2, radius_nm=None))

        assert item.pen().isCosmetic()
        assert item.brush().style() == item.brush().style().NoBrush


class TestWhatIsOnScreen:
    def test_a_run_puts_one_item_on_each_particle(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])

        analyse(session)

        run = session.run
        assert run is not None and run.detections
        assert len(viewer.view.overlay) == len(run.detections)
        assert viewer.show_detections.text() == f"Detections ({len(run.detections)})"

    def test_the_items_are_in_scene_coordinates(self, session: SessionViewModel) -> None:
        """Placed on the scene, so the view's own transform keeps them on their
        particles at every zoom — no arithmetic of ours."""
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])
        analyse(session)

        run = session.run
        assert run is not None
        first = run.detections[0]
        centres = [item.sceneBoundingRect().center() for item in viewer.view.overlay]

        assert any(
            abs(centre.x() - first.x_px) < 1 and abs(centre.y() - first.y_px) < 1
            for centre in centres
        )

    def test_turning_it_off_empties_the_scene_and_says_so(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])
        analyse(session)

        viewer.show_detections.setChecked(False)

        assert viewer.view.overlay == []
        #: The count stays on the box; the box being unticked is what says the
        #: overlay is off, and "Detections (0)" says the run found none.
        assert "(2" in viewer.show_detections.text() or "(1" in viewer.show_detections.text()

    def test_selecting_another_image_clears_them(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])
        analyse(session)

        session.select_image(image_ids(session)[1])

        assert session.run is None
        assert viewer.view.overlay == []
        assert viewer.show_detections.text() == "Detections"

    def test_the_newest_stored_run_is_shown_when_an_image_is_selected(
        self, session: SessionViewModel
    ) -> None:
        """`runs_for` has existed since M4-T05 and nothing read it: a scan
        analysed yesterday showed nothing today."""
        session.select_image(image_ids(session)[0])
        analyse(session)
        first_run = session.run
        assert first_run is not None

        session.select_image(image_ids(session)[1])
        session.select_image(image_ids(session)[0])

        assert session.run is not None
        assert session.run.id == first_run.id

    def test_a_new_run_replaces_the_old_one(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])
        analyse(session)
        first = session.run
        assert first is not None

        analyse(session)

        assert session.run is not None
        assert session.run.id != first.id
        assert len(viewer.view.overlay) == len(session.run.detections)

    def test_the_panel_and_the_viewer_agree(self, session: SessionViewModel) -> None:
        """Two panels, one run — neither is told by the other (ADR-0057)."""
        viewer = ImageViewer(session)
        panel = DetectionPanel(session)
        session.select_image(image_ids(session)[0])

        panel.start()
        settle(session.job)

        assert session.run is not None
        assert f"{len(session.run.detections)} detection(s)" in panel.report.text()
        assert len(viewer.view.overlay) == len(session.run.detections)
