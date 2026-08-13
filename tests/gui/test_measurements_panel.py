"""The measurements, beside the particles they belong to (M6-T05, ADR-0065).

M6's second exit criterion — *"selecting a table row highlights the particle, and
vice versa"* — and the two things that make it honest:

- the link is a **coordinate**, because the table is a subset of the detections
  (a height that is not a number is discarded, ADR-0033);
- the columns are the **producer's own** (ADR-0031), so the panel and the
  exported CSV say the same words.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from nanoscope.app.container import Nanoscope
from nanoscope.application.jobs import Job
from nanoscope.core.entities import PipelineConfig
from nanoscope.core.values import Modality
from nanoscope.gui.panels import ImageViewer, MeasurementsPanel
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def phantom(size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(4)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    height = rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    for cy, cx in ((14, 14), (32, 30), (14, 34)):
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


@pytest.fixture
def session(tmp_path: Path, project: Path) -> Iterator[SessionViewModel]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        model = SessionViewModel(container)
        model.open_project(project)
        model.select_image(model.project.images[0].id)  # type: ignore[union-attr]
        yield model


def settle(job: Job | None) -> None:
    assert job is not None
    assert job.wait(60.0)
    QApplication.processEvents()


def measure(session: SessionViewModel) -> None:
    """A mode that actually writes a table."""
    settle(session.detect(PipelineConfig(detector="log", mode="baseline")))


class TestWhatTheTableShows:
    def test_it_is_the_stored_table_with_its_own_column_names(
        self, session: SessionViewModel
    ) -> None:
        panel = MeasurementsPanel(session)

        measure(session)

        stored = session.measurements()
        assert stored is not None and len(stored)
        assert panel.table.rowCount() == len(stored)
        headers = [
            panel.table.horizontalHeaderItem(i).text() for i in range(panel.table.columnCount())
        ]
        assert headers == [str(name) for name in stored.columns]
        assert "x_px" in headers and "particle_id" in headers

    def test_a_run_that_measured_nothing_says_so(self, session: SessionViewModel) -> None:
        """`detect` writes no table at all (ADR-0042); columns with no rows
        under them would claim it found nothing rather than asked for nothing."""
        panel = MeasurementsPanel(session)

        settle(session.detect(PipelineConfig(detector="log", mode="detect")))

        assert panel.table.rowCount() == 0
        assert "measured nothing" in panel.note.text()

    def test_nothing_selected_shows_nothing(self, session: SessionViewModel) -> None:
        panel = MeasurementsPanel(session)

        assert panel.table.rowCount() == 0
        assert panel.note.text() == ""

    def test_the_table_is_a_subset_of_the_detections(self, session: SessionViewModel) -> None:
        """Which is why the link is a coordinate and not an index: a height that
        is not a number never reaches the table (ADR-0033)."""
        measure(session)

        stored = session.measurements()
        run = session.run
        assert stored is not None and run is not None
        assert len(stored) <= len(run.detections)


class TestBothDirections:
    def test_a_row_selects_the_particle(self, session: SessionViewModel) -> None:
        panel = MeasurementsPanel(session)
        measure(session)

        panel.table.selectRow(0)

        stored = session.measurements()
        run = session.run
        assert stored is not None and run is not None
        expected = session.particle_at(float(stored.iloc[0]["x_px"]), float(stored.iloc[0]["y_px"]))
        assert session.particle == expected
        assert expected is not None

    def test_a_click_on_the_canvas_selects_the_row(self, session: SessionViewModel) -> None:
        """The other direction, and neither widget tells the other: both talk to
        the session (ADR-0057, ADR-0065)."""
        panel = MeasurementsPanel(session)
        viewer = ImageViewer(session)
        measure(session)
        stored = session.measurements()
        assert stored is not None
        target = session.particle_at(float(stored.iloc[1]["x_px"]), float(stored.iloc[1]["y_px"]))
        assert target is not None

        viewer.view.picked.emit(target)

        assert session.particle == target
        assert {index.row() for index in panel.table.selectedIndexes()} == {1}

    def test_the_selected_particle_is_the_thicker_outline(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        measure(session)

        session.select_particle(1)

        widths = [item.pen().widthF() for item in viewer.view.overlay]
        assert widths[1] == max(widths)
        assert widths[0] < widths[1]

    def test_a_click_on_bare_image_clears_the_selection(self, session: SessionViewModel) -> None:
        panel = MeasurementsPanel(session)
        viewer = ImageViewer(session)
        measure(session)
        session.select_particle(0)

        viewer.view.picked.emit(None)

        assert session.particle is None
        assert panel.table.selectedIndexes() == []

    def test_an_index_no_run_has_is_refused(self, session: SessionViewModel) -> None:
        measure(session)

        session.select_particle(10_000)

        assert session.particle is None

    def test_a_new_run_clears_the_selection(self, session: SessionViewModel) -> None:
        measure(session)
        session.select_particle(0)

        measure(session)

        assert session.particle is None
