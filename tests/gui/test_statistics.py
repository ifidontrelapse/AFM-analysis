"""What a run says about the sample (M6-T06, ADR-0066).

The numbers are checked against `numpy` on the same values, because that is the
only assertion that means anything: a statistics panel is wrong in exactly one
interesting way, which is quietly including a `NaN` or an identifier column.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from nanoscope.app.container import Nanoscope
from nanoscope.application.jobs import Job
from nanoscope.application.use_cases.statistics import (
    BIN_RULE,
    histogram,
    numeric_columns,
    summarise,
)
from nanoscope.core.entities import PipelineConfig
from nanoscope.core.values import Modality
from nanoscope.gui.panels import StatisticsPanel
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "particle_id": [0, 1, 2, 3],
            "radius_nm": [10.0, 12.0, 14.0, np.nan],
            "x_px": [1.0, 2.0, 3.0, 4.0],
            "nothing_nm": [np.nan] * 4,
            "method": ["baseline_circle"] * 4,
        }
    )


def phantom(size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(5)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    height = rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    for cy, cx in ((14, 14), (32, 30), (14, 34)):
        height += 5.0 * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / 10.0))
    return height.astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name, scale in (("scaled.npy", 2.0), ("unscaled.npy", None)):
            source = tmp_path / name
            np.save(source, phantom())
            repo.import_image(source, modality=Modality.AFM, pixel_size_nm=scale)
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


def measure(session: SessionViewModel) -> None:
    job = session.detect(PipelineConfig(detector="log", mode="baseline"))
    assert isinstance(job, Job)
    assert job.wait(60.0)
    QApplication.processEvents()


class TestTheNumbers:
    def test_they_are_numpys_over_the_finite_values(self) -> None:
        summary = summarise(table(), "radius_nm")

        assert summary is not None
        finite = np.array([10.0, 12.0, 14.0])
        assert summary.count == 3
        assert summary.mean == pytest.approx(float(np.mean(finite)))
        assert summary.median == pytest.approx(float(np.median(finite)))
        assert summary.std == pytest.approx(float(np.std(finite, ddof=1)))
        assert (summary.minimum, summary.maximum) == (10.0, 14.0)

    def test_a_single_particle_has_no_spread(self) -> None:
        """`ddof=1` is undefined for one value, and reported as such rather than
        as a zero somebody could mistake for a measurement."""
        summary = summarise(pd.DataFrame({"radius_nm": [11.0]}), "radius_nm")

        assert summary is not None
        assert summary.count == 1
        assert not np.isfinite(summary.std)

    def test_a_column_with_nothing_finite_has_no_summary(self) -> None:
        assert summarise(table(), "nothing_nm") is None
        assert summarise(table(), "not_a_column") is None


class TestWhichColumns:
    def test_identifiers_are_not_measurements(self) -> None:
        """Averaging a `particle_id` produces a number with no meaning."""
        assert "particle_id" not in numeric_columns(table())

    def test_absent_columns_are_not_offered(self) -> None:
        """Every `_nm` column of an unscaled scan is `NaN`, and `nan ± nan` is
        not a statistic (ADR-0025)."""
        assert "nothing_nm" not in numeric_columns(table())

    def test_physical_quantities_come_first(self) -> None:
        columns = numeric_columns(table())

        assert columns[0] == "radius_nm"
        assert set(columns) == {"radius_nm", "x_px"}

    def test_text_columns_are_not_offered(self) -> None:
        assert "method" not in numeric_columns(table())


class TestTheHistogram:
    def test_the_bins_come_from_a_named_rule(self) -> None:
        """Not a 20 somebody liked: the shape of a histogram is the claim it
        makes."""
        counts, edges = histogram(table(), "radius_nm")

        expected_counts, expected_edges = np.histogram([10.0, 12.0, 14.0], bins=BIN_RULE)
        assert np.array_equal(counts, expected_counts)
        assert np.allclose(edges, expected_edges)
        assert counts.sum() == 3

    def test_nothing_finite_bins_to_nothing(self) -> None:
        counts, edges = histogram(table(), "nothing_nm")

        assert counts.size == 0 and edges.size == 0


class TestThePanel:
    def test_it_follows_the_run(self, session: SessionViewModel) -> None:
        panel = StatisticsPanel(session)
        session.select_image(image_ids(session)[0])

        measure(session)

        stored = session.measurements()
        assert stored is not None
        assert panel.column.count() == len(numeric_columns(stored))
        assert panel.values["Particles"].text() == str(len(stored))

    def test_an_unscaled_run_loses_the_sizes_and_keeps_the_heights(
        self, session: SessionViewModel
    ) -> None:
        """The finding this panel turned up: a height is calibrated by the **z**
        axis and stays in nanometres without a lateral scale, while a radius
        comes from the pixel size and is absent. "No physical columns" would
        have been wrong about half the table."""
        session.select_image(image_ids(session)[1])

        measure(session)
        panel = StatisticsPanel(session)

        assert "lateral scale is unknown" in panel.note.text()
        assert "height_nm" in _columns(panel)
        assert "radius_nm" not in _columns(panel)

    def test_a_run_that_measured_nothing_says_so(self, session: SessionViewModel) -> None:
        panel = StatisticsPanel(session)
        session.select_image(image_ids(session)[0])

        job = session.detect(PipelineConfig(detector="log", mode="detect"))
        assert isinstance(job, Job)
        assert job.wait(60.0)
        QApplication.processEvents()

        assert "measured nothing" in panel.note.text()
        assert panel.values["Mean"].text() == "—"

    def test_choosing_another_column_redescribes(self, session: SessionViewModel) -> None:
        panel = StatisticsPanel(session)
        session.select_image(image_ids(session)[0])
        measure(session)
        first = panel.values["Mean"].text()

        panel.column.setCurrentIndex(1)

        assert panel.values["Mean"].text() != first

    def test_nothing_selected_shows_nothing(self, session: SessionViewModel) -> None:
        panel = StatisticsPanel(session)

        assert panel.column.count() == 0
        assert panel.values["Mean"].text() == "—"
        assert panel.note.text() == ""


def _columns(panel: StatisticsPanel) -> list[str]:
    return [panel.column.itemText(i) for i in range(panel.column.count())]
