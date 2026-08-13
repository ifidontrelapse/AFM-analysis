"""The export, asked for from a window (M6-T07, ADR-0067).

M6's first exit criterion ends here: *load → detect → segment → measure →
**export CSV**, entirely through the UI*. `export_measurements` has been in
`application` since M4-T11 with tests as its only callers.

Two scopes, and they are the point: *this run* and *every run in the project*.
ADR-0048 built the second deliberately — statistics across a dataset is why the
measurements exist — and a menu item that silently meant one of them is one
somebody uses wrong once.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from nanoscope.app.container import Nanoscope
from nanoscope.application.jobs import Job
from nanoscope.core.entities import PipelineConfig
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def phantom(size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(6)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    height = rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    for cy, cx in ((14, 14), (32, 30)):
        height += 5.0 * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / 10.0))
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
def app(tmp_path: Path, project: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        yield container


@pytest.fixture
def session(app: Nanoscope, project: Path) -> SessionViewModel:
    model = SessionViewModel(app)
    model.open_project(project)
    return model


def image_ids(session: SessionViewModel) -> list[int]:
    assert session.project is not None
    return [image.id for image in session.project.images]


def settle(job: Job | None) -> None:
    assert job is not None
    assert job.wait(60.0)
    QApplication.processEvents()


def measure(session: SessionViewModel, image_id: int) -> None:
    session.select_image(image_id)
    settle(session.detect(PipelineConfig(detector="log", mode="baseline")))


def exports(project: Path) -> list[Path]:
    return sorted((project / "exports").glob("*.csv"))


class TestWhatEachScopeMeans:
    def test_this_run_exports_this_run(self, session: SessionViewModel, project: Path) -> None:
        measure(session, image_ids(session)[0])
        measure(session, image_ids(session)[1])

        settle(session.export(everything=False))

        written = exports(project)
        assert len(written) == 1
        rows = list(csv.DictReader(written[0].read_text().splitlines()))
        assert rows
        assert {row["image"] for row in rows} == {"two.npy"}

    def test_everything_exports_every_run(self, session: SessionViewModel, project: Path) -> None:
        """ADR-0048's reason for taking a collection: statistics across a
        dataset is why the measurements exist."""
        measure(session, image_ids(session)[0])
        measure(session, image_ids(session)[1])

        settle(session.export(everything=True))

        rows = list(csv.DictReader(exports(project)[0].read_text().splitlines()))
        assert {row["image"] for row in rows} == {"one.npy", "two.npy"}

    def test_the_file_lands_in_the_projects_exports(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """An export is part of the project (ADR-0003's layout), timestamped so
        today's does not replace yesterday's (ADR-0048)."""
        said: list[str] = []
        session.reported.connect(said.append)
        measure(session, image_ids(session)[0])

        settle(session.export(everything=True))

        assert said[-1].startswith("Exported to exports/")
        assert (project / said[-1].removeprefix("Exported to ")).is_file()

    def test_exporting_needs_something_to_export(self, session: SessionViewModel) -> None:
        assert session.export(everything=False) is None


class TestWhenThereIsNothing:
    def test_a_detect_only_run_is_refused_in_the_use_cases_own_words(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """A file with headers and no rows says "we measured and found nothing",
        which is a different statement (ADR-0048). The window shows that
        sentence rather than pre-empting it with a disabled button."""
        said: list[str] = []
        session.failed.connect(said.append)
        session.select_image(image_ids(session)[0])
        settle(session.detect(PipelineConfig(detector="log", mode="detect")))

        settle(session.export(everything=False))

        assert "nothing to export" in said[-1]
        assert exports(project) == []


class TestTheWindow:
    def test_both_scopes_are_offered_and_enabled_when_they_apply(
        self, app: Nanoscope, project: Path
    ) -> None:
        window = MainWindow(app)
        window.open_project(project)

        assert window.export_all_action.isEnabled()
        assert not window.export_run_action.isEnabled(), "no run is on screen yet"

        measure(window.session, image_ids(window.session)[0])

        assert window.export_run_action.isEnabled()

    def test_the_status_bar_says_where_the_file_went(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)
        measure(window.session, image_ids(window.session)[0])

        window.export(everything=True)
        settle(window.session.job)

        assert "Exported to exports/" in window.statusBar().currentMessage()
