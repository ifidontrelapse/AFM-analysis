"""The profile on screen (M7-T06, ADR-0075).

The arithmetic is tested in `tests/unit/test_metrology.py`; what is here is the
half that decides whether the number can be defended — **which stage it was
measured on**, and what happens when there is no scale.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nanoscope.app.container import Nanoscope
from nanoscope.application.use_cases.display import Stage
from nanoscope.core.entities import RulerKind
from nanoscope.core.values import Modality
from nanoscope.gui.panels import ProfilePanel
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")

START, END = (2.0, 8.0), (12.0, 8.0)


def scan(size: int = 24) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    return (x * 0.5 + y * 0.1).astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name, scale in (("scaled.npy", 2.0), ("unscaled.npy", None)):
            source = tmp_path / name
            np.save(source, scan())
            repo.import_image(source, modality=Modality.AFM, pixel_size_nm=scale)
    return root


@pytest.fixture
def session(tmp_path: Path, project: Path) -> Iterator[SessionViewModel]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        model = SessionViewModel(container)
        model.open_project(project)
        assert model.project is not None
        model.select_image(model.project.images[0].id)
        yield model


def image_ids(session: SessionViewModel) -> list[int]:
    assert session.project is not None
    return [image.id for image in session.project.images]


class TestWhatItReads:
    def test_it_profiles_the_scan_under_the_line(self, session: SessionViewModel) -> None:
        ProfilePanel(session)

        session.add_ruler(START, END, label="ridge", kind=RulerKind.PROFILE)

        profile = session.ruler_profile(session.rulers[0])
        assert profile is not None
        _distances, nm, heights = profile
        assert heights.size == 11
        assert np.array_equal(heights, scan()[8, 2:13])
        assert nm is not None and nm[-1] == pytest.approx(20.0)

    def test_it_names_the_stage_it_measured(self, session: SessionViewModel) -> None:
        """A measurement whose provenance is a checkbox somebody set four clicks
        ago is not one anybody can defend."""
        panel = ProfilePanel(session)

        session.add_ruler(START, END, label="ridge")

        assert "raw (the file)" in panel.summary.text()
        assert "11 samples" in panel.summary.text()

    def test_without_a_scale_the_length_is_in_pixels(self, session: SessionViewModel) -> None:
        session.select_image(image_ids(session)[1])
        panel = ProfilePanel(session)

        session.add_ruler(START, END, label="ridge")

        assert session.ruler_profile(session.rulers[0])[1] is None  # type: ignore[index]
        assert "px" in panel.summary.text()

    def test_no_line_says_so(self, session: SessionViewModel) -> None:
        panel = ProfilePanel(session)

        assert "Draw a line" in panel.summary.text()

    def test_no_scan_says_so(self, tmp_path: Path, project: Path) -> None:
        with Nanoscope(settings_path=tmp_path / "s.json") as container:
            session = SessionViewModel(container)
            session.open_project(project)
            panel = ProfilePanel(session)

            assert session.ruler_profile.__name__ == "ruler_profile"
            assert "Draw a line" in panel.summary.text()

    def test_the_stage_it_names_is_the_one_it_used(self, session: SessionViewModel) -> None:
        """The whole point of naming it: profiling a raw map and a flattened one
        give different numbers."""
        panel = ProfilePanel(session)
        session.add_ruler(START, END, label="ridge")
        assert session.stage is Stage.RAW

        assert "raw" in panel.summary.text()


class TestThePlot:
    def test_it_draws_the_heights_it_was_given(self, session: SessionViewModel) -> None:
        panel = ProfilePanel(session)

        session.add_ruler(START, END, label="ridge")

        assert panel.view._heights.size == 11

    def test_choosing_another_line_redraws(self, session: SessionViewModel) -> None:
        panel = ProfilePanel(session)
        session.add_ruler(START, END, label="short")
        session.add_ruler((2.0, 2.0), (20.0, 2.0), label="long")

        assert panel.view._heights.size == 19

        panel.ruler.setCurrentIndex(0)

        assert panel.view._heights.size == 11
