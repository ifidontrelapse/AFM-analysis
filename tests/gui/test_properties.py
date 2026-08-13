"""What the panel says about a scan (M5-T06).

The wording is the deliverable, so the wording is what is asserted — and the one
field that could undo a milestone of work is the physical size: a scan with no
scale has none, and says so (ADR-0025, the third surface in M5 to be asked).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nanoscope.app.container import Nanoscope
from nanoscope.application.use_cases.display import DisplayImage
from nanoscope.core.values import Modality
from nanoscope.gui.panels.properties import ABSENT, FIELDS, PropertiesPanel, _describe
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def phantom() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, (32, 64)).astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name, scale in (("scaled.npy", 2.5), ("unscaled.npy", None)):
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


class TestWhatItSays:
    def test_it_describes_the_loaded_image(self) -> None:
        described = _describe(DisplayImage("scan.npy", phantom(), Modality.AFM, 2.5))

        assert described["Name"] == "scan.npy"
        assert described["Modality"] == "AFM"
        assert described["Size"] == "64 x 32 px"
        assert described["Physical size"] == "160.0 x 80.0 nm"
        assert described["Pixel size"] == "2.5 nm/px"
        assert described["Data type"] == "float32"

    def test_an_unknown_scale_has_no_physical_size(self) -> None:
        """The invention ADR-0025 spent a milestone removing, refused once more."""
        described = _describe(DisplayImage("scan.npy", phantom(), Modality.AFM, None))

        assert described["Physical size"] == "scale unknown"
        assert described["Pixel size"] == "unknown"

    def test_nothing_selected_is_every_field_absent(self) -> None:
        assert _describe(None) == dict.fromkeys(FIELDS, ABSENT)

    def test_the_value_range_is_the_whole_array(self) -> None:
        """Not the display window: this panel answers "what is in the file?",
        which is the question the viewer's percentile clip cannot."""
        data = phantom()
        data[0, 0] = 1_000.0

        described = _describe(DisplayImage("scan.npy", data, Modality.AFM, 2.5))

        assert "1000" in described["Value range"]


class TestThePanel:
    def test_it_fills_when_the_session_loads_an_image(self, session: SessionViewModel) -> None:
        """The second consumer of one load — the whole justification for the
        viewmodel (ADR-0057)."""
        panel = PropertiesPanel(session)

        session.select_image(image_ids(session)[0])

        assert panel.values["Name"].text() == "scaled.npy"
        assert panel.values["Physical size"].text() == "160.0 x 80.0 nm"

    def test_it_empties_when_the_project_closes(self, session: SessionViewModel) -> None:
        panel = PropertiesPanel(session)
        session.select_image(image_ids(session)[0])

        session.close_project()

        assert panel.values["Name"].text() == ABSENT

    def test_it_starts_from_whatever_the_session_already_holds(
        self, session: SessionViewModel
    ) -> None:
        """A panel built after a selection shows it — a dock restored from a
        saved layout must not be blank until the next click."""
        session.select_image(image_ids(session)[0])

        panel = PropertiesPanel(session)

        assert panel.values["Name"].text() == "scaled.npy"

    def test_it_names_the_task_that_fills_the_rest(self, session: SessionViewModel) -> None:
        """An empty section is a promise when it names its task and a bug when
        it does not (M5-T02's rule, inside a panel)."""
        panel = PropertiesPanel(session)

        assert any("M6" in label.text() for label in panel.findChildren(type(panel.values["Name"])))
