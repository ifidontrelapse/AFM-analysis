"""Preprocessing, asked for from a window (M6-T01, ADR-0061).

The assertion that matters most is the dull one: **a panel nobody touches
produces exactly what `run_preprocessing` produced before the panel existed.**
M6's rule is that the UI introduces no defaults of its own, and the way to keep
it is to compare the two answers array by array.

The rest is about honesty on screen: the viewer names the array it is showing,
and a preview belongs to the scan it was computed from.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from nanoscope.app.container import Nanoscope
from nanoscope.application.jobs import Job
from nanoscope.application.use_cases import preprocess_image, run_preprocessing
from nanoscope.application.use_cases.display import STAGE_LABELS, Stage, stage_image
from nanoscope.application.use_cases.preprocessing import (
    DEFAULT_MIN_SIZE_NM,
    DEFAULT_OPENING_SCALE,
)
from nanoscope.core.errors import UnsupportedRequestError
from nanoscope.core.science.preprocessing import build_substrate_map
from nanoscope.core.values import Modality
from nanoscope.gui.panels import ImageViewer, PreprocessingPanel
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def phantom(size: int = 64) -> np.ndarray:
    """A tilted plane with a few bumps — something levelling has work to do on."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    height = 0.05 * x + 0.02 * y + rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    for cy, cx in ((20, 20), (40, 45), (30, 10)):
        height += 3.0 * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / 18.0))
    return height.astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name, scale in (("monday.npy", 2.0), ("tuesday.npy", 2.0)):
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


def settle(job: Job | None) -> None:
    assert job is not None
    assert job.wait(30.0)
    QApplication.processEvents()


class TestTheDefaultsAreNotTheUIs:
    def test_an_untouched_panel_asks_for_what_the_function_already_did(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """The whole of M6's *"the UI must not introduce its own defaults"*, in
        one comparison: the panel's blank state and the bare call agree array
        for array."""
        panel = PreprocessingPanel(session)
        session.select_image(image_ids(session)[0])

        panel.preview()
        settle(session.job)

        expected = run_preprocessing(
            project / "images" / "monday.npy", fmt="npy", pixel_size_nm=2.0
        )
        preview = session.preview
        assert preview is not None
        assert np.array_equal(preview.z_flat, expected.z_flat)
        assert np.array_equal(preview.substrate, expected.substrate)
        assert np.array_equal(preview.z_result, expected.z_result)
        assert preview.opening_radius == expected.opening_radius

    def test_the_panel_starts_on_the_values_the_application_names(
        self, session: SessionViewModel
    ) -> None:
        panel = PreprocessingPanel(session)

        assert panel.min_size.value() == DEFAULT_MIN_SIZE_NM
        assert panel.opening_scale.value() == DEFAULT_OPENING_SCALE
        assert panel.manual_radius.value() == 0.0  # "estimate it"

    def test_the_named_default_still_matches_the_science(self) -> None:
        """`DEFAULT_MIN_SIZE_NM` mirrors a bare literal in a science signature
        this task did not rewrite. If somebody changes one, this fails rather
        than the two quietly disagreeing."""
        import inspect

        assert inspect.signature(build_substrate_map).parameters[
            "min_size_nm"
        ].default == pytest.approx(DEFAULT_MIN_SIZE_NM)


class TestTheParametersReachTheScience:
    def test_a_manual_radius_is_the_radius(self, session: SessionViewModel) -> None:
        """ADR-0014: when it is given, the estimate is not consulted at all."""
        panel = PreprocessingPanel(session)
        session.select_image(image_ids(session)[0])
        panel.manual_radius.setValue(9.0)

        panel.preview()
        settle(session.job)

        assert session.preview is not None
        assert session.preview.opening_radius == 9

    def test_the_opening_scale_moves_the_radius(self, session: SessionViewModel) -> None:
        """ADR-0037 measured this trade-off; the panel exposes it rather than
        hiding a 2.5 inside a branch."""
        panel = PreprocessingPanel(session)
        session.select_image(image_ids(session)[0])

        panel.opening_scale.setValue(8.0)
        panel.preview()
        settle(session.job)
        wide = session.preview

        panel.opening_scale.setValue(DEFAULT_OPENING_SCALE)
        panel.preview()
        settle(session.job)
        narrow = session.preview

        assert wide is not None and narrow is not None
        assert wide.opening_radius > narrow.opening_radius

    def test_it_reports_what_was_used_not_what_was_asked(self, session: SessionViewModel) -> None:
        """ADR-0014 and ADR-0017 both end on that distinction."""
        panel = PreprocessingPanel(session)
        session.select_image(image_ids(session)[0])

        panel.preview()
        settle(session.job)

        assert "Opening radius used:" in panel.report.text()
        assert "Objects kept by the estimate:" in panel.report.text()


class TestItRunsAsAJob:
    def test_the_preview_is_asked_for_and_runs_in_the_background(
        self, session: SessionViewModel
    ) -> None:
        """Architecture §4.5: anything over ~100 ms is a job. Nothing runs until
        the button is pressed — a pipeline that re-runs on every keystroke is a
        UI that fights the operator (ADR-0061)."""
        panel = PreprocessingPanel(session)
        session.select_image(image_ids(session)[0])

        assert session.preview is None
        assert session.job is None

        panel.preview()

        assert session.job is not None
        settle(session.job)
        assert session.preview is not None

    def test_it_needs_a_selected_image(self, session: SessionViewModel) -> None:
        panel = PreprocessingPanel(session)

        assert not panel.run.isEnabled()
        assert session.preprocess() is None

    def test_a_failure_is_a_message_and_not_a_preview(
        self, session: SessionViewModel, project: Path
    ) -> None:
        said: list[str] = []
        session.failed.connect(said.append)
        session.select_image(image_ids(session)[0])
        (project / "images" / "monday.npy").unlink()

        settle(session.preprocess())

        assert session.preview is None
        assert said

    def test_it_is_written_down_as_well_as_shown(
        self, session: SessionViewModel, caplog: pytest.LogCaptureFixture
    ) -> None:
        session.select_image(image_ids(session)[0])

        with caplog.at_level(logging.INFO, logger="nanoscope.gui.viewmodels.session"):
            settle(session.preprocess())

        assert any("opening radius" in record.message for record in caplog.records)


class TestTheViewerSaysWhatItIsShowing:
    def test_it_names_the_stage(self, session: SessionViewModel) -> None:
        """ADR-0056's rule was never *show the file and nothing else* — it was
        never show something the file does not contain **without saying so**."""
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])
        assert viewer.stage_label.text() == "raw"

        settle(session.preprocess())

        assert session.stage is Stage.RESULT
        assert viewer.stage_label.text() == "result"
        #: The long form is the tooltip, because the short one has to fit
        #: beside the colormap without being clipped.
        assert viewer.stage_label.toolTip() == STAGE_LABELS[Stage.RESULT]

    def test_choosing_a_stage_redraws(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])
        settle(session.preprocess())

        session.show_stage(Stage.SUBSTRATE)

        assert viewer.stage_label.text() == "substrate"
        assert not viewer.view._item.pixmap().isNull()

    def test_a_stage_with_no_preview_behind_it_is_refused(self, session: SessionViewModel) -> None:
        """A viewer switched to an array nobody computed would draw the last one
        under a new name, which is the one thing ADR-0056 forbids."""
        session.select_image(image_ids(session)[0])

        session.show_stage(Stage.SUBSTRATE)

        assert session.stage is Stage.RAW

    def test_stage_image_falls_back_to_the_file(self, session: SessionViewModel) -> None:
        session.select_image(image_ids(session)[0])
        image = session.image
        assert image is not None

        assert stage_image(Stage.SUBSTRATE, image, None) is image


class TestAPreviewBelongsToItsScan:
    def test_selecting_another_image_drops_it(self, session: SessionViewModel) -> None:
        """A substrate map from another scan drawn over this one would be the
        worst possible version of this feature."""
        session.select_image(image_ids(session)[0])
        settle(session.preprocess())
        assert session.preview is not None

        session.select_image(image_ids(session)[1])

        assert session.preview is None
        assert session.stage is Stage.RAW

    def test_closing_the_project_drops_it(self, session: SessionViewModel) -> None:
        session.select_image(image_ids(session)[0])
        settle(session.preprocess())

        session.close_project()

        assert session.preview is None

    def test_nothing_is_written_to_the_project(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """A preview is a look at intermediate arrays; a *run* is what
        `run_analysis` records (ADR-0042, ADR-0061)."""
        session.select_image(image_ids(session)[0])
        settle(session.preprocess())

        assert not list((project / "results").iterdir())


class TestTheUseCase:
    def test_it_resolves_the_row_the_way_run_analysis_does(self, session: SessionViewModel) -> None:
        """The scale the project recorded is the scale the preview uses — the
        assembly M4-T05 found the D-07 family of defect hiding in."""
        repository = session._app.repository
        assert repository is not None

        result = preprocess_image(repository, image_ids(session)[0])

        assert result.pixel_size_nm == 2.0

    def test_a_non_afm_row_is_refused_with_a_reason(
        self, session: SessionViewModel, tmp_path: Path
    ) -> None:
        """SEM and TEM have no substrate to build; they are analysed as they
        are (ADR-0031)."""
        repository = session._app.repository
        assert repository is not None
        source = tmp_path / "sem.png"
        source.write_bytes(b"not really a png")
        record = repository.import_image(source, modality=Modality.SEM)

        with pytest.raises(UnsupportedRequestError, match="analysed as it is"):
            preprocess_image(repository, record.id)
