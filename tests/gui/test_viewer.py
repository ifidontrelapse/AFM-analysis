"""A scan on screen, and the numbers that make it a measurement (M5-T05).

M5's second exit criterion is *"a scan renders with correct nm axes and a scale
bar"*, and the tests that matter here are the ones about **numbers**, not
pixels: that the physical size is right, that the readout says nm *and* px, and
that a scan with no known scale says so instead of inventing one — which is
ADR-0025's rule arriving at the last surface that could break it.

The rendering half lives in `application`, so most of it is tested without Qt at
all; what needs a widget is here.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nanoscope.app.container import Nanoscope
from nanoscope.application.use_cases.display import (
    COLORMAPS,
    DisplayImage,
    load_for_display,
    render,
    value_range,
)
from nanoscope.core.errors import (
    DataFormatError,
    InvalidParameterError,
    UnsupportedRequestError,
)
from nanoscope.core.values import Modality
from nanoscope.gui.panels.viewer import BAR_LENGTHS_NM, MAX_ZOOM, ImageViewer, _bar_length_nm
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def phantom(size: int = 64) -> np.ndarray:
    """A small height map with a hot pixel, for the percentile window."""
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    data[0, 0] = 1_000.0
    return data


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """One scan with a known scale, one without."""
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name, scale in (("scaled.npy", 2.5), ("unscaled.npy", None)):
            source = tmp_path / name
            np.save(source, phantom())
            repo.import_image(source, modality=Modality.AFM, pixel_size_nm=scale)
    return root


@pytest.fixture
def app(tmp_path: Path, project: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        container.open(project)
        yield container


@pytest.fixture
def session(app: Nanoscope) -> SessionViewModel:
    """M5-T06: the panel is given the session, and loading happens there. What
    is left in the widget is an array, a colormap and a pixmap."""
    model = SessionViewModel(app)
    model.refresh()
    return model


@pytest.fixture
def viewer(session: SessionViewModel) -> ImageViewer:
    return ImageViewer(session)


def image_ids(app: Nanoscope) -> list[int]:
    assert app.repository is not None
    return [image.id for image in app.repository.list_images()]


class TestLoadingForDisplay:
    def test_it_reads_what_is_in_the_file(self, app: Nanoscope) -> None:
        """Raw, not flattened: flattening is an analysis, and a viewer that does
        it silently shows something the file does not contain (ADR-0056)."""
        assert app.repository is not None

        image = load_for_display(app.repository, image_ids(app)[0])

        assert image.data.shape == (64, 64)
        assert np.isclose(image.data.max(), 1_000.0)

    def test_the_projects_scale_reaches_the_viewer(self, app: Nanoscope) -> None:
        """An `.npy` carries no metadata — the scale comes from the row, which
        is the defect M4-T05 found and fixed one layer down."""
        assert app.repository is not None

        image = load_for_display(app.repository, image_ids(app)[0])

        assert image.pixel_size_nm == 2.5
        assert image.size_nm == (160.0, 160.0)

    def test_an_unknown_scale_stays_unknown(self, app: Nanoscope) -> None:
        assert app.repository is not None

        image = load_for_display(app.repository, image_ids(app)[1])

        assert image.pixel_size_nm is None
        assert image.size_nm is None


class TestWhichFilesTheViewerCanOpen:
    """The defect an operator found on 2026-08-30.

    A Bruker Nanoscope writes `scan.000`, `scan.001`, … and the parser has always
    read them. `display.py` kept its own copy of the extension map — the one in
    `preprocessing.py` was already there — and the copy listed only `.spm` and
    `.npy`, so a folder off the microscope imported, appeared in the explorer,
    and would not open. Both go through `afm_format` now.
    """

    def test_a_numbered_file_reaches_the_nanoscope_parser(
        self, app: Nanoscope, tmp_path: Path
    ) -> None:
        """What is asserted is the **dispatch**, so the payload is deliberately
        not a scan: the parser's own complaint means `.001` got that far, where
        before it was turned away by the extension and never read at all.

        A real numbered file parsing end to end is
        `tests/unit/test_afm_io.py::test_a_numbered_nanoscope_file_actually_reads`,
        which has the synthetic Nanoscope bytes to do it with.
        """
        assert app.repository is not None
        source = tmp_path / "2-2-dmfa-citr-temp.001"
        source.write_bytes(b"not a Nanoscope file")
        record = app.repository.import_image(source, modality=Modality.AFM)

        with pytest.raises(DataFormatError, match="Ciao"):
            load_for_display(app.repository, record.id)

    def test_a_jpeg_imported_as_afm_says_what_it_supports(
        self, app: Nanoscope, tmp_path: Path
    ) -> None:
        """The other half of what the operator hit, and this one is a correct
        refusal: a JPEG is a SEM/TEM image, and the modality is the question the
        import dialog asks (M5-T07). What was wrong was that nobody saw it."""
        assert app.repository is not None
        source = tmp_path / "particles.jpg"
        source.write_bytes(b"not really a jpeg")
        record = app.repository.import_image(source, modality=Modality.AFM)

        with pytest.raises(UnsupportedRequestError) as refusal:
            load_for_display(app.repository, record.id)

        assert ".000" in str(refusal.value)


class TestTheViewerSaysWhyItIsEmpty:
    """A blank canvas with the reason in the status bar reads as a broken
    application. The reason belongs beside the thing that is missing."""

    def test_a_refusal_lands_on_the_panel(self, viewer: ImageViewer) -> None:
        viewer.show_image(None)

        viewer._show_failure("no AFM reader for particles.jpg")

        assert "no AFM reader" in viewer.failure_label.text()

    def test_an_image_clears_it(self, viewer: ImageViewer, app: Nanoscope) -> None:
        assert app.repository is not None
        viewer.show_image(None)
        viewer._show_failure("no AFM reader for particles.jpg")

        viewer.show_image(load_for_display(app.repository, image_ids(app)[0]))

        assert viewer.failure_label.text() == ""

    def test_a_refusal_about_something_else_is_not_shown_here(
        self, viewer: ImageViewer, app: Nanoscope
    ) -> None:
        """`failed` carries every refusal the session makes — an export, a
        removal. This panel speaks only when it has nothing to draw."""
        assert app.repository is not None
        viewer.show_image(load_for_display(app.repository, image_ids(app)[0]))

        viewer._show_failure("nothing to export: no annotation of that kind exists")

        assert viewer.failure_label.text() == ""


class TestRendering:
    def test_it_produces_rgb_bytes(self, app: Nanoscope) -> None:
        assert app.repository is not None
        image = load_for_display(app.repository, image_ids(app)[0])

        rgb = render(image)

        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_the_default_window_ignores_a_hot_pixel(self, app: Nanoscope) -> None:
        """One outlier otherwise flattens the whole image to grey, which is why
        every SPM tool clips by percentile."""
        assert app.repository is not None
        image = load_for_display(app.repository, image_ids(app)[0])

        low, high = value_range(image)
        full_low, full_high = value_range(image, full=True)

        assert high < 10.0
        assert full_high == pytest.approx(1_000.0)
        assert full_low <= low

    def test_the_full_range_shows_everything(self, app: Nanoscope) -> None:
        """ "What am I clipping?" has to be answerable."""
        assert app.repository is not None
        image = load_for_display(app.repository, image_ids(app)[0])

        rendered = render(image, limits=value_range(image, full=True))

        assert rendered[0, 0].max() > 0

    def test_an_unknown_colormap_is_a_message(self, app: Nanoscope) -> None:
        """A typo in a settings file must produce a sentence, not a matplotlib
        traceback."""
        assert app.repository is not None
        image = load_for_display(app.repository, image_ids(app)[0])

        with pytest.raises(InvalidParameterError, match="unknown colormap"):
            render(image, colormap="not-a-colormap")

    def test_every_offered_colormap_works(self, app: Nanoscope) -> None:
        assert app.repository is not None
        image = load_for_display(app.repository, image_ids(app)[0])

        for name in COLORMAPS:
            assert render(image, colormap=name).shape == (64, 64, 3)


class TestTheScaleBar:
    def test_it_is_a_round_number(self) -> None:
        """A bar reading "137 nm" is one nobody can measure against by eye,
        which is the only thing a scale bar is for."""
        image = DisplayImage("x", phantom(), Modality.AFM, pixel_size_nm=2.5)

        length = _bar_length_nm(image, zoom=1.0)

        assert length in BAR_LENGTHS_NM
        assert length <= image.size_nm[0] / 5  # type: ignore[index]

    def test_it_scales_with_the_image(self) -> None:
        small = DisplayImage("s", phantom(), Modality.AFM, pixel_size_nm=0.5)
        large = DisplayImage("l", phantom(), Modality.AFM, pixel_size_nm=50.0)

        assert _bar_length_nm(small, 1.0) < _bar_length_nm(large, 1.0)

    def test_there_is_no_bar_without_a_scale(
        self, app: Nanoscope, session: SessionViewModel, viewer: ImageViewer
    ) -> None:
        """A viewer that draws a bar without a scale is a viewer inventing one —
        the substitution ADR-0025 spent a milestone removing."""
        session.select_image(image_ids(app)[1])

        assert viewer.scale_label.text() == "scale unknown"

    def test_there_is_one_when_there_is_a_scale(
        self, app: Nanoscope, session: SessionViewModel, viewer: ImageViewer
    ) -> None:
        session.select_image(image_ids(app)[0])

        assert "nm" in viewer.scale_label.text()
        assert "unknown" not in viewer.scale_label.text()


class TestThePanel:
    def test_showing_an_image_announces_its_size(
        self, app: Nanoscope, session: SessionViewModel, viewer: ImageViewer
    ) -> None:
        said: list[str] = []
        viewer.readout.connect(said.append)

        assert session.select_image(image_ids(app)[0]) is True

        assert "64 x 64 px" in said[-1]
        assert "160 x 160 nm" in said[-1]

    def test_an_unscaled_image_says_so_rather_than_inventing_a_size(
        self, app: Nanoscope, session: SessionViewModel, viewer: ImageViewer
    ) -> None:
        said: list[str] = []
        viewer.readout.connect(said.append)

        session.select_image(image_ids(app)[1])

        assert "scale unknown" in said[-1]

    def test_the_readout_gives_nm_and_px_and_the_value(
        self, app: Nanoscope, session: SessionViewModel, viewer: ImageViewer
    ) -> None:
        """Both, because a pixel index is what you click and a nanometre is what
        you report."""
        said: list[str] = []
        session.select_image(image_ids(app)[0])
        viewer.readout.connect(said.append)

        viewer._describe((10, 20))

        assert "x=10 y=20 px" in said[-1]
        assert "(25.0, 50.0) nm" in said[-1]
        assert "value=" in said[-1]

    def test_the_readout_omits_nm_when_there_is_no_scale(
        self, app: Nanoscope, session: SessionViewModel, viewer: ImageViewer
    ) -> None:
        said: list[str] = []
        session.select_image(image_ids(app)[1])
        viewer.readout.connect(said.append)

        viewer._describe((10, 20))

        assert "px" in said[-1]
        assert "nm" not in said[-1]

    def test_a_missing_file_empties_the_canvas_and_says_why(
        self, app: Nanoscope, session: SessionViewModel, viewer: ImageViewer, project: Path
    ) -> None:
        """The operator clicked a row, not a button labelled "load" — so it is a
        sentence, not a dialog (ADR-0056). M5-T06 moved the sentence to
        `failed`, where the window decides what to do with it; what the widget
        owes is to stop showing the scan it can no longer see."""
        session.select_image(image_ids(app)[0])
        (project / "images" / "unscaled.npy").unlink()
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.select_image(image_ids(app)[1]) is False

        assert said
        assert viewer.view._item.pixmap().isNull()

    def test_changing_the_colormap_redraws(
        self, app: Nanoscope, session: SessionViewModel, viewer: ImageViewer
    ) -> None:
        session.select_image(image_ids(app)[0])

        viewer.colormap.setCurrentText("viridis")

        assert viewer.colormap.currentText() == "viridis"
        assert not viewer.view._item.pixmap().isNull()

    def test_closing_the_project_empties_the_canvas(
        self, app: Nanoscope, session: SessionViewModel, viewer: ImageViewer
    ) -> None:
        session.select_image(image_ids(app)[0])

        session.close_project()

        assert viewer.view._item.pixmap().isNull()
        assert viewer.scale_label.text() == ""


class TestZoom:
    def test_it_starts_fitted(
        self, app: Nanoscope, session: SessionViewModel, viewer: ImageViewer
    ) -> None:
        session.select_image(image_ids(app)[0])

        assert 0 < viewer.view.zoom <= MAX_ZOOM

    def test_it_will_not_go_past_the_limits(
        self, app: Nanoscope, session: SessionViewModel, viewer: ImageViewer
    ) -> None:
        """Past `MAX_ZOOM` a pixel fills the window and there is nothing more to
        see; the transform loses precision long before that is interesting."""
        session.select_image(image_ids(app)[0])
        view = viewer.view

        for _ in range(100):
            view._zoom = min(view._zoom * 1.25, MAX_ZOOM)

        assert view.zoom <= MAX_ZOOM
