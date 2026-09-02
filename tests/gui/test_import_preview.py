"""A look at the file before it is imported (2026-09-02).

The operator's request, in their words: a preview when an image is picked in the
file browser. What it is worth is measurable in the file names — a Bruker writes
`2-6-dmfa-pvp.039`, and the number is the acquisition, not the sample.

The tests drive `show_preview` directly rather than opening a modal dialog: what
is worth asserting is *what the pane says* about a file, and a modal exec in a
headless suite asserts only that Qt can be blocked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog

from nanoscope.gui.dialogs import ImageChooser
from nanoscope.gui.dialogs.choose_images import NOTHING_SELECTED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic_spm import LINES, SAMPS, write_spm, z_field

pytestmark = pytest.mark.usefixtures("qt_app")


@pytest.fixture
def chooser() -> ImageChooser:
    return ImageChooser()


class TestThePaneExistsAtAll:
    def test_the_dialog_is_qts_own_because_a_native_one_has_nowhere_to_put_it(
        self, chooser: ImageChooser
    ) -> None:
        """`DontUseNativeDialog` is the precondition, not a preference: the pane
        is a child of Qt's grid layout and a native dialog has none."""
        assert chooser.testOption(QFileDialog.Option.DontUseNativeDialog)

    def test_the_pane_is_in_the_layout_beside_the_file_list(
        self, chooser: ImageChooser
    ) -> None:
        layout = chooser.layout()
        assert layout is not None
        assert chooser.picture.parentWidget() is not None
        assert layout.indexOf(chooser.picture.parentWidget()) != -1

    def test_it_says_what_it_is_waiting_for_before_anything_is_picked(
        self, chooser: ImageChooser
    ) -> None:
        assert chooser.facts.text() == NOTHING_SELECTED
        assert chooser.picture.pixmap().isNull()


class TestWhatItShows:
    def test_a_nanoscope_file_is_drawn_with_the_scale_its_header_states(
        self, chooser: ImageChooser, tmp_path: Path
    ) -> None:
        """The picture *and* the answer to the question the next dialog raises:
        this file states its own scale, so the pixel-size field is not for it
        (ADR-0083)."""
        path = write_spm(tmp_path, z_field(), name="2-6-dmfa-pvp.039")

        chooser.show_preview(str(path))

        assert not chooser.picture.pixmap().isNull()
        assert f"{SAMPS} x {LINES} px" in chooser.facts.text()
        assert "750 nm/px, from the file's header" in chooser.facts.text()
        assert "3000 x 4500 nm" in chooser.facts.text()

    def test_an_npy_is_drawn_and_says_it_carries_no_scale(
        self, chooser: ImageChooser, tmp_path: Path
    ) -> None:
        """ADR-0025 one surface earlier than usual: before a row exists, a
        preview that invented a scale would be inventing it for the import."""
        path = tmp_path / "flat.npy"
        np.save(path, np.linspace(0, 1, 64).reshape(8, 8).astype(np.float32))

        chooser.show_preview(str(path))

        assert not chooser.picture.pixmap().isNull()
        assert "no scale in the file" in chooser.facts.text()
        assert "nm/px" not in chooser.facts.text()

    def test_a_file_with_no_reader_says_why_instead_of_showing_nothing(
        self, chooser: ImageChooser, tmp_path: Path
    ) -> None:
        """Every refusal here is a sentence somebody wrote (ADR-0030), and the
        pane is where it costs nothing to read one."""
        path = tmp_path / "notes.txt"
        path.write_text("nothing to see")

        chooser.show_preview(str(path))

        assert chooser.picture.pixmap().isNull()
        assert "no preview:" in chooser.facts.text()
        assert "notes.txt" in chooser.facts.text()

    def test_a_truncated_nanoscope_file_says_what_the_parser_said(
        self, chooser: ImageChooser, tmp_path: Path
    ) -> None:
        """The case the preview is worth the most: found here, in a pane,
        instead of after importing forty files."""
        path = tmp_path / "half-a-scan.000"
        path.write_bytes(b"\x1a" * 64)

        chooser.show_preview(str(path))

        assert chooser.picture.pixmap().isNull()
        assert "Ciao image list" in chooser.facts.text()

    def test_a_directory_clears_the_pane_rather_than_refusing_it(
        self, chooser: ImageChooser, tmp_path: Path
    ) -> None:
        """Walking through folders is how a file dialog is used; every step of
        it is not an error."""
        chooser.show_preview(str(write_spm(tmp_path, z_field())))
        chooser.show_preview(str(tmp_path))

        assert chooser.picture.pixmap().isNull()
        assert chooser.facts.text() == NOTHING_SELECTED

    def test_nothing_highlighted_clears_it_too(
        self, chooser: ImageChooser, tmp_path: Path
    ) -> None:
        chooser.show_preview(str(write_spm(tmp_path, z_field())))
        chooser.show_preview("")

        assert chooser.facts.text() == NOTHING_SELECTED


class TestWhatItCosts:
    def test_a_large_scan_is_subsampled_before_it_is_coloured(
        self, chooser: ImageChooser, tmp_path: Path
    ) -> None:
        """A preview is a look, not a measurement — so it is rendered from a
        stride, and the pane never holds more pixels than it can show."""
        path = tmp_path / "big.npy"
        np.save(path, np.zeros((2048, 2048), dtype=np.float32))

        chooser.show_preview(str(path))

        pixmap = chooser.picture.pixmap()
        assert not pixmap.isNull()
        assert max(pixmap.width(), pixmap.height()) <= chooser.picture.width()
