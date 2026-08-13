"""The hand work, on screen (M7-T01, ADR-0070).

M4-T07 made an annotation a row **because it cannot be recomputed**, M4-T08 built
undo around it, M5-T04 counts them before a deletion — and nothing had ever drawn
one. The most expensive data in a project was the only data with no
representation.

What is asserted is the distinction that matters: a box a person **drew** and one
they **accepted from the machine** are not the same claim, and ADR-0044 made that
load-bearing for training.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QGraphicsSimpleTextItem

from nanoscope.app.container import Nanoscope
from nanoscope.core.entities.project import AnnotationSource
from nanoscope.core.values import Modality
from nanoscope.gui.panels import ImageViewer
from nanoscope.gui.panels.viewer import ANNOTATION_STYLES
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")

BOX = (10.0, 12.0, 30.0, 34.0)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name in ("one.npy", "two.npy"):
            source = tmp_path / name
            np.save(source, np.zeros((48, 48), dtype=np.float32))
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


def annotate(
    session: SessionViewModel,
    image_id: int,
    *,
    label: str = "particle",
    source: AnnotationSource = AnnotationSource.MANUAL,
    box: tuple[float, float, float, float] = BOX,
) -> None:
    repository = session._app.repository
    assert repository is not None
    repository.add_annotation(image_id, box, label=label, source=source)


class TestTheyReachTheScreen:
    def test_selecting_an_image_loads_its_annotations(self, session: SessionViewModel) -> None:
        """`annotations_for` has had one caller since M4-T07 — a dialog that
        counts them without ever showing one."""
        annotate(session, image_ids(session)[0])
        viewer = ImageViewer(session)

        session.select_image(image_ids(session)[0])

        assert len(session.annotations) == 1
        assert len(viewer.view.annotation_overlay) == 1
        assert viewer.show_annotations.text() == "Annotations (1)"

    def test_they_belong_to_their_own_image(self, session: SessionViewModel) -> None:
        annotate(session, image_ids(session)[0])
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])

        session.select_image(image_ids(session)[1])

        assert session.annotations == ()
        assert viewer.view.annotation_overlay == []

    def test_the_box_is_where_it_was_drawn(self, session: SessionViewModel) -> None:
        annotate(session, image_ids(session)[0])
        viewer = ImageViewer(session)

        session.select_image(image_ids(session)[0])

        rect = viewer.view.annotation_overlay[0].sceneBoundingRect()
        assert (rect.x(), rect.y()) == pytest.approx((BOX[0], BOX[1]), abs=1.5)
        assert rect.width() == pytest.approx(BOX[2] - BOX[0], abs=1.5)

    def test_closing_the_project_clears_them(self, session: SessionViewModel) -> None:
        annotate(session, image_ids(session)[0])
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])

        session.close_project()

        assert session.annotations == ()
        assert viewer.view.annotation_overlay == []


class TestHandWorkIsNotMachineWork:
    def test_the_two_sources_are_drawn_differently(self, session: SessionViewModel) -> None:
        """A model trained on its own output is confirming itself (ADR-0044), and
        a screen that draws the two alike undoes that where an operator would
        have noticed."""
        annotate(session, image_ids(session)[0], label="mine")
        annotate(
            session,
            image_ids(session)[0],
            label="the machine's",
            source=AnnotationSource.FROM_DETECTION,
            box=(40.0, 40.0, 46.0, 46.0),
        )
        viewer = ImageViewer(session)

        session.select_image(image_ids(session)[0])

        pens = [item.pen() for item in viewer.view.annotation_overlay]
        assert len({pen.color().name() for pen in pens}) == 2
        assert len({pen.style() for pen in pens}) == 2

    def test_every_source_has_a_style(self) -> None:
        """A source with no entry would raise while drawing, which is a crash in
        the layer whose whole job is to be trusted about provenance."""
        assert set(ANNOTATION_STYLES) == set(AnnotationSource)

    def test_annotations_are_drawn_above_the_detections(self, session: SessionViewModel) -> None:
        annotate(session, image_ids(session)[0])
        viewer = ImageViewer(session)

        session.select_image(image_ids(session)[0])

        assert all(item.zValue() > 0 for item in viewer.view.annotation_overlay)


class TestTheLabel:
    def test_it_is_the_operators_own_text(self, session: SessionViewModel) -> None:
        """A box with no label is a rectangle; the label is why it exists."""
        annotate(session, image_ids(session)[0], label="contamination?")
        viewer = ImageViewer(session)

        session.select_image(image_ids(session)[0])

        item = viewer.view.annotation_overlay[0]
        labels = [
            child for child in item.childItems() if isinstance(child, QGraphicsSimpleTextItem)
        ]
        assert [label.text() for label in labels] == ["contamination?"]

    def test_it_does_not_grow_with_the_zoom(self, session: SessionViewModel) -> None:
        """A label that fills the screen at 32x is a label nobody can read."""
        annotate(session, image_ids(session)[0])
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])

        label = viewer.view.annotation_overlay[0].childItems()[0]

        assert label.flags() & label.GraphicsItemFlag.ItemIgnoresTransformations


class TestTheToggle:
    def test_it_empties_the_layer_and_keeps_the_count(self, session: SessionViewModel) -> None:
        annotate(session, image_ids(session)[0])
        viewer = ImageViewer(session)
        session.select_image(image_ids(session)[0])

        viewer.show_annotations.setChecked(False)

        assert viewer.view.annotation_overlay == []
        assert viewer.show_annotations.text() == "Annotations (1)"

    def test_no_annotations_is_a_bare_label(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)

        session.select_image(image_ids(session)[0])

        assert viewer.show_annotations.text() == "Annotations"
