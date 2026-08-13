"""Correcting the machine without rewriting what it did (M7-T07, ADR-0076).

The task list says *"manual add/edit/delete of detections"*, and the whole of
this file is about the reading that makes it honest: **a detection is not
edited.** It is what a detector produced in a run, and an operator deleting one
makes the run describe an analysis that never happened.

What an operator edits is an **annotation** — and an annotation adopted from a
detector carries `from_detection` for ever, because *a model trained on its own
output is confirming itself* (ADR-0044).
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
from nanoscope.core.entities.project import AnnotationSource
from nanoscope.core.values import Modality
from nanoscope.gui.panels import AnnotatePanel, ImageViewer
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def phantom(size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(11)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    height = rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    for cy, cx in ((14, 14), (32, 30)):
        height += 5.0 * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / 10.0))
    return height.astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        source = tmp_path / "one.npy"
        np.save(source, phantom())
        repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
    return root


@pytest.fixture
def session(tmp_path: Path, project: Path) -> Iterator[SessionViewModel]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        model = SessionViewModel(container)
        model.open_project(project)
        assert model.project is not None
        model.select_image(model.project.images[0].id)
        yield model


def analyse(session: SessionViewModel) -> None:
    job = session.detect(PipelineConfig(detector="log", mode="detect"))
    assert isinstance(job, Job)
    assert job.wait(60.0)
    QApplication.processEvents()


class TestAdoption:
    def test_an_adopted_detection_says_where_it_came_from(self, session: SessionViewModel) -> None:
        """`AnnotationSource.FROM_DETECTION` has existed since M4-T07 with
        nothing that writes it. This is its first writer."""
        analyse(session)
        panel = AnnotatePanel(session)

        assert session.adopt_detection(0, label="particle") is True

        adopted = session.annotations[0]
        assert adopted.source is AnnotationSource.FROM_DETECTION
        assert adopted.label == "particle"
        assert panel is not None

    def test_the_box_is_the_detections_own(self, session: SessionViewModel) -> None:
        analyse(session)
        run = session.run
        assert run is not None
        detection = run.detections[0]

        session.adopt_detection(0, label="particle")

        x1, y1, x2, y2 = session.annotations[0].box
        assert x1 < detection.x_px < x2
        assert y1 < detection.y_px < y2

    def test_the_detection_itself_is_untouched(self, session: SessionViewModel) -> None:
        """A run is a record of what happened; editing it would make the project
        describe an analysis that never ran (ADR-0042, ADR-0076)."""
        analyse(session)
        run = session.run
        assert run is not None
        before = list(run.detections)

        session.adopt_detection(0, label="particle")
        session.remove_annotation(session.annotations[0].id)

        repository = session._app.repository
        assert repository is not None
        assert list(repository.get_run(run.id).detections) == before

    def test_adopting_everything_is_one_click(self, session: SessionViewModel) -> None:
        """Reviewing forty detections means keeping thirty-eight and fixing
        two."""
        analyse(session)
        run = session.run
        assert run is not None

        adopted = session.adopt_all_detections(label="particle")

        assert adopted == len(run.detections)
        assert len(session.annotations) == len(run.detections)
        assert all(one.source is AnnotationSource.FROM_DETECTION for one in session.annotations)

    def test_adoption_undoes(self, session: SessionViewModel) -> None:
        analyse(session)
        session.adopt_detection(0, label="particle")

        assert session.undo() is True
        assert session.annotations == ()

    def test_an_index_no_run_has_is_refused(self, session: SessionViewModel) -> None:
        analyse(session)

        assert session.adopt_detection(10_000, label="particle") is False

    def test_an_empty_label_is_refused(self, session: SessionViewModel) -> None:
        analyse(session)
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.adopt_detection(0, label=" ") is False
        assert "needs a label" in said[-1]


class TestEditing:
    def test_renaming_goes_through_the_stack(self, session: SessionViewModel) -> None:
        """`UpdateAnnotation` has had no caller outside its own tests since
        M4-T08."""
        session.add_annotation((5.0, 5.0, 15.0, 15.0), label="blob")
        annotation_id = session.annotations[0].id

        assert session.rename_annotation(annotation_id, "contamination") is True
        assert session.annotations[0].label == "contamination"

        assert session.undo() is True
        assert session.annotations[0].label == "blob"

    def test_deleting_goes_through_the_stack_and_restores_the_same_row(
        self, session: SessionViewModel
    ) -> None:
        session.add_annotation((5.0, 5.0, 15.0, 15.0), label="blob")
        annotation_id = session.annotations[0].id

        assert session.remove_annotation(annotation_id) is True
        assert session.annotations == ()

        assert session.undo() is True
        assert [one.id for one in session.annotations] == [annotation_id]

    def test_renaming_to_nothing_is_refused(self, session: SessionViewModel) -> None:
        session.add_annotation((5.0, 5.0, 15.0, 15.0), label="blob")
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.rename_annotation(session.annotations[0].id, "  ") is False
        assert "needs a label" in said[-1]
        assert session.annotations[0].label == "blob"


class TestTheSelection:
    def test_clicking_a_box_selects_it(self, session: SessionViewModel) -> None:
        """Annotations are drawn above the detections precisely so a click
        reaches them first (ADR-0070 §1)."""
        viewer = ImageViewer(session)
        session.add_annotation((5.0, 5.0, 15.0, 15.0), label="blob")

        viewer.view.annotation_picked.emit(0)

        assert session.selected_annotation == session.annotations[0].id

    def test_the_selected_box_is_the_thicker_one(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        session.add_annotation((5.0, 5.0, 15.0, 15.0), label="one")
        session.add_annotation((20.0, 20.0, 30.0, 30.0), label="two")

        session.select_annotation(session.annotations[1].id)

        widths = [item.pen().widthF() for item in viewer.view.annotation_overlay]
        assert widths[1] > widths[0]

    def test_deleting_the_selected_one_clears_the_selection(
        self, session: SessionViewModel
    ) -> None:
        panel = AnnotatePanel(session)
        session.add_annotation((5.0, 5.0, 15.0, 15.0), label="blob")
        session.select_annotation(session.annotations[0].id)
        assert panel.delete.isEnabled()

        session.remove_annotation(session.selected_annotation)  # type: ignore[arg-type]

        assert session.selected_annotation is None
        assert not panel.delete.isEnabled()

    def test_an_id_this_image_does_not_have_is_refused(self, session: SessionViewModel) -> None:
        session.select_annotation(9_999)

        assert session.selected_annotation is None
