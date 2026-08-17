"""Annotations out, and labels back in (M7-T09, ADR-0078).

**M7's fourth exit criterion:** *"annotations export to a format the M8 dataset
builder consumes"*. The format is YOLO's, and the assertions that matter are the
exact shape of a line, the **round trip**, and the two things the export cannot
carry — `source`, which the caller therefore chooses, and provenance, which the
import is told rather than guessing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nanoscope.app.container import Nanoscope
from nanoscope.application import use_cases
from nanoscope.application.jobs import Job
from nanoscope.core.entities.project import AnnotationSource
from nanoscope.core.errors import AnalysisFailedError, InvalidParameterError
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")

#: A 100 by 50 scan, so a test that swapped width for height fails instead of
#: passing by symmetry.
WIDTH, HEIGHT = 100, 50


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name in ("first.npy", "second.npy"):
            source = tmp_path / name
            np.save(source, np.zeros((HEIGHT, WIDTH), dtype=np.float32))
            repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
    return root


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        yield container


@pytest.fixture
def session(app: Nanoscope, project: Path) -> SessionViewModel:
    model = SessionViewModel(app)
    model.open_project(project)
    assert model.project is not None
    model.select_image(model.project.images[0].id)
    return model


def wait(job: Job | None) -> None:
    assert isinstance(job, Job)
    assert job.wait(30.0)


def exported(session: SessionViewModel, *, hand_drawn_only: bool = False) -> Path:
    wait(session.export_annotations(hand_drawn_only=hand_drawn_only))
    assert session.project is not None
    result = session.job.result if session.job else None
    assert isinstance(result, use_cases.AnnotationExport)
    return _root(session) / result.directory


def _root(session: SessionViewModel) -> Path:
    repository = session._app.repository
    assert repository is not None
    return Path(repository.root)


class TestTheLineIsYolos:
    def test_a_box_is_five_normalised_fields(self, session: SessionViewModel) -> None:
        """`class cx cy w h`, normalised to the image. The whole contract with
        every tool in this ecosystem, asserted as text."""
        session.add_annotation((10.0, 10.0, 30.0, 20.0), label="particle")

        where = exported(session)

        line = (where / "labels" / "first.txt").read_text().strip()
        assert line == "0 0.200000 0.300000 0.200000 0.200000"

    def test_the_class_list_is_the_index(self, session: SessionViewModel) -> None:
        session.add_annotation((10.0, 10.0, 30.0, 20.0), label="rod")
        session.add_annotation((40.0, 10.0, 60.0, 20.0), label="particle")

        where = exported(session)

        #: Sorted, so two exports of the same project agree about what class 0 is.
        assert (where / "classes.txt").read_text().split() == ["particle", "rod"]
        indices = [
            line.split()[0] for line in (where / "labels" / "first.txt").read_text().splitlines()
        ]
        assert sorted(indices) == ["0", "1"]

    def test_a_box_over_the_edge_is_clamped_not_refused(self, session: SessionViewModel) -> None:
        """A drag that ran off the scan is an ordinary thing an operator does;
        the part that is on the image is what they meant."""
        session.add_annotation((-20.0, -10.0, 50.0, 25.0), label="particle")

        where = exported(session)

        _index, centre_x, centre_y, width, height = (
            (where / "labels" / "first.txt").read_text().split()
        )
        assert (float(centre_x), float(width)) == (0.25, 0.5)
        assert (float(centre_y), float(height)) == (0.25, 0.5)

    def test_a_polygon_exports_as_the_box_the_row_already_holds(
        self, session: SessionViewModel
    ) -> None:
        """Nothing is lost that the row did not lose first: a polygon's box is
        stored beside its outline (ADR-0072)."""
        session.add_polygon(((10.0, 10.0), (30.0, 10.0), (30.0, 20.0)), label="particle")

        where = exported(session)

        assert (where / "labels" / "first.txt").read_text().strip() == (
            "0 0.200000 0.300000 0.200000 0.200000"
        )


class TestTheScopeIsNamed:
    def test_hand_drawn_only_leaves_the_adopted_boxes_out(self, session: SessionViewModel) -> None:
        """ADR-0044's reason, at the surface that would undo it: a model trained
        on boxes copied from its own output is confirming itself."""
        session.add_annotation((10.0, 10.0, 30.0, 20.0), label="particle")
        repository = session._app.repository
        assert repository is not None and session.image_id is not None
        repository.add_annotation(
            session.image_id,
            (40.0, 10.0, 60.0, 20.0),
            label="particle",
            source=AnnotationSource.FROM_DETECTION,
        )
        session.reload_annotations()

        hand_drawn = exported(session, hand_drawn_only=True)
        everything = exported(session)

        #: Two directories, because the scope is in the name — the two exports
        #: are a second apart and mixing them is a training set that quietly
        #: contains the model's own output.
        assert hand_drawn != everything
        assert len((hand_drawn / "labels" / "first.txt").read_text().splitlines()) == 1
        assert len((everything / "labels" / "first.txt").read_text().splitlines()) == 2

    def test_nothing_drawn_is_a_sentence_not_an_empty_directory(
        self, session: SessionViewModel
    ) -> None:
        """An empty label set reads as *"nothing was drawn"*, which is a
        different statement (ADR-0048's rule, second site)."""
        repository = session._app.repository
        assert repository is not None

        with pytest.raises(AnalysisFailedError, match="nothing to export"):
            use_cases.export_annotations(repository)

    def test_a_scan_with_no_annotation_gets_no_label_file(self, session: SessionViewModel) -> None:
        session.add_annotation((10.0, 10.0, 30.0, 20.0), label="particle")

        where = exported(session)

        assert not (where / "labels" / "second.txt").exists()


class TestTheRoundTrip:
    def test_what_went_out_comes_back(self, session: SessionViewModel) -> None:
        """The criterion, asserted end to end: the boxes survive the trip within
        the precision the format writes."""
        session.add_annotation((10.0, 10.0, 30.0, 20.0), label="particle")
        where = exported(session)
        session.remove_annotation(session.annotations[0].id)
        assert session.annotations == ()

        assert session.import_annotations(where, source=AnnotationSource.MANUAL) == 1

        restored = session.annotations[0]
        assert restored.label == "particle"
        assert restored.box == pytest.approx((10.0, 10.0, 30.0, 20.0), abs=1e-3)

    def test_the_source_is_stated_and_the_file_is_remembered(
        self, session: SessionViewModel
    ) -> None:
        """A `.txt` says nothing about who drew the box, so the operator does —
        and where it came from goes into the note (ADR-0044, ADR-0078 §4)."""
        session.add_annotation((10.0, 10.0, 30.0, 20.0), label="particle")
        where = exported(session)

        session.import_annotations(where, source=AnnotationSource.FROM_DETECTION)

        imported = session.annotations[-1]
        assert imported.source is AnnotationSource.FROM_DETECTION
        assert imported.note is not None and where.name in imported.note

    def test_a_whole_import_is_one_undo(self, session: SessionViewModel) -> None:
        """`Composite`'s second caller: two hundred labels are one `Ctrl+Z`
        (ADR-0077 §3)."""
        session.add_annotation((10.0, 10.0, 30.0, 20.0), label="particle")
        session.add_annotation((40.0, 10.0, 60.0, 20.0), label="particle")
        where = exported(session)

        assert session.import_annotations(where, source=AnnotationSource.MANUAL) == 2
        assert len(session.annotations) == 4
        assert session.undo_label == "import 2 label(s)"

        assert session.undo() is True

        assert len(session.annotations) == 2

    def test_labels_land_on_the_scan_they_name(self, session: SessionViewModel) -> None:
        """Matched by file stem, which is what makes the directory portable
        between this project and any tool that wrote it."""
        assert session.project is not None
        first, second = (image.id for image in session.project.images)
        session.select_image(second)
        session.add_annotation((10.0, 10.0, 30.0, 20.0), label="particle")
        where = exported(session)
        session.remove_annotation(session.annotations[0].id)

        session.import_annotations(where, source=AnnotationSource.MANUAL)

        repository = session._app.repository
        assert repository is not None
        assert repository.annotations_for(first) == []
        assert len(repository.annotations_for(second)) == 1


class TestWhatIsRefusedAndWhatIsReported:
    def test_a_label_file_naming_no_image_is_reported_not_raised(
        self, session: SessionViewModel, tmp_path: Path
    ) -> None:
        """ADR-0040's shape: a directory of labels for a bigger dataset is a
        normal thing to import from."""
        session.add_annotation((10.0, 10.0, 30.0, 20.0), label="particle")
        where = exported(session)
        (where / "labels" / "somebody_elses_scan.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        said: list[str] = []
        session.reported.connect(said.append)

        assert session.import_annotations(where, source=AnnotationSource.MANUAL) == 1

        assert "1 file(s) named no image here" in said[-1]

    def test_a_class_index_the_list_does_not_have_is_refused(self) -> None:
        with pytest.raises(InvalidParameterError, match="not in a list of 1"):
            use_cases.parse_labels("3 0.5 0.5 0.2 0.2", ("particle",), width=WIDTH, height=HEIGHT)

    def test_a_coordinate_outside_the_image_is_refused(self) -> None:
        """A YOLO label is normalised; 1.4 is not describing this image, and
        importing it anyway puts a box somewhere nobody drew one."""
        with pytest.raises(InvalidParameterError, match=r"outside \[0, 1\]"):
            use_cases.parse_labels("0 1.4 0.5 0.2 0.2", ("particle",), width=WIDTH, height=HEIGHT)

    def test_a_line_that_is_not_five_fields_is_refused(self) -> None:
        with pytest.raises(InvalidParameterError, match="got 4 field"):
            use_cases.parse_labels("0 0.5 0.5 0.2", ("particle",), width=WIDTH, height=HEIGHT)

    def test_a_directory_without_a_class_list_is_refused(
        self, session: SessionViewModel, tmp_path: Path
    ) -> None:
        """An index without the list is a number, not a label."""
        bare = tmp_path / "labels_only"
        bare.mkdir()
        (bare / "first.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.import_annotations(bare, source=AnnotationSource.MANUAL) == 0

        assert "classes.txt" in said[-1]
        assert session.annotations == ()

    def test_an_export_reads_the_scan_and_says_when_it_is_gone(
        self, session: SessionViewModel
    ) -> None:
        """Normalising needs the image's size, so a missing file is loud: a
        dataset silently missing one scan of twelve is wrong and looks right."""
        session.add_annotation((10.0, 10.0, 30.0, 20.0), label="particle")
        repository = session._app.repository
        assert repository is not None and session.image_id is not None
        repository.path_of(repository.get_image(session.image_id)).unlink()

        wait(session.export_annotations(hand_drawn_only=False))

        assert session.job is not None and session.job.error is not None


class TestTheWindow:
    def test_the_two_scopes_are_two_named_items(self, app: Nanoscope, project: Path) -> None:
        """ADR-0067's rule, one milestone on: an item that silently means one of
        two scopes is one somebody uses wrong exactly once."""
        window = MainWindow(app)

        assert window.export_hand_drawn_action.text() == "Export &Hand-Drawn Annotations…"
        assert window.export_annotations_action.text() == "Export All A&nnotations…"
        assert not window.export_hand_drawn_action.isEnabled()
        assert not window.import_annotations_action.isEnabled()

        window.open_project(project)

        assert window.export_hand_drawn_action.isEnabled()
        assert window.import_annotations_action.isEnabled()

    def test_the_export_says_where_it_went(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)
        window.session.select_image(window.session.project.images[0].id)  # type: ignore[union-attr]
        window.session.add_annotation((10.0, 10.0, 30.0, 20.0), label="particle")
        said: list[str] = []
        window.session.reported.connect(said.append)

        window.export_annotations(hand_drawn_only=True)
        wait(window.session.job)
        window.session.job_changed.emit(window.session.job)

        assert "1 box(es) over 1 scan(s)" in said[-1]
        assert "exports/annotations_" in said[-1]
