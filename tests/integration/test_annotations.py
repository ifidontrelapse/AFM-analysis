"""The one thing in a project that cannot be recomputed (M4-T07, ADR-0044).

An image can be re-imported and an analysis re-run. A box somebody drew at
two in the morning exists once. That is why these are rows rather than a
rewritten document, why an edit keeps its id, why a deletion that matched
nothing is an error rather than a shrug — and why the cascade in
`test_removing_the_image_takes_them_with_it` is written down as a decision
instead of discovered in M6.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from schema_history import revert_to

from nanoscope.core.entities import Annotation, AnnotationSource
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import SqliteProjectRepository

BOX = (10.5, 20.5, 30.0, 44.0)


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SqliteProjectRepository]:
    """A project with one image in it, ready to be annotated."""
    with SqliteProjectRepository.create(tmp_path / "P", "P") as repository:
        (repository.root / "images" / "scan.spm").write_bytes(b"AFM")
        repository.add_image("images/scan.spm", modality=Modality.AFM)
        yield repository


def image_id(repo: SqliteProjectRepository) -> int:
    return repo.list_images()[0].id


class TestDrawingOne:
    def test_it_comes_back_as_it_was_drawn(self, repo: SqliteProjectRepository) -> None:
        stored = repo.add_annotation(image_id(repo), BOX, label="particle")

        assert repo.get_annotation(stored.id) == stored
        assert stored.box == BOX
        assert stored.label == "particle"

    def test_it_is_hand_drawn_unless_it_says_otherwise(self, repo: SqliteProjectRepository) -> None:
        """The default is the honest one: something a person did."""
        assert repo.add_annotation(image_id(repo), BOX, label="p").source is AnnotationSource.MANUAL

    def test_an_adopted_box_says_so(self, repo: SqliteProjectRepository) -> None:
        """M8 has to be able to exclude these: a model trained on boxes copied
        from its own output is confirming itself."""
        adopted = repo.add_annotation(
            image_id(repo), BOX, label="p", source=AnnotationSource.FROM_DETECTION
        )

        assert adopted.source is AnnotationSource.FROM_DETECTION

    def test_the_coordinates_are_not_rounded(self, repo: SqliteProjectRepository) -> None:
        """A drag is not on the pixel grid, and rounding is the trainer's
        decision to make with the whole box in hand, not the database's."""
        stored = repo.add_annotation(image_id(repo), (1.25, 2.75, 3.5, 4.5), label="p")

        assert stored.box == (1.25, 2.75, 3.5, 4.5)

    def test_a_note_is_optional_and_absent_when_not_given(
        self, repo: SqliteProjectRepository
    ) -> None:
        assert repo.add_annotation(image_id(repo), BOX, label="p").note is None
        assert repo.add_annotation(image_id(repo), BOX, label="p", note="odd").note == "odd"

    def test_a_box_with_no_area_is_refused(self, repo: SqliteProjectRepository) -> None:
        """A mis-drag. As a training example it is a picture of nothing."""
        with pytest.raises(InvalidParameterError, match="no area"):
            repo.add_annotation(image_id(repo), (10.0, 10.0, 10.0, 20.0), label="p")

    def test_an_inverted_box_is_refused(self, repo: SqliteProjectRepository) -> None:
        """`(x1, y1, x2, y2)` with `x2 > x1` is the project's box convention
        (PROJECT_RULES §3), not a preference of this table."""
        with pytest.raises(InvalidParameterError, match="no area"):
            repo.add_annotation(image_id(repo), (30.0, 44.0, 10.0, 20.0), label="p")

    def test_annotating_an_image_that_does_not_exist_is_refused(
        self, repo: SqliteProjectRepository
    ) -> None:
        with pytest.raises(InvalidParameterError, match="no image with id 99"):
            repo.add_annotation(99, BOX, label="p")


class TestTheOnesOnAnImage:
    def test_they_come_back_in_the_order_they_were_drawn(
        self, repo: SqliteProjectRepository
    ) -> None:
        first = repo.add_annotation(image_id(repo), BOX, label="a")
        second = repo.add_annotation(image_id(repo), (50.0, 50.0, 60.0, 60.0), label="b")

        assert repo.annotations_for(image_id(repo)) == [first, second]

    def test_an_image_nobody_annotated_has_none(self, repo: SqliteProjectRepository) -> None:
        assert repo.annotations_for(image_id(repo)) == []

    def test_they_survive_the_session(self, tmp_path: Path) -> None:
        with SqliteProjectRepository.create(tmp_path / "Q", "Q") as repo:
            (repo.root / "images" / "scan.spm").write_bytes(b"AFM")
            image = repo.add_image("images/scan.spm", modality=Modality.AFM)
            drawn = repo.add_annotation(image.id, BOX, label="particle", note="the odd one")

        with SqliteProjectRepository.open(tmp_path / "Q") as repo:
            assert repo.annotations_for(image.id) == [drawn]


class TestEditingOne:
    def test_moving_the_box_keeps_the_annotation(self, repo: SqliteProjectRepository) -> None:
        """An edit, not a delete-and-add: the id is what undo (M4-T08) and every
        later reference hold on to."""
        drawn = repo.add_annotation(image_id(repo), BOX, label="particle")

        moved = repo.update_annotation(drawn.id, box=(11.0, 21.0, 31.0, 45.0))

        assert moved.id == drawn.id
        assert moved.box == (11.0, 21.0, 31.0, 45.0)
        assert repo.annotations_for(image_id(repo)) == [moved]

    def test_what_is_not_given_is_not_changed(self, repo: SqliteProjectRepository) -> None:
        drawn = repo.add_annotation(image_id(repo), BOX, label="particle", note="odd")

        relabelled = repo.update_annotation(drawn.id, label="contaminant")

        assert (relabelled.box, relabelled.note) == (drawn.box, "odd")
        assert relabelled.label == "contaminant"

    def test_when_it_was_drawn_does_not_move(self, repo: SqliteProjectRepository) -> None:
        drawn = repo.add_annotation(image_id(repo), BOX, label="particle")

        edited = repo.update_annotation(drawn.id, label="other")

        assert edited.created_utc == drawn.created_utc

    def test_an_edit_to_no_area_is_refused(self, repo: SqliteProjectRepository) -> None:
        drawn = repo.add_annotation(image_id(repo), BOX, label="particle")

        with pytest.raises(InvalidParameterError, match="no area"):
            repo.update_annotation(drawn.id, box=(5.0, 5.0, 5.0, 9.0))

        assert repo.get_annotation(drawn.id).box == BOX

    def test_editing_something_that_is_not_there_is_refused(
        self, repo: SqliteProjectRepository
    ) -> None:
        with pytest.raises(InvalidParameterError, match="no annotation with id 42"):
            repo.update_annotation(42, label="x")


class TestDeletingOne:
    def test_it_goes(self, repo: SqliteProjectRepository) -> None:
        drawn = repo.add_annotation(image_id(repo), BOX, label="particle")

        repo.remove_annotation(drawn.id)

        assert repo.annotations_for(image_id(repo)) == []

    def test_deleting_something_absent_is_not_silent(self, repo: SqliteProjectRepository) -> None:
        """A typo must not look like a successful deletion — this is hand work."""
        with pytest.raises(InvalidParameterError, match="no annotation with id 42"):
            repo.remove_annotation(42)

    def test_removing_the_image_takes_them_with_it(self, repo: SqliteProjectRepository) -> None:
        """The decision written down: they cascade, because a box pointing at an
        image the project no longer knows about is not an annotation of
        anything. `remove_image` therefore discards hand work — which is why
        `annotations_for` exists to be counted **before** a confirmation dialog
        asks (ADR-0044)."""
        image = image_id(repo)
        repo.add_annotation(image, BOX, label="particle")
        assert len(repo.annotations_for(image)) == 1

        repo.remove_image(image)

        assert repo.annotations_for(image) == []


class TestTheMigrationThatBroughtThem:
    def test_a_project_at_v2_gains_the_table_and_keeps_its_rows(self, tmp_path: Path) -> None:
        """The third step of ADR-0039's mechanism, over a database with rows —
        and this time the older version is one that shipped in this milestone."""
        root = tmp_path / "P"
        with SqliteProjectRepository.create(root, "P") as repo:
            (root / "images" / "a.spm").write_bytes(b"AFM")
            recorded = repo.add_image("images/a.spm", modality=Modality.AFM)
            revert_to(repo._conn, 2)

        with SqliteProjectRepository.open(root) as repo:
            assert repo.list_images() == [recorded]
            drawn = repo.add_annotation(recorded.id, BOX, label="particle")
            assert isinstance(drawn, Annotation)
