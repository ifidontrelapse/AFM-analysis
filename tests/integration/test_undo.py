"""Undo against a real database (M4-T08, ADR-0045).

M4's exit criterion — *"undo/redo proven on at least one mutating use case"* —
is the reason this file exists, and the criterion is about the **database**
going back, not about a stack popping. Every assertion here reads the project,
not the command.

The test that earned the design is `test_a_sequence_undoes_and_redoes_whole`:
with a *new* id on each redo, everything stacked above an annotation would
point at a row that no longer exists, and undo would work only one command
deep.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from nanoscope.application.commands import (
    AddAnnotation,
    CommandStack,
    RemoveAnnotation,
    UpdateAnnotation,
)
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import SqliteProjectRepository

BOX = (10.0, 20.0, 30.0, 40.0)
MOVED = (11.0, 21.0, 31.0, 41.0)


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SqliteProjectRepository]:
    with SqliteProjectRepository.create(tmp_path / "P", "P") as repository:
        (repository.root / "images" / "scan.spm").write_bytes(b"AFM")
        repository.add_image("images/scan.spm", modality=Modality.AFM)
        yield repository


def image_id(repo: SqliteProjectRepository) -> int:
    return repo.list_images()[0].id


class TestAddingABox:
    def test_undo_removes_it_from_the_database(self, repo: SqliteProjectRepository) -> None:
        stack = CommandStack()
        stack.run(AddAnnotation(repo, image_id(repo), BOX, label_text="particle"))

        stack.undo()

        assert repo.annotations_for(image_id(repo)) == []

    def test_redo_puts_back_the_same_annotation(self, repo: SqliteProjectRepository) -> None:
        """The same one, not another like it: its id, its timestamps, itself."""
        stack = CommandStack()
        command = stack.run(AddAnnotation(repo, image_id(repo), BOX, label_text="particle"))
        before = command.annotation
        assert before is not None

        stack.undo()
        stack.redo()

        assert repo.annotations_for(image_id(repo)) == [before]


class TestEditingABox:
    def test_undo_restores_the_previous_values(self, repo: SqliteProjectRepository) -> None:
        drawn = repo.add_annotation(image_id(repo), BOX, label="particle", note="odd")
        stack = CommandStack()
        stack.run(UpdateAnnotation(repo, drawn.id, box=MOVED, label_text="contaminant"))

        stack.undo()

        assert repo.get_annotation(drawn.id).box == BOX
        assert repo.get_annotation(drawn.id).label == "particle"
        assert repo.get_annotation(drawn.id).note == "odd"

    def test_it_restores_what_it_changed_not_what_is_there_now(
        self, repo: SqliteProjectRepository
    ) -> None:
        """The previous values are captured when the command runs. A command
        that looked them up at undo time would restore the *second* edit's
        starting point, not its own."""
        drawn = repo.add_annotation(image_id(repo), BOX, label="particle")
        stack = CommandStack()
        stack.run(UpdateAnnotation(repo, drawn.id, box=MOVED))
        stack.run(UpdateAnnotation(repo, drawn.id, box=(90.0, 90.0, 99.0, 99.0)))

        stack.undo()
        stack.undo()

        assert repo.get_annotation(drawn.id).box == BOX


class TestDeletingABox:
    def test_undo_brings_it_back_as_itself(self, repo: SqliteProjectRepository) -> None:
        drawn = repo.add_annotation(image_id(repo), BOX, label="particle", note="the odd one")
        stack = CommandStack()
        stack.run(RemoveAnnotation(repo, drawn.id))
        assert repo.annotations_for(image_id(repo)) == []

        stack.undo()

        assert repo.annotations_for(image_id(repo)) == [drawn]

    def test_redo_deletes_it_again(self, repo: SqliteProjectRepository) -> None:
        drawn = repo.add_annotation(image_id(repo), BOX, label="particle")
        stack = CommandStack()
        stack.run(RemoveAnnotation(repo, drawn.id))
        stack.undo()

        stack.redo()

        assert repo.annotations_for(image_id(repo)) == []


class TestASequence:
    def test_a_sequence_undoes_and_redoes_whole(self, repo: SqliteProjectRepository) -> None:
        """The test that decided the design. Add, then edit the thing that was
        added: if the redo of the add produced a *new* id, the redo of the edit
        would raise on a row that does not exist, and undo would be one command
        deep in practice."""
        stack = CommandStack()
        added = stack.run(AddAnnotation(repo, image_id(repo), BOX, label_text="particle"))
        assert added.annotation is not None
        stack.run(UpdateAnnotation(repo, added.annotation.id, box=MOVED, label_text="dust"))

        stack.undo()
        stack.undo()
        assert repo.annotations_for(image_id(repo)) == []

        stack.redo()
        stack.redo()

        restored = repo.annotations_for(image_id(repo))
        assert len(restored) == 1
        assert restored[0].box == MOVED
        assert restored[0].label == "dust"

    def test_the_ids_survive_the_round_trip(self, repo: SqliteProjectRepository) -> None:
        stack = CommandStack()
        added = stack.run(AddAnnotation(repo, image_id(repo), BOX, label_text="particle"))
        assert added.annotation is not None
        original_id = added.annotation.id

        stack.undo()
        stack.redo()

        assert repo.annotations_for(image_id(repo))[0].id == original_id


class TestWhenTheWorldMovedUnderneath:
    def test_an_undo_that_cannot_happen_says_so(self, repo: SqliteProjectRepository) -> None:
        """Something deleted the row behind the stack's back — a cascade, a
        script, another window. The undo fails loudly and the history stays
        where it was, rather than pretending (ADR-0045 §4)."""
        drawn = repo.add_annotation(image_id(repo), BOX, label="particle")
        stack = CommandStack()
        stack.run(UpdateAnnotation(repo, drawn.id, box=MOVED))
        repo.remove_annotation(drawn.id)

        with pytest.raises(InvalidParameterError):
            stack.undo()

        assert stack.can_undo
