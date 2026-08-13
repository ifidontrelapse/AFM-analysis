"""Every mutating action, and the stack that can take it back (M4-T08).

Architecture §4.5 fixed the shape before there was anything to undo: *a command
stack in `application/commands.py`; every mutating user action is a command with
`do()` and `undo()`; the GUI dispatches commands and never mutates state
directly.*

**The stack knows nothing but order.** It calls `do()`, keeps the object, and
clears the redo list — it never learns what a command *is*. A stack that knows
about annotations is a stack that has to be extended for measurements, and then
for settings (ADR-0045).

**Undo is a session, and that is a promise deliberately not made.** The history
does not survive closing the project. Persisting it would mean replaying edits
against a directory that may have changed on disk in between, and ADR-0040
already established that this application does not assume it is the only thing
touching a project. An undo history that can be wrong is worse than one that is
honestly short; the durable record of what happened is M4-T14's log, which is
history rather than reversibility.

**An annotation keeps its id through undo and redo**, which is what makes a
*sequence* of commands reversible: everything above an annotation on the stack
refers to it by id, so restoring it as a new row would leave every one of them
pointing at nothing. `restore_annotation` on the repository is the operation
that says so — creation and restoration are different acts, and only one of them
may choose an id (ADR-0045).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from nanoscope.core.entities import Annotation, AnnotationSource
from nanoscope.core.ports import ProjectRepository


class Command(Protocol):
    """One reversible action.

    `do()` must be safe to call again after `undo()`: redo is `do()`, not a
    third method. Whatever a command needs to reverse itself, it captures
    during `do()` — a command that reads it from the world at `undo()` time is
    a command that undoes whatever happens to be there instead.
    """

    @property
    def label(self) -> str:
        """What the menu item says, without "Undo" in front of it."""
        ...

    def do(self) -> None: ...

    def undo(self) -> None: ...


class CommandStack:
    """What has been done, and what has been taken back.

    Not thread-safe on purpose: undo/redo is a user's sequence of actions, and
    a second thread pushing onto it would mean two people editing one project
    through one history. Jobs (M4-T06) run *work*; commands are *edits*.
    """

    def __init__(self) -> None:
        self._done: list[Command] = []
        self._undone: list[Command] = []

    def run(self, command: Command) -> Command:
        """Do it, remember it, and drop anything that was waiting to be redone.

        The redo list is cleared because a new edit makes the old future
        unreachable — this is what every editor does, and doing anything else
        means describing a history that never happened.

        Returns:
            The command, so a caller can read what it produced.

        Raises:
            Anything `do()` raises. Nothing is remembered in that case: a
            command that failed is not on the history.
        """
        command.do()
        self._done.append(command)
        self._undone.clear()
        return command

    def undo(self) -> Command | None:
        """Take back the last command, or return `None` if there is none.

        Raises:
            Anything `undo()` raises — and the history does **not** move. The
            stack assumes it is the only writer; if something changed the
            project behind its back, the honest response is to say so and leave
            the history describing the state that actually exists.
        """
        if not self._done:
            return None
        command = self._done[-1]
        command.undo()
        self._done.pop()
        self._undone.append(command)
        return command

    def redo(self) -> Command | None:
        """Do the last undone command again, or return `None` if there is none."""
        if not self._undone:
            return None
        command = self._undone[-1]
        command.do()
        self._undone.pop()
        self._done.append(command)
        return command

    def clear(self) -> None:
        """Forget everything. What a project being closed does to its history."""
        self._done.clear()
        self._undone.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._done)

    @property
    def can_redo(self) -> bool:
        return bool(self._undone)

    @property
    def undo_label(self) -> str | None:
        """What "Undo" would take back, for the menu. `None` when nothing would."""
        return self._done[-1].label if self._done else None

    @property
    def redo_label(self) -> str | None:
        return self._undone[-1].label if self._undone else None


class AddAnnotation:
    """Draw a box. Undoing it removes the row; redoing it puts that row back.

    `annotation` is the current row, and it keeps its id across an undo/redo
    pair — so a command stacked above this one still refers to something.
    """

    def __init__(
        self,
        repository: ProjectRepository,
        image_id: int,
        box: tuple[float, float, float, float],
        *,
        label_text: str,
        source: AnnotationSource = AnnotationSource.MANUAL,
        note: str | None = None,
        points: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        self._repository = repository
        self._image_id = image_id
        self._box = box
        #: The outline, when there is one. It travels with the command so a
        #: **redo** puts the polygon back rather than its bounding box, which
        #: would quietly redraw the operator's work as a rectangle (M7-T03).
        self._points = points
        self._label_text = label_text
        self._source = source
        self._note = note
        self._removed: Annotation | None = None
        self.annotation: Annotation | None = None

    @property
    def label(self) -> str:
        return f"add {self._label_text}"

    def do(self) -> None:
        if self._removed is not None:
            # A redo: the same annotation, not another one like it.
            self.annotation = self._repository.restore_annotation(self._removed)
            self._removed = None
            return
        self.annotation = self._repository.add_annotation(
            self._image_id,
            self._box,
            label=self._label_text,
            source=self._source,
            note=self._note,
            points=self._points,
        )

    def undo(self) -> None:
        if self.annotation is not None:
            self._removed = self.annotation
            self._repository.remove_annotation(self.annotation.id)
            self.annotation = None


class UpdateAnnotation:
    """Move, relabel or annotate a box, reversibly.

    The previous values are captured in `do()`, not read at `undo()` time: a
    command that looks them up when it is undone would restore whatever
    happens to be there, which after a second edit is not what it changed.
    """

    def __init__(
        self,
        repository: ProjectRepository,
        annotation_id: int,
        *,
        box: tuple[float, float, float, float] | None = None,
        label_text: str | None = None,
        note: str | None = None,
    ) -> None:
        self._repository = repository
        self._id = annotation_id
        self._box = box
        self._label_text = label_text
        self._note = note
        self._before: Annotation | None = None
        self.annotation: Annotation | None = None

    @property
    def label(self) -> str:
        return "edit annotation"

    def do(self) -> None:
        self._before = self._repository.get_annotation(self._id)
        self.annotation = self._repository.update_annotation(
            self._id, box=self._box, label=self._label_text, note=self._note
        )

    def undo(self) -> None:
        if self._before is None:
            return
        self.annotation = self._repository.update_annotation(
            self._id,
            box=self._before.box,
            label=self._before.label,
            note=self._before.note,
        )


class RemoveAnnotation:
    """Delete a box, and be able to put it back — as itself.

    The whole row is captured before the deletion, so what comes back has the
    id, the label and the timestamps it had. Anything else would be a new
    annotation that merely looks the same.
    """

    def __init__(self, repository: ProjectRepository, annotation_id: int) -> None:
        self._repository = repository
        self._id = annotation_id
        self._removed: Annotation | None = None
        self.annotation: Annotation | None = None

    @property
    def label(self) -> str:
        return "delete annotation"

    def do(self) -> None:
        self._removed = self._repository.get_annotation(self._id)
        self._repository.remove_annotation(self._id)
        self.annotation = None

    def undo(self) -> None:
        if self._removed is None:
            return
        self.annotation = self._repository.restore_annotation(self._removed)
