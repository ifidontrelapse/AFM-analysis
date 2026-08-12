"""The stack, and the promises it makes about order (M4-T08, ADR-0045).

The stack is tested against commands that do nothing but record being called,
because that is all it is entitled to know about them. What the annotation
commands actually do to a database is `tests/integration/test_undo.py`, where
there is a database to check.
"""

from __future__ import annotations

import pytest

from nanoscope.application.commands import CommandStack


class Recording:
    """A command that only remembers what was done to it."""

    def __init__(self, name: str = "edit", *, fails_undo: bool = False) -> None:
        self.label = name
        self.calls: list[str] = []
        self._fails_undo = fails_undo

    def do(self) -> None:
        self.calls.append("do")

    def undo(self) -> None:
        if self._fails_undo:
            raise RuntimeError("something else changed it")
        self.calls.append("undo")


class TestRunning:
    def test_it_does_the_thing(self) -> None:
        stack, command = CommandStack(), Recording()

        stack.run(command)

        assert command.calls == ["do"]
        assert stack.can_undo

    def test_a_command_that_failed_is_not_on_the_history(self) -> None:
        class Failing(Recording):
            def do(self) -> None:
                raise ValueError("no")

        stack = CommandStack()

        with pytest.raises(ValueError, match="no"):
            stack.run(Failing())

        assert not stack.can_undo

    def test_a_new_command_drops_what_was_waiting_to_be_redone(self) -> None:
        """A new edit makes the old future unreachable — anything else describes
        a history that never happened."""
        stack = CommandStack()
        stack.run(Recording("first"))
        stack.undo()
        assert stack.can_redo

        stack.run(Recording("second"))

        assert not stack.can_redo


class TestUndoAndRedo:
    def test_undo_then_redo_returns_to_where_it_was(self) -> None:
        stack, command = CommandStack(), Recording()
        stack.run(command)

        stack.undo()
        stack.redo()

        assert command.calls == ["do", "undo", "do"]
        assert stack.can_undo and not stack.can_redo

    def test_they_happen_in_reverse_order(self) -> None:
        """Last in, first undone — which is also what makes restoring an id
        safe: anything created after a deletion is undone before it."""
        stack = CommandStack()
        first, second = Recording("first"), Recording("second")
        stack.run(first)
        stack.run(second)

        assert stack.undo() is second
        assert stack.undo() is first

    def test_an_empty_stack_says_no_rather_than_raising(self) -> None:
        stack = CommandStack()

        assert stack.undo() is None
        assert stack.redo() is None

    def test_a_failing_undo_leaves_the_history_where_it_was(self) -> None:
        """The stack assumes it is the only writer. When it is wrong, saying so
        and standing still beats a pointer describing a state that never
        existed."""
        stack = CommandStack()
        stack.run(Recording("fragile", fails_undo=True))

        with pytest.raises(RuntimeError, match="changed it"):
            stack.undo()

        assert stack.can_undo
        assert not stack.can_redo


class TestWhatTheMenuShows:
    def test_the_labels_name_the_next_action(self) -> None:
        stack = CommandStack()
        stack.run(Recording("add particle"))

        assert stack.undo_label == "add particle"
        assert stack.redo_label is None

        stack.undo()

        assert stack.undo_label is None
        assert stack.redo_label == "add particle"

    def test_clearing_forgets_everything(self) -> None:
        """What closing a project does to its history, since undo is a session
        and not a promise beyond one (ADR-0045)."""
        stack = CommandStack()
        stack.run(Recording())
        stack.undo()
        stack.run(Recording())

        stack.clear()

        assert not stack.can_undo
        assert not stack.can_redo
