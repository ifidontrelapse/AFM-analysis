"""Nothing is waiting to be saved (M4-T09, ADR-0046).

Autosave was scheduled before there was any storage to autosave. There is no
dirty state: every mutating method commits before it returns, so the only work
a crash can lose is the operation in flight, which SQLite's own transaction
rolls back.

This file is that claim made executable. It kills repositories without closing
them, starts a **separate process** and shoots it between writes, and asserts
that what was written is there. If a future change introduces buffering — a
dirty flag, a batched write, an in-memory working copy — these go red, and
ADR-0046 gets revisited with evidence rather than by argument.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import SqliteProjectRepository

BOX = (10.0, 20.0, 30.0, 40.0)


def project_with_an_image(root: Path) -> int:
    """A project, an image in it, and the image's id."""
    with SqliteProjectRepository.create(root, "P") as repo:
        (repo.root / "images" / "scan.spm").write_bytes(b"AFM")
        return repo.add_image("images/scan.spm", modality=Modality.AFM).id


class TestAWriteIsDurableWhenItReturns:
    def test_a_repository_that_is_never_closed_still_saved(self, tmp_path: Path) -> None:
        """No `close()`, no context manager, no flush — the row is on disk
        because `add_annotation` committed before it returned."""
        image = project_with_an_image(tmp_path / "P")

        abandoned = SqliteProjectRepository.open(tmp_path / "P")
        drawn = abandoned.add_annotation(image, BOX, label="particle")
        del abandoned  # no close: exactly what a crash leaves behind

        with SqliteProjectRepository.open(tmp_path / "P") as repo:
            assert repo.annotations_for(image) == [drawn]

    def test_another_connection_sees_it_immediately(self, tmp_path: Path) -> None:
        """Not just after a close: a second reader opened while the first is
        still alive sees the committed row. There is no window in which the
        edit exists only in memory."""
        image = project_with_an_image(tmp_path / "P")

        with SqliteProjectRepository.open(tmp_path / "P") as writer:
            drawn = writer.add_annotation(image, BOX, label="particle")

            with SqliteProjectRepository.open(tmp_path / "P") as reader:
                assert reader.annotations_for(image) == [drawn]

    def test_every_kind_of_write_is_committed(self, tmp_path: Path) -> None:
        """One assertion per write path, because "everything commits" is only
        true until somebody adds the one that does not."""
        root = tmp_path / "P"
        image = project_with_an_image(root)

        unclosed = SqliteProjectRepository.open(root)
        drawn = unclosed.add_annotation(image, BOX, label="particle")
        edited = unclosed.update_annotation(drawn.id, label="dust")
        second = unclosed.add_annotation(image, (50.0, 50.0, 60.0, 60.0), label="other")
        unclosed.remove_annotation(second.id)
        del unclosed

        with SqliteProjectRepository.open(root) as repo:
            assert repo.annotations_for(image) == [edited]


@pytest.mark.slow
class TestAKilledProcess:
    """The one test here that spawns a process, and therefore the one that reads
    the code from **disk** rather than from the parent's memory. It fails
    honestly if the working tree changes while the suite is running — which is
    not a property of the application, but is worth knowing before someone
    debugs it as one."""

    def test_what_was_written_before_the_kill_is_there(self, tmp_path: Path) -> None:
        """The real shape of the question autosave was scheduled to answer: a
        process that dies without warning. `SIGKILL`, no handler, no flush —
        and the annotations written before it survive."""
        root = tmp_path / "P"
        image = project_with_an_image(root)

        script = textwrap.dedent(f"""
            import os, signal
            from nanoscope.infrastructure.storage import SqliteProjectRepository

            repo = SqliteProjectRepository.open({str(root)!r})
            for index in range(3):
                repo.add_annotation({image}, (float(index), 1.0, float(index) + 5.0, 9.0),
                                    label="particle")
            os.kill(os.getpid(), signal.SIGKILL)
        """)
        killed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, check=False, timeout=120
        )

        assert killed.returncode == -9, killed.stderr.decode()
        with SqliteProjectRepository.open(root) as repo:
            assert len(repo.annotations_for(image)) == 3


class TestWhatIsDisposableIsDisposable:
    def test_deleting_the_cache_costs_nothing(self, tmp_path: Path) -> None:
        """ADR-0003's promise, and the other half of "nothing is waiting to be
        saved": what a project keeps in `cache/` is never the only copy."""
        root = tmp_path / "P"
        image = project_with_an_image(root)
        with SqliteProjectRepository.open(root) as repo:
            drawn = repo.add_annotation(image, BOX, label="particle")
            (root / "cache" / "thumbnail.png").write_bytes(b"PNG")

        shutil.rmtree(root / "cache")

        with SqliteProjectRepository.open(root) as repo:
            assert repo.annotations_for(image) == [drawn]
            assert repo.check_integrity().is_clean
