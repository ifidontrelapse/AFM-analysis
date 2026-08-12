"""Create, open, populate, close — headless (M4-T04, ADR-0041).

The first of M4's exit criteria, executed:

> *A project can be created, opened, populated with images and closed — from
> Python, headless.*

Everything here goes through the real adapter onto a real directory, because
the criterion is about a project an operator could open in a file manager, and
a fake would prove the opposite of what is being claimed. The use cases are
driven through the port, exactly as `app/` will drive them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanoscope.application.jobs import JobContext, JobRunner, JobState
from nanoscope.application.use_cases import import_images, open_project
from nanoscope.core.errors import InvalidParameterError, MissingFileError
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import (
    DIRECTORIES,
    MANIFEST_NAME,
    SqliteProjectRepository,
    sha256_of,
)


@pytest.fixture
def scans(tmp_path: Path) -> Path:
    """A folder of files to import, standing in for the operator's `data/`."""
    source = tmp_path / "from_the_instrument"
    source.mkdir()
    (source / "scan_01.spm").write_bytes(b"AFM one")
    (source / "scan_02.spm").write_bytes(b"AFM two")
    return source


def test_the_whole_lifecycle(tmp_path: Path, scans: Path) -> None:
    """The exit criterion, in one function, in the order an operator does it."""
    project_dir = tmp_path / "Nanoparticles 2026"

    with SqliteProjectRepository.create(project_dir, "Nanoparticles 2026") as repo:
        report = import_images(
            repo, sorted(scans.iterdir()), modality=Modality.AFM, pixel_size_nm=1.95
        )

    assert report.is_complete
    assert len(report.imported) == 2

    with SqliteProjectRepository.open(project_dir) as repo:
        opened = open_project(repo)

    assert opened.name == "Nanoparticles 2026"
    assert [image.display_name for image in opened.images] == ["scan_01.spm", "scan_02.spm"]
    assert opened.integrity.is_clean
    assert all(image.pixel_size_nm == 1.95 for image in opened.images)


class TestCreatingAProject:
    def test_the_directory_has_the_layout_the_contract_specifies(self, tmp_path: Path) -> None:
        with SqliteProjectRepository.create(tmp_path / "P", "P"):
            pass

        assert (tmp_path / "P" / MANIFEST_NAME).is_file()
        for directory in DIRECTORIES:
            assert (tmp_path / "P" / directory).is_dir()

    def test_an_existing_empty_directory_is_fine(self, tmp_path: Path) -> None:
        (tmp_path / "P").mkdir()

        with SqliteProjectRepository.create(tmp_path / "P", "P") as repo:
            assert repo.name == "P"

    def test_a_directory_with_something_in_it_is_refused(self, tmp_path: Path) -> None:
        """Writing a manifest into a folder that has files in it turns somebody
        else's `Documents/` into a project directory."""
        (tmp_path / "P").mkdir()
        (tmp_path / "P" / "thesis.odt").write_bytes(b"years of work")

        with pytest.raises(InvalidParameterError, match="not empty"):
            SqliteProjectRepository.create(tmp_path / "P", "P")

    def test_the_project_name_is_not_the_directory_name(self, tmp_path: Path) -> None:
        with SqliteProjectRepository.create(tmp_path / "2026-08-12-run", "Gold on mica") as repo:
            assert repo.name == "Gold on mica"


class TestImportingIntoTheProject:
    def test_the_file_is_copied_in_and_the_original_is_untouched(
        self, tmp_path: Path, scans: Path
    ) -> None:
        """`images/` holds a copy, byte-identical to what the operator imported.
        The source is theirs and stays where it is."""
        source = scans / "scan_01.spm"

        with SqliteProjectRepository.create(tmp_path / "P", "P") as repo:
            record = repo.import_image(source, modality=Modality.AFM)
            copied = repo.root / record.relative_path

            assert copied.read_bytes() == source.read_bytes()
            assert record.sha256 == sha256_of(source)

        assert source.is_file()

    def test_two_files_with_the_same_name_both_arrive(self, tmp_path: Path) -> None:
        """Two folders, two `scan.spm`, one instrument — the ordinary shape of
        this work. Refusing the second would be hostile; overwriting it would be
        worse."""
        first, second = tmp_path / "monday", tmp_path / "tuesday"
        for folder, content in ((first, b"one"), (second, b"two")):
            folder.mkdir()
            (folder / "scan.spm").write_bytes(content)

        with SqliteProjectRepository.create(tmp_path / "P", "P") as repo:
            a = repo.import_image(first / "scan.spm", modality=Modality.AFM)
            b = repo.import_image(second / "scan.spm", modality=Modality.AFM)

            assert a.relative_path == "images/scan.spm"
            assert b.relative_path == "images/scan_1.spm"
            assert (repo.root / b.relative_path).read_bytes() == b"two"

    def test_the_display_name_is_the_one_the_operator_gave_the_file(self, tmp_path: Path) -> None:
        """The suffix is a filesystem detail. `scan.spm` is what they called it,
        and what they will look for in a list."""
        for folder in ("monday", "tuesday"):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / "scan.spm").write_bytes(b"x")

        with SqliteProjectRepository.create(tmp_path / "P", "P") as repo:
            repo.import_image(tmp_path / "monday" / "scan.spm", modality=Modality.AFM)
            second = repo.import_image(tmp_path / "tuesday" / "scan.spm", modality=Modality.AFM)

            assert second.display_name == "scan.spm"

    def test_the_same_file_twice_is_two_images(self, tmp_path: Path, scans: Path) -> None:
        """Deliberate, and deferred: deduplicating by checksum needs a UNIQUE
        index, a migration, and an answer to whether two identical scans are
        ever legitimate — an operator's question (ADR-0041). What it must not be
        is silent, and two rows in a list are not silent."""
        with SqliteProjectRepository.create(tmp_path / "P", "P") as repo:
            repo.import_image(scans / "scan_01.spm", modality=Modality.AFM)
            repo.import_image(scans / "scan_01.spm", modality=Modality.AFM)

            assert len(repo.list_images()) == 2
            assert repo.check_integrity().is_clean

    def test_a_file_that_is_not_there_is_refused(self, tmp_path: Path) -> None:
        with (
            SqliteProjectRepository.create(tmp_path / "P", "P") as repo,
            pytest.raises(MissingFileError),
        ):
            repo.import_image(tmp_path / "nowhere.spm", modality=Modality.AFM)

    def test_a_bad_file_costs_only_itself(self, tmp_path: Path, scans: Path) -> None:
        """The batch policy, on the real adapter rather than a fake."""
        with SqliteProjectRepository.create(tmp_path / "P", "P") as repo:
            report = import_images(
                repo,
                [scans / "scan_01.spm", tmp_path / "nowhere.spm", scans / "scan_02.spm"],
                modality=Modality.AFM,
            )

            assert len(report.imported) == 2
            assert len(report.failed) == 1
            assert repo.check_integrity().is_clean

    def test_nothing_is_left_behind_by_a_failed_import(self, tmp_path: Path) -> None:
        """A copy that happened without a row would be an untracked file — the
        integrity check would find it, which is exactly the mess this asserts
        does not happen."""
        with SqliteProjectRepository.create(tmp_path / "P", "P") as repo:
            import_images(repo, [tmp_path / "nowhere.spm"], modality=Modality.AFM)

            assert list((repo.root / "images").iterdir()) == []
            assert repo.check_integrity().is_clean


class TestImportingAsAJob:
    """The batch under M4-T06's runner — the shape M5 will use."""

    def test_a_project_opened_here_is_usable_from_a_worker_thread(self, tmp_path: Path) -> None:
        """Python's `sqlite3` binds a connection to the thread that made it and
        refuses it anywhere else — so a project opened on the main thread was
        unusable inside every job, which is how this would have arrived in M5:
        as a crash in a background task. `connect` passes
        `check_same_thread=False` and the repository serialises itself
        (ADR-0043)."""
        with (
            SqliteProjectRepository.create(tmp_path / "P", "P") as repo,
            JobRunner(max_workers=1) as runner,
        ):
            job = runner.submit("listing", lambda ctx: repo.list_images())

            assert job.wait(5.0)
            assert job.state is JobState.SUCCEEDED, job.error

    def test_it_reports_which_file_it_is_on(self, tmp_path: Path, scans: Path) -> None:
        seen: list[tuple[int, int]] = []

        with (
            SqliteProjectRepository.create(tmp_path / "P", "P") as repo,
            JobRunner(max_workers=1) as runner,
        ):
            job = runner.submit(
                "importing",
                lambda ctx: import_images(
                    repo, sorted(scans.iterdir()), modality=Modality.AFM, progress=ctx
                ),
                listener=lambda j: seen.append((j.progress.done, j.progress.total)),
            )
            assert job.wait(5.0)

            assert job.state is JobState.SUCCEEDED
            assert (0, 2) in seen and (2, 2) in seen
            assert len(job.result.imported) == 2

    def test_cancelling_keeps_what_was_already_imported(self, tmp_path: Path, scans: Path) -> None:
        """Between files is the only clean place to stop, and the files that
        made it are real files with real rows. The report says what was done
        before the stop rather than pretending nothing happened."""
        with (
            SqliteProjectRepository.create(tmp_path / "P", "P") as repo,
            JobRunner(max_workers=1) as runner,
        ):
            job = runner.submit(
                "importing",
                lambda ctx: import_images(
                    repo,
                    sorted(scans.iterdir()),
                    modality=Modality.AFM,
                    progress=_StopAfter(ctx, reports=1),
                ),
            )
            assert job.wait(5.0)

            report = job.result
            assert len(report.imported) == 1
            assert repo.list_images() == list(report.imported)
            assert repo.check_integrity().is_clean


class _StopAfter:
    """A `JobContext` that says "cancelled" once it has seen `reports` reports.

    A double rather than a sleep and a race: the cancellation lands at a known
    file, so the assertion is about behaviour and not about timing.
    """

    def __init__(self, inner: JobContext, *, reports: int) -> None:
        self._inner = inner
        self._limit = reports
        self._seen = 0

    @property
    def cancelled(self) -> bool:
        return self._seen > self._limit

    def raise_if_cancelled(self) -> None:
        self._inner.raise_if_cancelled()

    def report(self, done: int, total: int = 0, message: str = "") -> None:
        self._seen += 1
        self._inner.report(done, total, message)


class TestTheProjectSurvivesTheSession:
    def test_reopening_finds_everything(self, tmp_path: Path, scans: Path) -> None:
        with SqliteProjectRepository.create(tmp_path / "P", "P") as repo:
            import_images(repo, sorted(scans.iterdir()), modality=Modality.TEM)
            before = open_project(repo)

        with SqliteProjectRepository.open(tmp_path / "P") as repo:
            assert open_project(repo) == before

    def test_a_file_deleted_between_sessions_is_reported_not_lost(
        self, tmp_path: Path, scans: Path
    ) -> None:
        """The two halves of this milestone meeting: an operator deletes a scan
        with a file manager, and the next open says so without dropping the
        row (ADR-0040)."""
        with SqliteProjectRepository.create(tmp_path / "P", "P") as repo:
            record = repo.import_image(scans / "scan_01.spm", modality=Modality.AFM)

        (tmp_path / "P" / record.relative_path).unlink()

        with SqliteProjectRepository.open(tmp_path / "P") as repo:
            opened = open_project(repo)

            assert opened.images == (record,)
            assert opened.integrity.missing_files == (record,)
