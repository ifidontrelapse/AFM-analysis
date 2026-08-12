"""A project on a real disk, with a real database in it (M4-T03, ADR-0040).

Integration rather than unit, and the distinction is not bureaucratic: every
interesting failure of a repository is a disagreement between two systems that
each work perfectly. So nothing here is mocked — the directory exists, the
database is a file, and the tests that matter most do the things that go wrong
in the field:

- a file **deleted behind the application's back**, which ADR-0003 named as the
  price of two sources of truth and left unowned until this task
- a file dropped into `images/` by an operator with a file manager
- the whole project **moved to another directory**, which is what every relative
  path in the design is for, and which ADR-0003 asks for by name
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from nanoscope.core.entities import ImageRecord, IntegrityReport
from nanoscope.core.errors import InvalidParameterError, MissingFileError, ProjectFormatError
from nanoscope.core.ports import ProjectRepository
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import (
    DIRECTORIES,
    SqliteProjectRepository,
    new_manifest,
    sha256_of,
    write_manifest,
)


def make_project(root: Path, name: str = "MyProject") -> Path:
    """A project directory, built by hand.

    `CreateProject` is M4-T04 and owns the lifecycle; these three lines are the
    deliberate alternative to this task quietly taking it over.
    """
    root.mkdir(parents=True, exist_ok=True)
    for directory in DIRECTORIES:
        (root / directory).mkdir(exist_ok=True)
    write_manifest(root, new_manifest(name))
    return root


def put_scan(root: Path, name: str = "scan.spm", content: bytes = b"AFM") -> str:
    """A file under `images/`, and the path the repository will store."""
    (root / "images" / name).write_bytes(content)
    return f"images/{name}"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return make_project(tmp_path / "MyProject")


@pytest.fixture
def repo(project: Path) -> Iterator[SqliteProjectRepository]:
    with SqliteProjectRepository.open(project) as repository:
        yield repository


class TestOpeningAProject:
    def test_it_reads_its_name_from_the_manifest(self, project: Path) -> None:
        """Authoritative for identity, and readable when the database is not."""
        with SqliteProjectRepository.open(project) as repo:
            assert repo.name == "MyProject"

    def test_it_creates_the_database_on_first_open(self, project: Path) -> None:
        with SqliteProjectRepository.open(project) as repo:
            assert repo.list_images() == []

    def test_a_directory_that_is_not_a_project_is_refused(self, tmp_path: Path) -> None:
        """No manifest, no project — never guessed from the presence of `images/`."""
        (tmp_path / "images").mkdir()

        with pytest.raises(ProjectFormatError, match="not a project directory"):
            SqliteProjectRepository.open(tmp_path)

    def test_it_satisfies_the_port(self, repo: SqliteProjectRepository) -> None:
        """Structurally, without importing it — which is what keeps the arrow
        pointing inward. mypy checks this statically; this asserts it at run
        time so a signature change cannot pass unnoticed in an untyped caller."""
        port: ProjectRepository = repo

        assert isinstance(port.list_images(), list)


class TestRecordingAnImage:
    def test_a_recorded_image_comes_back(self, repo: SqliteProjectRepository) -> None:
        path = put_scan(repo.root)

        stored = repo.add_image(path, modality=Modality.AFM, pixel_size_nm=1.95)

        assert repo.get_image(stored.id) == stored
        assert repo.list_images() == [stored]

    def test_it_is_an_entity_not_a_row(self, repo: SqliteProjectRepository) -> None:
        """A `sqlite3.Row` would put the database's vocabulary into every layer
        above this one, and it is untyped besides."""
        record = repo.add_image(put_scan(repo.root), modality=Modality.AFM)

        assert isinstance(record, ImageRecord)
        assert record.modality is Modality.AFM

    def test_the_display_name_defaults_to_the_file_name(
        self, repo: SqliteProjectRepository
    ) -> None:
        record = repo.add_image(put_scan(repo.root, "sample_01.spm"), modality=Modality.AFM)

        assert record.display_name == "sample_01.spm"

    def test_an_unknown_scale_stays_unknown(self, repo: SqliteProjectRepository) -> None:
        """`None` survives the round trip. A fabricated 1.0 is indistinguishable
        from a measured one, which is the whole of D-07 (ADR-0019, ADR-0025)."""
        record = repo.add_image(put_scan(repo.root, "a.npy"), modality=Modality.AFM)

        assert record.pixel_size_nm is None
        assert repo.get_image(record.id).pixel_size_nm is None

    def test_the_checksum_is_of_the_file_it_points_at(self, repo: SqliteProjectRepository) -> None:
        """Computed here, never passed in: a checksum a caller supplies can
        describe a different file, and then it only proves the callers agreed."""
        path = put_scan(repo.root, "scan.spm", b"height data")

        record = repo.add_image(path, modality=Modality.AFM)

        assert record.sha256 == sha256_of(repo.root / path)

    def test_an_absolute_path_inside_the_project_is_stored_relative(
        self, repo: SqliteProjectRepository
    ) -> None:
        """The caller usually has an absolute path; refusing it would only move
        this conversion into every caller, which is where it goes wrong."""
        put_scan(repo.root)

        record = repo.add_image(str(repo.root / "images" / "scan.spm"), modality=Modality.AFM)

        assert record.relative_path == "images/scan.spm"

    def test_a_path_outside_the_project_is_refused(
        self, repo: SqliteProjectRepository, tmp_path: Path
    ) -> None:
        """It would not survive the directory being moved, which is the whole
        reason ADR-0003 requires relative paths."""
        outside = tmp_path / "elsewhere.spm"
        outside.write_bytes(b"AFM")

        with pytest.raises(InvalidParameterError, match="outside the project"):
            repo.add_image(str(outside), modality=Modality.AFM)

    def test_a_path_that_escapes_upward_is_refused(self, repo: SqliteProjectRepository) -> None:
        with pytest.raises(InvalidParameterError, match="escapes the project"):
            repo.add_image("../elsewhere.spm", modality=Modality.AFM)

    def test_a_file_that_is_not_there_is_refused(self, repo: SqliteProjectRepository) -> None:
        """A row whose file does not exist is the dangling row the integrity
        check reports; there is no reason to create one deliberately."""
        with pytest.raises(MissingFileError):
            repo.add_image("images/absent.spm", modality=Modality.AFM)

    def test_images_come_back_in_import_order(self, repo: SqliteProjectRepository) -> None:
        first = repo.add_image(put_scan(repo.root, "b.spm"), modality=Modality.AFM)
        second = repo.add_image(put_scan(repo.root, "a.spm"), modality=Modality.AFM)

        assert [image.id for image in repo.list_images()] == [first.id, second.id]

    def test_an_unknown_id_is_refused_by_name(self, repo: SqliteProjectRepository) -> None:
        with pytest.raises(InvalidParameterError, match="no image with id 42"):
            repo.get_image(42)


class TestForgettingAnImage:
    def test_the_row_goes(self, repo: SqliteProjectRepository) -> None:
        record = repo.add_image(put_scan(repo.root), modality=Modality.AFM)

        repo.remove_image(record.id)

        assert repo.list_images() == []

    def test_the_file_stays(self, repo: SqliteProjectRepository) -> None:
        """Forgetting a scan and deleting it are different decisions, and this
        layer does not get to make the second one."""
        record = repo.add_image(put_scan(repo.root), modality=Modality.AFM)

        repo.remove_image(record.id)

        assert (repo.root / record.relative_path).is_file()

    def test_removing_something_absent_is_not_silent(self, repo: SqliteProjectRepository) -> None:
        with pytest.raises(InvalidParameterError, match="no image with id 42"):
            repo.remove_image(42)


class TestTheIntegrityCheck:
    def test_a_project_that_agrees_with_itself_is_clean(
        self, repo: SqliteProjectRepository
    ) -> None:
        repo.add_image(put_scan(repo.root), modality=Modality.AFM)

        assert repo.check_integrity() == IntegrityReport()
        assert repo.check_integrity().is_clean

    def test_an_empty_project_is_clean(self, repo: SqliteProjectRepository) -> None:
        assert repo.check_integrity().is_clean

    def test_a_file_deleted_behind_our_back_is_reported(
        self, repo: SqliteProjectRepository
    ) -> None:
        """ADR-0003 named this as the price of two sources of truth and left it
        unowned. This is the collection."""
        record = repo.add_image(put_scan(repo.root), modality=Modality.AFM)
        (repo.root / record.relative_path).unlink()

        report = repo.check_integrity()

        assert report.missing_files == (record,)
        assert not report.is_clean

    def test_a_missing_file_does_not_cost_the_row(self, repo: SqliteProjectRepository) -> None:
        """The decision this task is really about. A missing file is as likely to
        be an unmounted drive as a deletion, and the row carries measurements the
        file does not — so reporting is the contract, and acting on it is not."""
        record = repo.add_image(put_scan(repo.root), modality=Modality.AFM)
        (repo.root / record.relative_path).unlink()

        repo.check_integrity()

        assert repo.list_images() == [record]

    def test_a_file_nobody_imported_is_reported(self, repo: SqliteProjectRepository) -> None:
        """An operator with a file manager, which is a workflow ADR-0003 chose
        to support rather than prevent."""
        put_scan(repo.root, "dropped_in.spm")

        report = repo.check_integrity()

        assert report.untracked_files == ("images/dropped_in.spm",)

    def test_an_untracked_file_is_not_imported(self, repo: SqliteProjectRepository) -> None:
        """Importing it would guess that it was meant to be in the project, and
        invent a modality for it."""
        put_scan(repo.root, "dropped_in.spm")

        repo.check_integrity()

        assert repo.list_images() == []

    def test_it_looks_into_subdirectories(self, repo: SqliteProjectRepository) -> None:
        (repo.root / "images" / "batch_02").mkdir()
        (repo.root / "images" / "batch_02" / "a.spm").write_bytes(b"AFM")

        assert repo.check_integrity().untracked_files == ("images/batch_02/a.spm",)

    def test_both_directions_at_once(self, repo: SqliteProjectRepository) -> None:
        record = repo.add_image(put_scan(repo.root, "recorded.spm"), modality=Modality.AFM)
        (repo.root / record.relative_path).unlink()
        put_scan(repo.root, "dropped_in.spm")

        report = repo.check_integrity()

        assert report.missing_files == (record,)
        assert report.untracked_files == ("images/dropped_in.spm",)


class TestTheProjectIsMovable:
    def test_everything_resolves_after_the_directory_moves(
        self, project: Path, tmp_path: Path
    ) -> None:
        """ADR-0003's compliance clause, executed: *"an integration test opens a
        project moved to a new directory and asserts everything resolves"*. It
        is the one property every relative path in the design exists for."""
        with SqliteProjectRepository.open(project) as repo:
            repo.add_image(put_scan(repo.root), modality=Modality.AFM, pixel_size_nm=1.95)
            before = repo.list_images()

        moved = tmp_path / "somewhere" / "else" / "MyProject"
        moved.parent.mkdir(parents=True)
        shutil.move(str(project), str(moved))

        with SqliteProjectRepository.open(moved) as repo:
            assert repo.list_images() == before
            assert repo.check_integrity().is_clean
            assert (repo.root / before[0].relative_path).is_file()

    def test_a_copy_is_a_project_too(self, project: Path, tmp_path: Path) -> None:
        """`cp -r`, `rsync`, a backup tool — the operator owning their data means
        the copy opens as readily as the original."""
        with SqliteProjectRepository.open(project) as repo:
            repo.add_image(put_scan(repo.root), modality=Modality.AFM)
            before = repo.list_images()

        copy = tmp_path / "MyProject-backup"
        shutil.copytree(project, copy)

        with SqliteProjectRepository.open(copy) as repo:
            assert repo.list_images() == before
            assert repo.check_integrity().is_clean


class TestItSurvivesBeingClosedAndReopened:
    def test_rows_written_in_one_session_are_there_in_the_next(self, project: Path) -> None:
        with SqliteProjectRepository.open(project) as repo:
            written = repo.add_image(put_scan(repo.root), modality=Modality.TEM)

        with SqliteProjectRepository.open(project) as repo:
            assert repo.list_images() == [written]

    def test_deleting_the_cache_costs_nothing(self, project: Path) -> None:
        """`cache/` is disposable by contract (`ProjectFormat.md` §1), so this is
        the test that says so out loud."""
        with SqliteProjectRepository.open(project) as repo:
            written = repo.add_image(put_scan(repo.root), modality=Modality.AFM)
            (repo.root / "cache" / "thumbnail.png").write_bytes(b"PNG")

        shutil.rmtree(project / "cache")

        with SqliteProjectRepository.open(project) as repo:
            assert repo.list_images() == [written]
            assert repo.check_integrity().is_clean
