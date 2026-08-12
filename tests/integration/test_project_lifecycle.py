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
