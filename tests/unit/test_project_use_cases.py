"""The two use cases, against a repository that is not SQLite (M4-T04, ADR-0041).

The fake here is doing two jobs. The obvious one is speed and control — a
failure on the third of five files is one line to arrange. The other is the
reason the port exists at all: if `ProjectRepository` can only ever be satisfied
by the class it was extracted from, it is a type alias with extra steps. This
file is the second implementation, and mypy checks it structurally.

What is *not* tested here: that files land on disk. That is the adapter's
promise, and `tests/integration/` holds it to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanoscope.application.use_cases import import_images, open_project
from nanoscope.core.entities import ImageRecord, IntegrityReport
from nanoscope.core.errors import MissingFileError, NanoscopeError
from nanoscope.core.ports import ProjectRepository
from nanoscope.core.values import Modality


class FakeRepository:
    """A `ProjectRepository` with a dict where the database goes."""

    def __init__(self, name: str = "MyProject", *, refuse: set[str] | None = None) -> None:
        self.name = name
        self.images: list[ImageRecord] = []
        self.integrity = IntegrityReport()
        self.closed = False
        #: Sources this repository will not accept, by name.
        self._refuse = refuse or set()

    def import_image(
        self,
        source: Path | str,
        *,
        modality: Modality,
        display_name: str | None = None,
        pixel_size_nm: float | None = None,
    ) -> ImageRecord:
        if Path(source).name in self._refuse:
            raise MissingFileError(f"no file to import at {source}")
        record = ImageRecord(
            id=len(self.images) + 1,
            relative_path=f"images/{Path(source).name}",
            display_name=display_name or Path(source).name,
            modality=modality,
            sha256="0" * 64,
            pixel_size_nm=pixel_size_nm,
            imported_utc="2026-08-12T00:00:00+00:00",
        )
        self.images.append(record)
        return record

    def add_image(
        self,
        relative_path: str,
        *,
        modality: Modality,
        display_name: str | None = None,
        pixel_size_nm: float | None = None,
    ) -> ImageRecord:
        return self.import_image(
            relative_path,
            modality=modality,
            display_name=display_name,
            pixel_size_nm=pixel_size_nm,
        )

    def get_image(self, image_id: int) -> ImageRecord:
        return self.images[image_id - 1]

    def list_images(self) -> list[ImageRecord]:
        return list(self.images)

    def remove_image(self, image_id: int) -> None:
        del self.images[image_id - 1]

    def check_integrity(self) -> IntegrityReport:
        return self.integrity

    def close(self) -> None:
        self.closed = True


def test_the_fake_satisfies_the_port() -> None:
    """The assertion this whole file rests on. mypy checks it structurally; this
    says so at run time, and it is what makes the port more than a type alias
    for the one class that exists."""
    repository: ProjectRepository = FakeRepository()

    assert repository.list_images() == []


class TestOpeningAProject:
    def test_it_carries_the_name_and_the_images(self) -> None:
        repo = FakeRepository("Nanoparticles 2026")
        repo.import_image("a.spm", modality=Modality.AFM)

        opened = open_project(repo)

        assert opened.name == "Nanoparticles 2026"
        assert opened.images == tuple(repo.images)

    def test_the_integrity_report_comes_with_it(self) -> None:
        """ADR-0040 ended on an obligation — a report nobody reads is a report
        that did nothing — and this is where it is discharged. The report is
        *handed over*, not made available on request."""
        repo = FakeRepository()
        missing = repo.import_image("gone.spm", modality=Modality.AFM)
        repo.integrity = IntegrityReport(missing_files=(missing,))

        opened = open_project(repo)

        assert opened.integrity.missing_files == (missing,)
        assert not opened.integrity.is_clean

    def test_a_healthy_project_says_so(self) -> None:
        assert open_project(FakeRepository()).integrity.is_clean


class TestImportingABatch:
    def test_every_file_is_imported(self) -> None:
        repo = FakeRepository()

        report = import_images(repo, ["a.spm", "b.spm"], modality=Modality.AFM)

        assert [record.display_name for record in report.imported] == ["a.spm", "b.spm"]
        assert report.is_complete

    def test_one_bad_file_does_not_lose_the_others(self) -> None:
        """The whole reason this use case exists. Thirty-nine good scans are not
        thrown away because the fortieth was a partial download."""
        repo = FakeRepository(refuse={"broken.spm"})

        report = import_images(repo, ["a.spm", "broken.spm", "c.spm"], modality=Modality.AFM)

        assert [record.display_name for record in report.imported] == ["a.spm", "c.spm"]
        assert not report.is_complete

    def test_a_failure_names_the_file_and_the_reason(self) -> None:
        """A report that says "2 of 3" and nothing else makes the operator open
        every file to find out which."""
        repo = FakeRepository(refuse={"broken.spm"})

        report = import_images(repo, ["a.spm", "broken.spm"], modality=Modality.AFM)

        assert len(report.failed) == 1
        assert report.failed[0].source == "broken.spm"
        assert "broken.spm" in report.failed[0].reason

    def test_the_scale_is_passed_through_and_may_be_unknown(self) -> None:
        repo = FakeRepository()

        report = import_images(repo, ["a.spm"], modality=Modality.AFM, pixel_size_nm=1.95)
        unknown = import_images(repo, ["b.npy"], modality=Modality.AFM)

        assert report.imported[0].pixel_size_nm == 1.95
        assert unknown.imported[0].pixel_size_nm is None

    def test_an_empty_batch_is_an_empty_report(self) -> None:
        report = import_images(FakeRepository(), [], modality=Modality.AFM)

        assert report.imported == ()
        assert report.is_complete

    def test_a_bug_is_not_swallowed_as_a_bad_file(self) -> None:
        """Only `NanoscopeError` is caught. A `TypeError` from our own code is a
        bug, and a bug that keeps going for another thirty-nine files is worse."""

        class Broken(FakeRepository):
            def import_image(self, source: Path | str, **kwargs: object) -> ImageRecord:
                raise TypeError("a bug, not a bad file")

        with pytest.raises(TypeError):
            import_images(Broken(), ["a.spm"], modality=Modality.AFM)

    def test_a_library_refusal_is_data_not_a_crash(self) -> None:
        """The other side of the same line: what this library says no to is
        about the operator's file, and belongs in the report."""

        class Refusing(FakeRepository):
            def import_image(self, source: Path | str, **kwargs: object) -> ImageRecord:
                raise NanoscopeError("the library said no")

        report = import_images(Refusing(), ["a.spm"], modality=Modality.AFM)

        assert report.failed[0].reason == "the library said no"
