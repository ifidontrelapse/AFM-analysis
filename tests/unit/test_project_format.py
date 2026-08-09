"""The project format's compatibility matrix, executed (M4-T01, ADR-0038).

`docs/ProjectFormat.md` is the contract. These tests are the reason it stays
true: every row of its matrix — same version, older, newer, no manifest, an
unparseable one, a manifest missing a field — is a case here, and each refusal
is checked for naming the path or the versions, because a contract that refuses
without saying what it read is not usable by the operator holding the directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoscope.core.errors import NanoscopeError, ProjectFormatError
from nanoscope.infrastructure.storage import (
    DIRECTORIES,
    FORMAT_VERSION,
    MANIFEST_NAME,
    ProjectManifest,
    check_compatible,
    new_manifest,
    open_manifest,
    read_manifest,
    write_manifest,
)


def _project(tmp_path: Path, **overrides: object) -> Path:
    """A directory with a manifest in it, optionally malformed."""
    data = {
        "name": "MyProject",
        "format_version": FORMAT_VERSION,
        "created_utc": "2026-08-09T12:00:00+00:00",
    }
    data.update(overrides)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


class TestTheManifestRoundTrips:
    def test_what_is_written_is_what_is_read(self, tmp_path: Path) -> None:
        written = new_manifest("MyProject")
        write_manifest(tmp_path, written)

        assert read_manifest(tmp_path) == written

    def test_a_new_manifest_carries_this_version(self) -> None:
        assert new_manifest("x").format_version == FORMAT_VERSION

    def test_the_manifest_is_readable_by_anything_that_reads_json(self, tmp_path: Path) -> None:
        """The operator owning their data (ADR-0003) means `cat project.json`
        has to work, so the file is indented JSON with no framing of ours."""
        write_manifest(tmp_path, new_manifest("MyProject"))

        data = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))

        assert data["name"] == "MyProject"
        assert data["format_version"] == FORMAT_VERSION

    def test_an_unknown_field_survives_a_rewrite(self, tmp_path: Path) -> None:
        """The rule that makes an additive change additive. Without it, an older
        application rewriting a newer project's manifest deletes what it did not
        recognise — data loss with no error attached."""
        (tmp_path / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "name": "MyProject",
                    "format_version": FORMAT_VERSION,
                    "created_utc": "2026-08-09T12:00:00+00:00",
                    "written_by_a_later_version": {"colour": "blue"},
                }
            ),
            encoding="utf-8",
        )

        write_manifest(tmp_path, read_manifest(tmp_path))

        data = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert data["written_by_a_later_version"] == {"colour": "blue"}


class TestTheCompatibilityMatrix:
    def test_the_same_version_opens(self, tmp_path: Path) -> None:
        assert open_manifest(_project(tmp_path)).format_version == FORMAT_VERSION

    def test_an_older_version_opens(self, tmp_path: Path) -> None:
        """Accepted, not migrated — migrations are M4-T02's. What this task
        fixes is that an old project is not refused for being old."""
        assert open_manifest(_project(tmp_path, format_version=FORMAT_VERSION - 1))

    def test_a_newer_version_is_refused_and_names_both_versions(self, tmp_path: Path) -> None:
        """A forward migration cannot be written by the past."""
        with pytest.raises(ProjectFormatError) as excinfo:
            open_manifest(_project(tmp_path, format_version=FORMAT_VERSION + 1))

        message = str(excinfo.value)
        assert str(FORMAT_VERSION + 1) in message
        assert str(FORMAT_VERSION) in message

    def test_a_directory_without_a_manifest_is_not_a_project(self, tmp_path: Path) -> None:
        """Never guessed from the presence of `images/`: a directory says it is
        a project by carrying a manifest, and nothing else counts."""
        for name in DIRECTORIES:
            (tmp_path / name).mkdir()

        with pytest.raises(ProjectFormatError, match="not a project directory"):
            read_manifest(tmp_path)

    def test_a_missing_directory_is_the_same_refusal(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectFormatError, match="not a project directory"):
            read_manifest(tmp_path / "nowhere")

    def test_an_unparseable_manifest_is_refused_by_path(self, tmp_path: Path) -> None:
        (tmp_path / MANIFEST_NAME).write_text("{not json", encoding="utf-8")

        with pytest.raises(ProjectFormatError, match=MANIFEST_NAME):
            read_manifest(tmp_path)

    def test_a_manifest_that_is_not_an_object_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / MANIFEST_NAME).write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ProjectFormatError, match="JSON object"):
            read_manifest(tmp_path)

    @pytest.mark.parametrize("field", ["name", "format_version", "created_utc"])
    def test_a_missing_field_is_named(self, tmp_path: Path, field: str) -> None:
        data = {
            "name": "MyProject",
            "format_version": FORMAT_VERSION,
            "created_utc": "2026-08-09T12:00:00+00:00",
        }
        del data[field]
        (tmp_path / MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ProjectFormatError, match=field):
            read_manifest(tmp_path)

    @pytest.mark.parametrize(
        "version", ["1", 1.0, True, None], ids=["str", "float", "bool", "null"]
    )
    def test_a_version_that_is_not_an_integer_is_refused(
        self, tmp_path: Path, version: object
    ) -> None:
        """`"2"` compares as neither newer nor older, and `True` is an `int` in
        Python — both would sail past the version check as "compatible"."""
        with pytest.raises(ProjectFormatError, match="integer"):
            read_manifest(_project(tmp_path, format_version=version))


class TestTheRefusalIsCatchableAsOurs:
    def test_it_is_a_nanoscope_error(self, tmp_path: Path) -> None:
        with pytest.raises(NanoscopeError):
            read_manifest(tmp_path)

    def test_it_is_a_value_error_for_callers_that_predate_the_taxonomy(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError):
            read_manifest(tmp_path)


def test_check_compatible_is_usable_on_its_own() -> None:
    """The read and the check are separate so a repair tool can read a manifest
    it is not allowed to open."""
    check_compatible(ProjectManifest("x", FORMAT_VERSION, "2026-08-09T12:00:00+00:00"))

    with pytest.raises(ProjectFormatError):
        check_compatible(ProjectManifest("x", FORMAT_VERSION + 1, "2026-08-09T12:00:00+00:00"))
