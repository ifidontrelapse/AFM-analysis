"""A project's own preferences, in the project (M4-T10, ADR-0047).

The project-scope half of the settings story, against a real database: what a
project states travels with its directory, survives the session, and beats the
operator's default when the two disagree.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import revert_to

from nanoscope.application.settings import Scope, Settings
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import JsonSettings, SqliteProjectRepository


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SqliteProjectRepository]:
    with SqliteProjectRepository.create(tmp_path / "P", "P") as repository:
        yield repository


class TestAProjectStatesItsOwn:
    def test_what_it_states_comes_back(self, repo: SqliteProjectRepository) -> None:
        repo.set_setting("detector", "log")

        assert repo.get_setting("detector") == "log"

    def test_types_survive_the_database(self, repo: SqliteProjectRepository) -> None:
        repo.set_setting("tiling", value=False)
        repo.set_setting("threshold", 0.35)
        repo.set_setting("classes", ["particle", "contaminant"])

        assert repo.get_setting("tiling") is False
        assert repo.get_setting("threshold") == 0.35
        assert repo.get_setting("classes") == ["particle", "contaminant"]

    def test_stating_it_twice_replaces_it(self, repo: SqliteProjectRepository) -> None:
        repo.set_setting("detector", "log")
        repo.set_setting("detector", "yolo")

        assert repo.get_setting("detector") == "yolo"
        assert repo.all_settings() == {"detector": "yolo"}

    def test_an_unstated_preference_gives_the_default(self, repo: SqliteProjectRepository) -> None:
        assert repo.get_setting("nothing", "fallback") == "fallback"

    def test_they_travel_with_the_directory(self, tmp_path: Path) -> None:
        """A project's settings are about the work, so they are in the work —
        copy the directory and the choices come along."""
        with SqliteProjectRepository.create(tmp_path / "Q", "Q") as repo:
            repo.set_setting("detector", "log")

        with SqliteProjectRepository.open(tmp_path / "Q") as repo:
            assert repo.get_setting("detector") == "log"


class TestTheTwoScopesTogether:
    def test_the_project_wins(self, tmp_path: Path, repo: SqliteProjectRepository) -> None:
        application = JsonSettings(tmp_path / "settings.json")
        application.set_setting("detector", "yolo")
        repo.set_setting("detector", "log")

        settings = Settings(application, repo)

        assert settings.get("detector") == "log"
        assert settings.scope_of("detector") is Scope.PROJECT

    def test_the_operator_default_shows_through_where_the_project_is_silent(
        self, tmp_path: Path, repo: SqliteProjectRepository
    ) -> None:
        application = JsonSettings(tmp_path / "settings.json")
        application.set_setting("colormap", "afmhot")

        settings = Settings(application, repo)

        assert settings.get("colormap") == "afmhot"
        assert settings.scope_of("colormap") is Scope.APPLICATION

    def test_a_project_write_lands_in_the_project(
        self, tmp_path: Path, repo: SqliteProjectRepository
    ) -> None:
        application = JsonSettings(tmp_path / "settings.json")
        settings = Settings(application, repo)

        settings.set("detector", "log", Scope.PROJECT)

        assert repo.all_settings() == {"detector": "log"}
        assert application.all_settings() == {}


class TestTheMigrationThatBroughtThem:
    def test_a_project_at_v3_gains_the_table_and_keeps_its_rows(self, tmp_path: Path) -> None:
        root = tmp_path / "P"
        with SqliteProjectRepository.create(root, "P") as repo:
            (root / "images" / "a.spm").write_bytes(b"AFM")
            recorded = repo.add_image("images/a.spm", modality=Modality.AFM)
            revert_to(repo._conn, 3)

        with SqliteProjectRepository.open(root) as repo:
            assert repo.list_images() == [recorded]
            repo.set_setting("detector", "log")
            assert repo.all_settings() == {"detector": "log"}
