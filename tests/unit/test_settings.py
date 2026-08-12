"""Two stores, one answer (M4-T10, ADR-0047).

The merged view is the whole point of the module, so most of this file is about
precedence and about the write that refuses to guess. The JSON store is here
too — it is a file, but a small one, and its failure modes (a corrupt file, a
half-written one) are the reason it exists in this shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoscope.application.settings import Scope, Settings
from nanoscope.infrastructure.storage import JsonSettings, default_settings_path


class TestTheApplicationStore:
    def test_what_is_written_is_read_back(self, tmp_path: Path) -> None:
        store = JsonSettings(tmp_path / "settings.json")

        store.set_setting("colormap", "afmhot")

        assert store.get_setting("colormap") == "afmhot"

    def test_types_survive(self, tmp_path: Path) -> None:
        """JSON, not `str()`: a store that returns everything as text makes
        every reader parse it back, and one of them gets it wrong."""
        store = JsonSettings(tmp_path / "settings.json")

        store.set_setting("tiling", value=True)
        store.set_setting("threshold", 0.35)
        store.set_setting("recent", ["/a", "/b"])

        assert store.get_setting("tiling") is True
        assert store.get_setting("threshold") == 0.35
        assert store.get_setting("recent") == ["/a", "/b"]

    def test_an_absent_key_gives_the_default(self, tmp_path: Path) -> None:
        assert JsonSettings(tmp_path / "s.json").get_setting("nothing", "fallback") == "fallback"

    def test_the_file_is_readable_by_anything_that_reads_json(self, tmp_path: Path) -> None:
        """`~/.config/nanoscope/settings.json` is where a Linux operator already
        knows to look, so what they find has to be legible."""
        store = JsonSettings(tmp_path / "settings.json")
        store.set_setting("colormap", "afmhot")

        assert json.loads((tmp_path / "settings.json").read_text()) == {"colormap": "afmhot"}

    def test_a_corrupt_file_reads_as_empty_rather_than_refusing_to_start(
        self, tmp_path: Path
    ) -> None:
        """A preferences file somebody hand-edited must not stop the application
        from opening. The file is left alone, not deleted."""
        path = tmp_path / "settings.json"
        path.write_text("{not json", encoding="utf-8")

        assert JsonSettings(path).all_settings() == {}
        assert path.read_text() == "{not json"

    def test_writing_leaves_no_temporary_behind(self, tmp_path: Path) -> None:
        """Written by replacement: a file truncated by a crash mid-write is a
        preferences reset for somebody who did nothing wrong."""
        store = JsonSettings(tmp_path / "settings.json")

        store.set_setting("a", 1)
        store.set_setting("b", 2)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["settings.json"]

    def test_the_default_location_follows_xdg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/somewhere")

        assert default_settings_path() == Path("/tmp/somewhere/nanoscope/settings.json")

    def test_without_xdg_it_is_the_conventional_place(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        assert default_settings_path() == Path.home() / ".config/nanoscope/settings.json"


class FakeStore:
    """A `SettingsStore` that is a dict."""

    def __init__(self, **values: object) -> None:
        self.values: dict[str, object] = dict(values)

    def get_setting(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def set_setting(self, key: str, value: object) -> None:
        self.values[key] = value

    def all_settings(self) -> dict[str, object]:
        return dict(self.values)


class TestTheMergedView:
    def test_the_project_answers_first(self) -> None:
        """A project that states something is stating it about itself."""
        settings = Settings(FakeStore(colormap="viridis"), FakeStore(colormap="afmhot"))

        assert settings.get("colormap") == "afmhot"

    def test_the_application_answers_when_the_project_does_not(self) -> None:
        settings = Settings(FakeStore(colormap="viridis"), FakeStore())

        assert settings.get("colormap") == "viridis"

    def test_a_null_stored_in_a_project_is_an_answer(self) -> None:
        """`None` is a value somebody chose, not an absence — so the project is
        asked whether it *has* the key, not what it returns for it."""
        settings = Settings(FakeStore(scale=1.95), FakeStore(scale=None))

        assert settings.get("scale") is None

    def test_with_no_project_open_there_is_one_place_to_look(self) -> None:
        settings = Settings(FakeStore(colormap="viridis"))

        assert settings.get("colormap") == "viridis"
        assert settings.get("absent", "fallback") == "fallback"

    def test_everything_visible_is_merged_the_same_way(self) -> None:
        settings = Settings(FakeStore(a=1, b=2), FakeStore(b=3, c=4))

        assert settings.all() == {"a": 1, "b": 3, "c": 4}

    def test_it_can_say_where_an_answer_came_from(self) -> None:
        """What a settings dialog needs to show "this project overrides your
        default" instead of a value with no explanation."""
        settings = Settings(FakeStore(a=1, b=2), FakeStore(b=3))

        assert settings.scope_of("b") is Scope.PROJECT
        assert settings.scope_of("a") is Scope.APPLICATION
        assert settings.scope_of("nothing") is None


class TestWriting:
    def test_a_write_goes_where_the_caller_says(self) -> None:
        application, project = FakeStore(), FakeStore()
        settings = Settings(application, project)

        settings.set("colormap", "afmhot", Scope.PROJECT)
        settings.set("theme", "dark", Scope.APPLICATION)

        assert project.values == {"colormap": "afmhot"}
        assert application.values == {"theme": "dark"}

    def test_the_default_scope_is_the_operator(self) -> None:
        """A preference expressed without a project in mind is about the person
        expressing it."""
        application = FakeStore()
        Settings(application, FakeStore()).set("theme", "dark")

        assert application.values == {"theme": "dark"}

    def test_a_project_write_with_no_project_is_refused(self) -> None:
        """Silently writing it to the application scope would put one project's
        choice in front of every other project the operator opens."""
        settings = Settings(FakeStore())

        with pytest.raises(ValueError, match="no project is open"):
            settings.set("colormap", "afmhot", Scope.PROJECT)
