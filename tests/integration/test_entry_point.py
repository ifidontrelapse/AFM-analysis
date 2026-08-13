"""`nanoscope`, run as a program (M5-T01, ADR-0052).

Two things are checked here, and they fail for different reasons.

The **container** is the one place allowed to construct adapters
(PROJECT_RULES §2.7), so what matters is that it constructs them consistently
and takes them down in the right order — including closing a project's log with
the project.

The **entry point** is a user interface. Its contract is what a person sees and
what the shell gets back: a project summarised, an integrity report *shown* at
last (ADR-0040's closing obligation), and a refusal that is a sentence with a
non-zero exit rather than a traceback.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from nanoscope.app.container import DEVICE_SETTING, Nanoscope
from nanoscope.app.main import OK, REFUSED, main
from nanoscope.application.settings import LOG_LEVEL_SETTING, Scope
from nanoscope.core.values import DeviceKind, Modality
from nanoscope.infrastructure.storage import JsonSettings, SqliteProjectRepository


@pytest.fixture(autouse=True)
def clean_root() -> Iterator[None]:
    """These tests configure logging, which attaches to the root logger."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project with one image in it."""
    root = tmp_path / "Gold on mica"
    with SqliteProjectRepository.create(root, "Gold on mica") as repo:
        (repo.root / "images" / "scan.spm").write_bytes(b"AFM")
        repo.add_image("images/scan.spm", modality=Modality.AFM, pixel_size_nm=1.95)
    return root


def run(argv: list[str], tmp_path: Path) -> int:
    """`main`, with the log pointed somewhere disposable."""
    return main([*argv, "--log-file", str(tmp_path / "state" / "nanoscope.log")])


class TestTheContainer:
    def test_it_constructs_everything(self, tmp_path: Path) -> None:
        with Nanoscope(settings_path=tmp_path / "settings.json") as app:
            assert app.devices.available()
            assert app.jobs is not None
            assert app.commands.can_undo is False
            assert app.repository is None

    def test_opening_a_project_makes_it_the_open_one(self, project: Path) -> None:
        with Nanoscope() as app:
            opened = app.open(project)

            assert opened.name == "Gold on mica"
            assert app.repository is not None
            assert app.settings.get("anything", "fallback") == "fallback"

    def test_the_project_log_goes_with_the_project(self, project: Path) -> None:
        """Attached on open and detached on close, so one project's log never
        continues in another's file (ADR-0051)."""
        with Nanoscope() as app:
            app.open(project)
            logging.getLogger("x").info("while open")
            app.close_project()
            logging.getLogger("x").info("after closing")

        lines = (project / "logs" / "nanoscope.log").read_text(encoding="utf-8").splitlines()
        messages = [json.loads(line)["message"] for line in lines]
        assert "after closing" not in messages

    def test_closing_a_project_clears_the_undo_history(self, project: Path) -> None:
        """Undo is a session (ADR-0045), and commands referring to another
        project's rows are worse than no history at all."""
        with Nanoscope() as app:
            app.open(project)
            app.commands.run(_Noop())
            assert app.commands.can_undo

            app.close_project()

            assert not app.commands.can_undo

    def test_opening_a_second_project_closes_the_first(self, project: Path, tmp_path: Path) -> None:
        second = tmp_path / "Second"
        with SqliteProjectRepository.create(second, "Second"):
            pass

        with Nanoscope() as app:
            app.open(project)
            app.open(second)

            assert app.repository is not None
            assert app.repository.name == "Second"

    def test_the_device_preference_is_honoured(self, project: Path, tmp_path: Path) -> None:
        """The wiring ADR-0004 implies and neither component can do alone: the
        manager knows what exists, the settings know what was asked for."""
        with Nanoscope(settings_path=tmp_path / "settings.json") as app:
            app.open(project)
            app.settings.set(DEVICE_SETTING, str(DeviceKind.CPU), Scope.PROJECT)

            assert app.select_device() == "cpu"

    def test_a_nonsense_device_preference_is_ignored_not_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A typo in a settings file must not stop an analysis from running."""
        with Nanoscope(settings_path=tmp_path / "settings.json") as app:
            app.settings.set(DEVICE_SETTING, "quantum")

            with caplog.at_level(logging.WARNING):
                assert app.select_device()

        assert "quantum" in caplog.text


class TestTheEntryPoint:
    def test_it_reports_a_project(
        self, project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run(["--project", str(project)], tmp_path) == OK

        out = capsys.readouterr().out
        assert "Gold on mica: 1 image(s)" in out
        assert "scan.spm (afm, 1.95 nm/px)" in out

    def test_the_integrity_report_is_shown(
        self, project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ADR-0040 ended on an obligation — *a report nobody reads is a report
        that did nothing* — and this is the first caller that can read it."""
        (project / "images" / "scan.spm").unlink()
        (project / "images" / "dropped_in.spm").write_bytes(b"AFM")

        assert run(["--project", str(project)], tmp_path) == OK

        out = capsys.readouterr().out
        assert "missing: images/scan.spm" in out
        assert "untracked: images/dropped_in.spm" in out

    def test_a_clean_project_says_so(
        self, project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run(["--project", str(project)], tmp_path)

        assert "the index and the files agree" in capsys.readouterr().out

    def test_a_directory_that_is_not_a_project_is_a_sentence_not_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ADR-0030 built the distinction so this surface could use it: our
        errors are messages, and anything else keeps its traceback."""
        assert run(["--project", str(tmp_path / "not-a-project")], tmp_path) == REFUSED

        error = capsys.readouterr().err
        assert "not a project directory" in error
        assert "Traceback" not in error

    def test_it_lists_devices(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert run(["--devices"], tmp_path) == OK

        out = capsys.readouterr().out
        assert "CPU (cpu)" in out
        assert "selected:" in out

    def test_asking_for_a_window_hands_over_to_the_launcher(
        self, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--gui` runs an event loop, so what is asserted here is the
        **handover**: the container and the project reach the launcher, and
        whatever it returns is the exit code.

        The launcher itself is exercised for real — window, event loop and all —
        in `tests/gui/test_launcher.py`. Before M5-T02 this test called `--gui`
        directly, and when the branch stopped being a stub it **hung** rather
        than failed, which is the worse of the two.
        """
        handed: dict[str, object] = {}

        def fake_run(app: object, project_dir: object = None) -> int:
            handed["app"] = app
            handed["project"] = project_dir
            return 0

        monkeypatch.setattr("nanoscope.gui.launcher.run", fake_run)

        assert run(["--gui", "--project", str(project)], tmp_path) == OK
        assert isinstance(handed["app"], Nanoscope)
        assert handed["project"] == str(project)

    def test_asking_for_nothing_says_where_the_log_is(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run([], tmp_path) == OK

        out = capsys.readouterr().out
        assert "--help" in out
        assert "nanoscope.log" in out

    def test_the_log_is_written_where_it_was_told(self, project: Path, tmp_path: Path) -> None:
        run(["--project", str(project)], tmp_path)

        written = (tmp_path / "state" / "nanoscope.log").read_text(encoding="utf-8")
        assert "opened project" in written

    def test_a_bad_flag_is_argparses_business(self, tmp_path: Path) -> None:
        """Usage errors keep argparse's exit code 2; ours start above it."""
        with pytest.raises(SystemExit) as exit_info:
            run(["--nonsense"], tmp_path)

        assert exit_info.value.code == 2


class _Noop:
    """A command that does nothing, for testing the stack's lifetime."""

    label = "nothing"

    def do(self) -> None: ...

    def undo(self) -> None: ...


class TestTheStoredLogLevel:
    """M5-T09: the level a settings dialog writes has to survive a restart, or
    it is a control that works until the operator closes the window."""

    def test_a_stored_level_is_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        JsonSettings().set_setting(LOG_LEVEL_SETTING, logging.WARNING)

        assert run([], tmp_path) == OK

        assert logging.getLogger().level == logging.WARNING

    def test_the_flag_beats_the_preference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Somebody typing `--debug` is answering the question right now; a
        stored preference is an answer they gave once."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        JsonSettings().set_setting(LOG_LEVEL_SETTING, logging.WARNING)

        assert run(["--debug"], tmp_path) == OK

        assert logging.getLogger().level == logging.DEBUG

    def test_an_unreadable_level_is_ignored_rather_than_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same rule as the device preference: a typo in a settings file
        must not stop the application starting (ADR-0052)."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        JsonSettings().set_setting(LOG_LEVEL_SETTING, "very loud")

        assert run([], tmp_path) == OK

        assert logging.getLogger().level == logging.INFO
