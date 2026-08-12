"""Where log records end up, and in what shape (M4-T14, ADR-0051).

`tests/unit/test_logging.py` (M2-T11) asserts that library modules *emit* rather
than print. This one is the other half: that somebody configured a destination,
that a record arrives as one JSON object, and that configuring twice does not
write everything twice — which is what a restarted GUI, a reopened project and a
test suite all do.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from nanoscope.app.logging import attach_project_log, configure_logging, detach_project_log
from nanoscope.infrastructure.logging import JsonLinesFormatter, application_log_path


@pytest.fixture(autouse=True)
def clean_root() -> Iterator[None]:
    """Leave the root logger as it was found: these tests attach to it."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


def records_in(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TestTheRecordShape:
    def test_a_record_is_one_json_object_per_line(self, tmp_path: Path) -> None:
        path = configure_logging(path=tmp_path / "app.log")

        logging.getLogger("nanoscope.test").info("imported %d files", 12)

        [record] = records_in(path)
        assert record["message"] == "imported 12 files"
        assert record["level"] == "INFO"
        assert record["logger"] == "nanoscope.test"
        assert record["time"].endswith("+00:00")

    def test_lazy_arguments_are_applied_not_left_as_a_template(self, tmp_path: Path) -> None:
        """ADR-0013 requires `%`-style arguments so nothing is rendered when the
        level is off. The record still has to carry the finished sentence."""
        path = configure_logging(path=tmp_path / "app.log")

        logging.getLogger("x").warning("device fallback: %s", "no CUDA")

        assert records_in(path)[0]["message"] == "device fallback: no CUDA"

    def test_an_exception_is_kept(self, tmp_path: Path) -> None:
        path = configure_logging(path=tmp_path / "app.log")

        try:
            raise ValueError("the file was not a file")
        except ValueError:
            logging.getLogger("x").exception("import failed")

        record = records_in(path)[0]
        assert "ValueError: the file was not a file" in str(record["exception"])

    def test_extra_fields_survive_as_fields(self, tmp_path: Path) -> None:
        """The point of structured logging: a GUI panel reads `image_id`, it
        does not regex it back out of a sentence."""
        path = configure_logging(path=tmp_path / "app.log")

        logging.getLogger("x").info("analysed", extra={"image_id": 7, "detector": "log"})

        record = records_in(path)[0]
        assert record["image_id"] == 7
        assert record["detector"] == "log"

    def test_an_unserialisable_extra_does_not_lose_the_record(self) -> None:
        """A log line must never fail because somebody logged an object: the
        record is the thing being kept, and a repr beats an exception."""
        record = logging.LogRecord("x", logging.INFO, "f", 1, "msg", None, None)
        record.thing = object()  # type: ignore[attr-defined]

        payload = json.loads(JsonLinesFormatter().format(record))

        assert payload["thing"].startswith("<object object")


class TestWhereItGoes:
    def test_the_application_log_is_state_not_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A log is generated data a user does not edit, and XDG has a directory
        for exactly that — `~/.local/state`, not `~/.config`."""
        monkeypatch.setenv("XDG_STATE_HOME", "/tmp/state")

        assert application_log_path() == Path("/tmp/state/nanoscope/nanoscope.log")

    def test_a_project_gets_its_own_log(self, tmp_path: Path) -> None:
        configure_logging(path=tmp_path / "app.log")
        project = tmp_path / "P"

        written = attach_project_log(project)
        logging.getLogger("x").info("opened")

        assert written == project / "logs" / "nanoscope.log"
        assert records_in(written)[0]["message"] == "opened"

    def test_the_application_log_keeps_everything_too(self, tmp_path: Path) -> None:
        """Two destinations, two questions: what did the application do, and
        what happened to this work."""
        app_log = configure_logging(path=tmp_path / "app.log")
        attach_project_log(tmp_path / "P")

        logging.getLogger("x").info("opened")

        assert len(records_in(app_log)) == 1

    def test_opening_another_project_stops_writing_into_the_first(self, tmp_path: Path) -> None:
        """One project's log must not continue in another's file."""
        configure_logging(path=tmp_path / "app.log")
        first = attach_project_log(tmp_path / "First")
        second = attach_project_log(tmp_path / "Second")

        logging.getLogger("x").info("after switching")

        assert records_in(first) == []
        assert records_in(second)[0]["message"] == "after switching"

    def test_closing_a_project_detaches_its_log(self, tmp_path: Path) -> None:
        configure_logging(path=tmp_path / "app.log")
        project_log = attach_project_log(tmp_path / "P")

        detach_project_log()
        logging.getLogger("x").info("after closing")

        assert records_in(project_log) == []

    def test_configuring_twice_does_not_duplicate_records(self, tmp_path: Path) -> None:
        """A restarted GUI, a reopened project and a test suite all do this."""
        path = configure_logging(path=tmp_path / "app.log")
        configure_logging(path=tmp_path / "app.log")

        logging.getLogger("x").info("once")

        assert len(records_in(path)) == 1

    def test_it_rotates(self, tmp_path: Path) -> None:
        """An unbounded log on a laptop is a disk-full bug that arrives months
        later. This asserts the policy is set, not that 5 MB is right."""
        configure_logging(path=tmp_path / "app.log")

        handler = next(h for h in logging.getLogger().handlers if h.name == "nanoscope:application")

        assert handler.maxBytes > 0  # type: ignore[attr-defined]
        assert handler.backupCount > 0  # type: ignore[attr-defined]
