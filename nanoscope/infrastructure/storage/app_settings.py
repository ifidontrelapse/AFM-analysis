"""The preferences that follow the operator rather than the project (M4-T10).

A JSON file under `$XDG_CONFIG_HOME/nanoscope/settings.json`, which on a Linux
desktop is `~/.config/nanoscope/settings.json` — the place a Linux user already
knows to look, and the one a backup tool already collects. This is a Linux
desktop application (ADR-0002), so the XDG basedir spec is *the* convention and
not one of several.

Written by replacement, not by mutation: a temporary file in the same directory
and an atomic `rename`. A settings file truncated by a crash mid-write is a
preferences reset for someone who did nothing wrong, and `os.replace` is one
line (ADR-0047).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_DIRECTORY = "nanoscope"
_FILE = "settings.json"


def default_settings_path() -> Path:
    """Where this machine keeps them, honouring `XDG_CONFIG_HOME` if it is set."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / _DIRECTORY / _FILE


class JsonSettings:
    """Application-scope settings, in one small JSON file.

    Read from disk on every access rather than cached: another window of this
    application, or the operator with a text editor, is a second writer, and a
    cache would mean the last process to exit wins. The file is a few hundred
    bytes.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_settings_path()

    def get_setting(self, key: str, default: object = None) -> object:
        return self.all_settings().get(key, default)

    def set_setting(self, key: str, value: object) -> None:
        settings = self.all_settings()
        settings[key] = value
        self._write(settings)

    def all_settings(self) -> dict[str, object]:
        """Everything stored, or `{}` when there is no file yet.

        A malformed file is treated as empty rather than raised over: this is a
        preferences file, the application must still start, and refusing to open
        because somebody hand-edited a comma is a worse failure than defaults.
        The file is not deleted — an operator who broke it can still see it.
        """
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, settings: dict[str, object]) -> None:
        """Replace the file atomically, creating its directory if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)
