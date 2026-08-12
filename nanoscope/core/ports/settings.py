"""Where a preference is kept, whoever keeps it (M4-T10).

Two stores satisfy this: a JSON file in the operator's config directory for
choices that follow *them*, and the project's database for choices that belong
to *the work*. The application merges them and never learns which is which
beyond that (ADR-0047).

Values are whatever JSON can carry — a string, a number, a boolean, a list, a
mapping. Nothing here validates what a key means: a setting's meaning belongs
to whoever reads it, and a store that knew every key would have to be edited
by every feature.
"""

from __future__ import annotations

from typing import Protocol


class SettingsStore(Protocol):
    """A place preferences are read from and written to."""

    def get_setting(self, key: str, default: object = None) -> object:
        """The stored value, or `default` when nothing was ever stored."""
        ...

    def set_setting(self, key: str, value: object) -> None:
        """Store `value` under `key`, replacing whatever was there."""
        ...

    def all_settings(self) -> dict[str, object]:
        """Everything this store holds, for a settings dialog to show at once."""
        ...
