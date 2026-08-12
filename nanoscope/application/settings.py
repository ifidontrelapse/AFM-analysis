"""One question, two places to look for the answer (M4-T10, ADR-0047).

A preference is either about the **operator** — which colormap they like, where
the last project was — or about **the work**: the pixel scale this project's
scans were taken at, the detector its measurements were made with. The first
follows them to every project; the second belongs to one and should travel with
its directory.

So there are two stores, and this is the only thing that knows both:

- **reads merge**, project first, because a project that states something is
  stating it about itself;
- **writes name their scope**, because "save this preference" without saying
  where is a question, not an instruction — and guessing it wrong either leaks
  one project's choice into every other, or hides a global preference inside
  one directory.
"""

from __future__ import annotations

from enum import StrEnum

from nanoscope.core.ports.settings import SettingsStore


class Scope(StrEnum):
    """Which of the two stores a write is aimed at."""

    #: Follows the operator: every project sees it.
    APPLICATION = "application"
    #: Belongs to this project, and travels with its directory.
    PROJECT = "project"


class Settings:
    """The merged view, and the only thing that knows there are two stores.

    A project store is optional: before a project is open there is exactly one
    place a preference can be, and `Settings(app)` is that situation rather than
    a special case to handle.
    """

    def __init__(self, application: SettingsStore, project: SettingsStore | None = None) -> None:
        self._application = application
        self._project = project

    def get(self, key: str, default: object = None) -> object:
        """The project's answer if it has one, otherwise the operator's.

        `None` stored in a project is an answer, not an absence — so the project
        is asked whether it *has* the key rather than what it returns for it.
        """
        if self._project is not None and key in self._project.all_settings():
            return self._project.get_setting(key, default)
        return self._application.get_setting(key, default)

    def set(self, key: str, value: object, scope: Scope = Scope.APPLICATION) -> None:
        """Store a preference, in the scope the caller names.

        The default is `APPLICATION`, because a preference expressed without a
        project in mind is about the person expressing it.

        Raises:
            ValueError: `PROJECT` was asked for and no project is open. Silently
                writing it to the application scope would put one project's
                choice in front of every other project the operator opens.
        """
        if scope is Scope.PROJECT:
            if self._project is None:
                raise ValueError(f"cannot store {key!r} in the project scope: no project is open")
            self._project.set_setting(key, value)
            return
        self._application.set_setting(key, value)

    def all(self) -> dict[str, object]:
        """Everything visible right now, merged the way `get` resolves it."""
        merged = dict(self._application.all_settings())
        if self._project is not None:
            merged.update(self._project.all_settings())
        return merged

    def scope_of(self, key: str) -> Scope | None:
        """Which store an answer would come from, or `None` if nothing has one.

        What a settings dialog needs to say "this project overrides your
        default" instead of showing a value with no explanation.
        """
        if self._project is not None and key in self._project.all_settings():
            return Scope.PROJECT
        if key in self._application.all_settings():
            return Scope.APPLICATION
        return None
