"""The one place that constructs things (M5-T01, ADR-0052).

PROJECT_RULES §2.7: *"One composition root. All wiring happens in `app/`.
Nothing else constructs infrastructure objects."* M4 built eight things that
need constructing, and until now every caller — every test — built them by hand.
That is fine in a test and wrong in an application: it is how two places end up
deciding where the settings file lives.

**Not a DI framework.** The wiring is a handful of lines, and a framework to hold
a handful of lines is the thing this project keeps deciding not to build
(M2-T08's ports, ADR-0041's use cases, ADR-0046's autosave). What is here is what
is expensive or stateful: the settings store, the device manager, the job runner,
and the project that is open.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Self

from nanoscope.app.logging import attach_project_log, detach_project_log
from nanoscope.application.commands import CommandStack
from nanoscope.application.jobs import JobRunner
from nanoscope.application.settings import Settings
from nanoscope.application.use_cases import open_project
from nanoscope.core.entities.project import OpenedProject
from nanoscope.core.values import DeviceKind
from nanoscope.infrastructure.device import DeviceManager
from nanoscope.infrastructure.storage import JsonSettings, SqliteProjectRepository

logger = logging.getLogger(__name__)

#: The application-scope key holding the operator's device preference. Named
#: here rather than in `DeviceManager`, because *which* setting drives a policy
#: is a wiring decision and the manager should not know a settings store exists.
DEVICE_SETTING = "device"


class Nanoscope:
    """Everything the application is made of, constructed once.

    Use it as a context manager, or call `close`. A project is opened *into* it
    — `open(path)` — because the open project is the one piece of state every
    layer above asks about, and passing it around by hand is what a composition
    root exists to stop.
    """

    def __init__(
        self,
        settings_path: Path | str | None = None,
        devices: DeviceManager | None = None,
        max_workers: int = 2,
    ) -> None:
        self.application_settings = JsonSettings(settings_path)
        self.devices = devices or DeviceManager()
        self.jobs = JobRunner(max_workers=max_workers)
        self.commands = CommandStack()
        self.repository: SqliteProjectRepository | None = None

    # ── The open project ──────────────────────────────────────────────────────

    def open(self, project_dir: Path | str) -> OpenedProject:
        """Open a project, attach its log, and return what is in it.

        Closes whatever was open first: one application, one project, and a log
        that does not continue in another project's file (ADR-0051).

        Returns:
            Its name, its images, and the integrity report — which somebody has
            to *read*, and which is why ADR-0040 ended on an obligation.

        Raises:
            ProjectFormatError: the directory is not a project, or declares a
                version this application does not understand.
        """
        self.close_project()

        self.repository = SqliteProjectRepository.open(project_dir)
        attach_project_log(self.repository.root)
        opened = open_project(self.repository)

        logger.info(
            "opened project %r with %d image(s)",
            opened.name,
            len(opened.images),
            extra={"project": str(self.repository.root)},
        )
        if not opened.integrity.is_clean:
            logger.warning(
                "project %r has %d missing file(s) and %d untracked file(s)",
                opened.name,
                len(opened.integrity.missing_files),
                len(opened.integrity.untracked_files),
            )
        return opened

    def close_project(self) -> None:
        """Close the open project, if there is one. Idempotent."""
        if self.repository is None:
            return
        detach_project_log()
        self.repository.close()
        self.repository = None
        #: The history goes with the project: undo is a session, and a stack
        #: whose commands refer to another project's rows is worse than none
        #: (ADR-0045).
        self.commands.clear()

    @property
    def settings(self) -> Settings:
        """The merged view — the project's answers first, then the operator's.

        Rebuilt per access rather than cached, because the project half changes
        when a project opens or closes, and a stale `Settings` would answer for
        a project that is no longer open (ADR-0047).
        """
        return Settings(self.application_settings, self.repository)

    # ── Policy that needs two components to answer ────────────────────────────

    def select_device(self) -> str:
        """Resolve the device, honouring the operator's stored preference.

        The wiring ADR-0004 implies and neither component can do alone: the
        manager knows what exists and the settings know what was asked for. An
        unreadable preference is ignored with a warning rather than raising —
        a typo in a settings file must not stop an analysis from running.
        """
        preferred = self.settings.get(DEVICE_SETTING)
        kind: DeviceKind | None = None
        if preferred is not None:
            try:
                kind = DeviceKind(str(preferred))
            except ValueError:
                logger.warning(
                    "ignoring unknown %s setting %r; the choice is one of %s",
                    DEVICE_SETTING,
                    preferred,
                    ", ".join(DeviceKind),
                )

        selection = self.devices.select(kind)
        if selection.is_fallback:
            logger.warning("%s", selection.reason)
        return selection.device.torch_name

    # ── Lifetime ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Shut everything down: the project, then the jobs.

        In that order: a job holding the repository must not find it closed
        underneath itself, and `shutdown` waits for what is running (ADR-0043).
        """
        self.jobs.shutdown()
        self.close_project()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
