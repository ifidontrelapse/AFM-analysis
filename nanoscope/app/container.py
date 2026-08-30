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
from nanoscope.application.settings import DEVICE_SETTING, Settings
from nanoscope.application.use_cases import open_project
from nanoscope.core.entities.model import ModelTask
from nanoscope.core.entities.project import OpenedProject
from nanoscope.core.ports import ProjectRepository
from nanoscope.core.values import DeviceKind
from nanoscope.infrastructure.device import DeviceManager
from nanoscope.infrastructure.models.registry import resolve
from nanoscope.infrastructure.storage import JsonSettings, SqliteProjectRepository

logger = logging.getLogger(__name__)

#: Re-exported: the key lives in `application.settings` beside the other two a
#: settings dialog writes (M5-T09), and *which* setting drives a policy is still
#: a wiring decision the `DeviceManager` knows nothing about.
__all__ = ["DEVICE_SETTING", "Nanoscope"]


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
        #: Typed as the **port**: the GUI may not import `infrastructure`
        #: (Architecture §3.2), and a container that hands out a concrete
        #: adapter is a container that makes the rule unenforceable (M5-T04).
        self.repository: ProjectRepository | None = None
        #: Built on demand and kept: loading weights is seconds of disk and GPU,
        #: and a second run must not pay it again (M6-T04).
        self._predictor: object | None = None

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

    def create(self, project_dir: Path | str, name: str) -> OpenedProject:
        """Make a new project and open it.

        The last unwired half of M4-T04: `SqliteProjectRepository.create` has
        existed since that task and, until now, was called only by tests — so an
        operator could open a project and never make one, and `Import Images…`
        stays disabled until a project is open. The application could not be
        started from an empty machine.

        Creating and opening are one call because they are one intention. It
        costs opening the database twice — `create` hands back an open
        repository, which this closes and `open` reopens — which is one SQLite
        handshake on a directory that was just made, against duplicating every
        line of `open`: closing the previous project, attaching the log,
        reading the integrity report.

        Args:
            project_dir: where to put it. Must not exist, or must be empty.
            name: the display name, which the directory's name need not match.

        Returns:
            What `open` returns for it: an empty project, and a clean report.

        Raises:
            InvalidParameterError: there is already something in that directory.
                Writing a manifest into somebody's folder turns it into a
                project (M4-T04).
        """
        SqliteProjectRepository.create(project_dir, name).close()
        logger.info("created project %r", name, extra={"project": str(Path(project_dir))})
        return self.open(project_dir)

    def close_project(self) -> None:
        """Close the open project, if there is one. Idempotent."""
        if self.repository is None:
            return
        detach_project_log()
        self.repository.close()
        self.repository = None
        #: A predictor belongs to the project whose model produced it.
        self._predictor = None
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

    def segmentation_predictor(self) -> object | None:
        """The predictor a `segment` run needs, built once (M6-T04, ADR-0064).

        The last unwired half of ADR-0050: the registry hands back **factories**,
        and until now nothing called one. This looks up a segmentation model in
        the open project, resolves its factory, and constructs it with the device
        `select_device` chose (ADR-0049).

        **Constructing loads weights**, so this is called inside a job and never
        to answer *"can this project segment?"* — that question is answered by a
        registered model, which is why the registry was made cheap.

        Returns:
            The predictor, or `None` when the project has no segmentation model.

        Raises:
            UnsupportedRequestError: a model is registered for a framework this
                version cannot load (ADR-0050's own message).
        """
        if self._predictor is not None:
            return self._predictor
        if self.repository is None:
            return None

        models = [m for m in self.repository.list_models() if m.task is ModelTask.SEGMENT]
        if not models:
            return None

        descriptor = models[0]
        factory = resolve(descriptor)
        device = self.devices.select(self._preferred_kind()).device
        logger.info("loading segmentation model %r on %s", descriptor.model_id, device.torch_name)
        self._predictor = factory(self.repository.path_of_model(descriptor), device)
        return self._predictor

    # ── Policy that needs two components to answer ────────────────────────────

    def _preferred_kind(self) -> DeviceKind | None:
        """The stored device preference, or `None` for "let the policy decide"."""
        preferred = self.settings.get(DEVICE_SETTING)
        if preferred is None:
            return None
        try:
            return DeviceKind(str(preferred))
        except ValueError:
            return None

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
