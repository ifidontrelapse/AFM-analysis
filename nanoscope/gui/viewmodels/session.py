"""What the session is, and the only thing in `gui/` that asks (M5-T06, ADR-0057).

ADR-0055 §4 declined a viewmodel with a condition attached — *"when M5-T05's
viewer needs the same selection, there will be two consumers and a reason"*. The
viewer shipped, and with it three consumers of one selection: the explorer that
makes it, the viewer that draws it, and the properties panel that describes it.

So this object holds the session and the panels hold none of it:

- **intent comes in as a method call**, `select_image(7)`;
- **state goes out as a signal**, `image_changed(DisplayImage | None)`;
- **no panel connects to another panel** — every one of them subscribes here,
  which is n connections instead of n².

It is a `QObject` and not a plain class because M5-T07 needs it to be: ADR-0043
says a job's listener fires **on the worker thread**, and a queued signal is how
Qt gets that onto the thread the widgets live on. A widget method called from a
worker is a crash that happens later, somewhere else.

It holds no widget and opens no dialog. A viewmodel that pops a `QMessageBox` is
one that cannot be tested without a window — and *how* to ask a question is a
view's decision (ADR-0055's confirmation stays in the panel that asks it).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from nanoscope.application import use_cases
from nanoscope.application.jobs import Job, JobContext, JobState
from nanoscope.application.settings import Scope
from nanoscope.application.use_cases.display import DisplayImage, load_for_display
from nanoscope.core.entities.device import Device
from nanoscope.core.entities.project import ImageRecord, ImportReport, OpenedProject
from nanoscope.core.errors import NanoscopeError
from nanoscope.core.values import Modality

if TYPE_CHECKING:
    from nanoscope.app.container import Nanoscope

logger = logging.getLogger(__name__)


class SessionViewModel(QObject):
    """The open project, the selected image, and what happened to them."""

    #: The project that is open, or `None`. Carries the integrity report with
    #: it, because every surface that shows a project owes it (ADR-0040).
    project_changed = Signal(object)

    #: The image that is loaded, or `None` — including when loading failed, so a
    #: panel showing the previous scan does not keep showing it.
    image_changed = Signal(object)

    #: A refusal, as a sentence. Ours are messages (ADR-0030); the window decides
    #: whether one deserves a dialog or a status line.
    failed = Signal(str)

    #: The running job, every time it changes state or progress — **and this
    #: signal is the whole of the marshalling ADR-0043 asked M5 for.** The
    #: listener handed to the runner is `job_changed.emit`, called on the worker
    #: thread; Qt queues it onto the thread this object lives on, which is where
    #: every widget connected to it also lives. Anything more elaborate would be
    #: a second thread policy in the layer that must not have one (M5-T07).
    job_changed = Signal(object)

    #: An outcome worth a line in the status bar — what an import did. Separate
    #: from `failed`, because "38 imported, 2 refused" is not a refusal.
    reported = Signal(str)

    #: A stored preference changed. Panels that read one re-read it; the signal
    #: carries no key, because every consumer so far reads exactly one and
    #: filtering by name is work nobody has asked for (M5-T09).
    settings_changed = Signal()

    def __init__(self, app: Nanoscope, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self._project: OpenedProject | None = None
        self._image_id: int | None = None
        self._image: DisplayImage | None = None
        self._job: Job | None = None
        #: The job whose ending has already been dealt with. **A queued signal
        #: carries the handle, not a snapshot**, so every update emitted during
        #: the job is delivered *after* it — each one reading a finished job.
        #: Without this, an import refreshes the project and reports its outcome
        #: once per progress report it ever made.
        self._settled: Job | None = None
        #: Its own listener, on the main thread: the queued signal above is
        #: what makes reading the repository here safe.
        self.job_changed.connect(self._job_changed)

    # ── What is true right now ────────────────────────────────────────────────

    @property
    def project(self) -> OpenedProject | None:
        return self._project

    @property
    def image_id(self) -> int | None:
        """The selected image, whether or not it could be loaded.

        Kept apart from `image` on purpose: a scan whose file is missing is
        still selected, and removing it is exactly what an operator would want
        to do next.
        """
        return self._image_id

    @property
    def image(self) -> DisplayImage | None:
        return self._image

    def image_record(self, image_id: int) -> ImageRecord | None:
        """One of the open project's rows, without going back to the database.

        The panels need a name and a path to put in a sentence, and the project
        they were built from already holds both.
        """
        if self._project is None:
            return None
        return next((image for image in self._project.images if image.id == image_id), None)

    def annotation_count(self, image_id: int) -> int:
        """How much hand work removing this image would destroy (ADR-0044)."""
        repository = self._app.repository
        return 0 if repository is None else len(repository.annotations_for(image_id))

    # ── Preferences, for the panels that may not reach the container ──────────

    def preference(self, key: str, default: object = None) -> object:
        """What is stored for `key`, the project's answer first (ADR-0047).

        Here because a panel may not import the composition root (ADR-0057), and
        a widget reaching for `JsonSettings` would be the same hole in a
        different wall.
        """
        return self._app.settings.get(key, default)

    def own_preference(self, key: str, default: object = None) -> object:
        """What the **operator** stored, ignoring any project override.

        What a settings dialog has to show: it edits the operator's scope, so
        displaying the merged value would put a project's answer in a control
        that writes somewhere else — and OK would then copy the project's choice
        into every other project (ADR-0047's first failure mode, exactly).
        """
        return self._app.application_settings.get_setting(key, default)

    def remember(self, key: str, value: object) -> None:
        """Store a preference for the **operator**, and say that it changed.

        The application scope, always: this application writes no project-scoped
        setting yet, and a dialog that guesses the scope is the failure ADR-0047
        was written to prevent (M5-T09).
        """
        self._app.settings.set(key, value)
        self.settings_changed.emit()

    def devices(self) -> list[Device]:
        """What this machine can run inference on, best first (ADR-0049).

        Cached in the manager, which says so in its own docstring: probing
        imports torch and asks a driver, and a settings dialog must not do that
        on every repaint.
        """
        return self._app.devices.available()

    def overridden_by_project(self, key: str) -> bool:
        """Whether the open project answers this key, and would win.

        `Settings.scope_of` has waited for a caller since M4-T10, whose docstring
        named this one: *what a settings dialog needs to say "this project
        overrides your default"*.
        """
        return self._app.settings.scope_of(key) is Scope.PROJECT

    @property
    def job(self) -> Job | None:
        """The job this window is running, finished or not."""
        return self._job

    @property
    def is_busy(self) -> bool:
        """Whether something is running that owns the project's connection.

        The window disables what would pull the project out from under it —
        `close_project` closes the SQLite connection the worker thread is using.
        """
        return self._job is not None and not self._job.is_finished

    # ── What a widget can ask for ─────────────────────────────────────────────

    def open_project(self, project_dir: Path | str) -> OpenedProject | None:
        """Open a project through the container, and announce it.

        Returns:
            What was opened, or `None` if it was refused — in which case
            `failed` has already carried the reason.
        """
        try:
            opened = self._app.open(project_dir)
        except NanoscopeError as refusal:
            self._refuse(str(refusal))
            return None

        #: Another project's rows are not this selection: ids are per-project,
        #: and image 3 of the old one is not image 3 of the new one.
        self._clear_image()
        self._set_project(opened)
        return opened

    def close_project(self) -> None:
        self._app.close_project()
        self._set_project(None)

    def refresh(self) -> None:
        """Re-read the open project — after a removal, or an import.

        Goes back to the repository rather than editing the list held here: the
        integrity report is part of what is shown, and recomputing it is the
        only way to be sure the panels are not describing the project as it was.
        """
        repository = self._app.repository
        self._set_project(None if repository is None else use_cases.open_project(repository))

    def select_image(self, image_id: int) -> bool:
        """Load one of the project's images, for everything that shows one.

        Loaded **once**, here, rather than once per panel: the viewer draws the
        array and the properties panel describes it, and reading the file twice
        would also make it possible for the two to disagree.

        Returns:
            Whether it could be loaded. A missing file, or a format with no
            reader, is a `failed` message and an `image_changed(None)`.
        """
        repository = self._app.repository
        if repository is None:
            return False

        self._image_id = image_id
        try:
            self._image = load_for_display(repository, image_id)
        except NanoscopeError as refusal:
            self._image = None
            self.image_changed.emit(None)
            self._refuse(str(refusal))
            return False

        self.image_changed.emit(self._image)
        return True

    def remove_image(self, image_id: int) -> bool:
        """Forget an image, and everything that pointed at it (ADR-0044).

        The *asking* is not here: whether to confirm, and in what words, is the
        panel's decision (ADR-0055). This performs it and says what the project
        is now.
        """
        repository = self._app.repository
        if repository is None:
            return False

        record = self.image_record(image_id)
        try:
            repository.remove_image(image_id)
        except NanoscopeError as refusal:  # pragma: no cover — the row was just read
            self._refuse(str(refusal))
            return False

        logger.info(
            "removed image %r",
            record.display_name if record else image_id,
            extra={"image_id": image_id},
        )
        #: `refresh` clears the selection if it was this image — one mechanism
        #: for "the project no longer has it", rather than two that can drift.
        self.refresh()
        return True

    # ── Work that takes long enough to watch (M5-T07) ─────────────────────────

    def import_images(
        self,
        sources: Iterable[Path | str],
        *,
        modality: Modality,
        pixel_size_nm: float | None = None,
    ) -> Job | None:
        """Copy files into the project, in the background.

        The listener is `job_changed.emit` and nothing else: it is called on the
        worker thread, and Qt queues it onto this object's thread — which is the
        adapter ADR-0043 said M5 owed it.

        Returns:
            The handle, or `None` when there is no project or something is
            already running. One job at a time: a status bar has one strip, and
            two would either need two or silently describe the newer.
        """
        repository = self._app.repository
        if repository is None or self.is_busy:
            return None

        files = [Path(source) for source in sources]

        def work(context: JobContext) -> ImportReport:
            return use_cases.import_images(
                repository,
                files,
                modality=modality,
                pixel_size_nm=pixel_size_nm,
                progress=context,
            )

        name = f"Importing {len(files)} file(s)"
        self._job = self._app.jobs.submit(name, work, listener=self.job_changed.emit)
        return self._job

    def cancel_job(self) -> None:
        """Ask the running job to stop. **Ask** — ADR-0043 §3.

        A queued job is dropped; a running one stops at its next checkpoint;
        one with no checkpoint finishes anyway. The button that hides this is
        the one an operator presses twice before deciding the application has
        frozen, so the widget says *stopping* rather than *stopped*.
        """
        if self._job is not None:
            self._job.cancel()

    def _job_changed(self, job: Job) -> None:
        """On the **main** thread, because the signal above is queued.

        Which is what makes this safe: it reads the repository, and a repository
        read from two threads at once is what ADR-0043 §7 had to fix once
        already.
        """
        if not job.is_finished or job is not self._job or job is self._settled:
            return
        self._settled = job

        if job.state is JobState.FAILED:
            logger.error("job %r failed: %s", job.name, job.error)
            self.failed.emit(str(job.error))
            return

        #: Even a stopped import copied files and wrote rows; the panels are
        #: describing a project that has changed either way (ADR-0043 §8).
        self.refresh()
        report = job.result if isinstance(job.result, ImportReport) else None
        #: Logged as well as shown. A status line lasts until the next one; the
        #: project's log is what ADR-0051 set aside to answer *what happened to
        #: this work*, and until M5-T08 put a panel on screen nothing noticed
        #: that an import wrote nothing into it.
        logger.info(
            "%s finished: %s",
            job.name,
            _summarise(report, cancelled=job.cancellation_requested),
        )
        #: **`cancellation_requested`, not `state is CANCELLED`.** `import_images`
        #: stops by *returning* its partial report, so a cancelled import is a
        #: job that succeeded — the state machine describes the job, and the
        #: request is the only record that an operator pressed the button.
        self.reported.emit(_summarise(report, cancelled=job.cancellation_requested))

    # ── Saying so ─────────────────────────────────────────────────────────────

    def _set_project(self, opened: OpenedProject | None) -> None:
        self._project = opened
        #: A selection the project no longer contains is not a selection —
        #: after a close, and after removing the image being looked at.
        if self._image_id is not None and self.image_record(self._image_id) is None:
            self._clear_image()
        self.project_changed.emit(opened)

    def _clear_image(self) -> None:
        """Silent when there is nothing to clear: a signal that says nothing
        changed is one every listener has to learn to ignore."""
        if self._image_id is None and self._image is None:
            return
        self._image_id = None
        self._image = None
        self.image_changed.emit(None)

    def _refuse(self, message: str) -> None:
        logger.error("%s", message)
        self.failed.emit(message)


def _summarise(report: ImportReport | None, *, cancelled: bool) -> str:
    """What an import did, in one line — including what it refused.

    The refusals are counted rather than swallowed: `import_images` collects
    them instead of aborting the batch (ADR-0041), and a batch that quietly
    imports 38 of 40 is a batch an operator finds out about in M6.
    """
    if report is None:  # pragma: no cover — every import returns one
        return "cancelled" if cancelled else "done"

    line = f"Imported {len(report.imported)} file(s)"
    if report.failed:
        line += f", {len(report.failed)} refused: {report.failed[0].reason}"
    if cancelled:
        line += " — cancelled; what was copied is in the project"
    return line
