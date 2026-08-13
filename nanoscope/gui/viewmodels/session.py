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
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from nanoscope.application import use_cases
from nanoscope.application.use_cases.display import DisplayImage, load_for_display
from nanoscope.core.entities.project import ImageRecord, OpenedProject
from nanoscope.core.errors import NanoscopeError

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

    def __init__(self, app: Nanoscope, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self._project: OpenedProject | None = None
        self._image_id: int | None = None
        self._image: DisplayImage | None = None

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
