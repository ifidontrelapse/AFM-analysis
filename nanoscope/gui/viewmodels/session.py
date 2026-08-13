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

from nanoscope.application import capabilities, use_cases
from nanoscope.application.capabilities import DetectorOption
from nanoscope.application.jobs import Job, JobContext, JobState
from nanoscope.application.settings import Scope
from nanoscope.application.use_cases.display import DisplayImage, Stage, load_for_display
from nanoscope.application.use_cases.preprocessing import PreprocessingParams
from nanoscope.core.entities import AnalysisRun, PipelineConfig, PreprocessingResult
from nanoscope.core.entities.device import Device
from nanoscope.core.entities.model import ModelTask
from nanoscope.core.entities.project import (
    Annotation,
    ImageRecord,
    ImportReport,
    OpenedProject,
)
from nanoscope.core.errors import NanoscopeError
from nanoscope.core.values import Modality

if TYPE_CHECKING:
    import pandas as pd

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

    #: The preprocessing preview, or `None` when there is none. Not a run: a
    #: preview is a look at intermediate arrays, and what a *run* is belongs to
    #: `run_analysis` and the rows it stores (ADR-0042, ADR-0061).
    preview_changed = Signal(object)

    #: A run was stored, with its detections and its measurement table. Unlike
    #: a preview, this is a *result* — the difference ADR-0061 §5 exists for.
    run_stored = Signal(object)

    #: The run whose detections are on screen, or `None`. The newest one for
    #: the selected image, replaced by a run this session stores (M6-T03).
    run_changed = Signal(object)

    #: Which particle is selected — an index into the current run's detections,
    #: or `None`. Emitted for both directions: the table asks, the canvas asks,
    #: and both listen for the answer (ADR-0065).
    particle_selected = Signal(object)

    #: The selected image's annotations, or an empty tuple. Hand work is the
    #: one thing in a project that cannot be recomputed (ADR-0044), and until
    #: M7-T01 it was also the only data with no representation on screen.
    annotations_changed = Signal(object)

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
        self._preview: PreprocessingResult | None = None
        #: The numbers the preprocessing panel is showing. Held here so a
        #: *detection* uses the same ones: a scan previewed at one opening scale
        #: and analysed at another, with nothing saying so, is the defect this
        #: single value prevents (M6-T02).
        self._preprocessing = PreprocessingParams()
        self._run: AnalysisRun | None = None
        self._annotations: tuple[Annotation, ...] = ()
        self._particle: int | None = None
        self._stage = Stage.RAW
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

    @property
    def image_position(self) -> tuple[int, int] | None:
        """`(which, how many)`, one-based, or `None` when nothing is selected.

        Half of navigating is knowing whether there is anywhere left to go
        (M6-T08).
        """
        images = () if self._project is None else self._project.images
        ids = [image.id for image in images]
        if self._image_id is None or self._image_id not in ids:
            return None
        return ids.index(self._image_id) + 1, len(ids)

    def position_text(self) -> str:
        """ "3 of 40", or nothing at all when nothing is selected."""
        position = self.image_position
        return "" if position is None else f"{position[0]} of {position[1]}"

    def select_next(self) -> bool:
        """The next scan in the project's own order. **No wrapping.**

        Wrapping takes an operator from the fortieth scan to the first without
        saying so, and the review that asks *"did I look at all of them?"* is
        exactly the one that must not lie (ADR-0068).
        """
        return self._step(+1)

    def select_previous(self) -> bool:
        return self._step(-1)

    def _step(self, by: int) -> bool:
        position = self.image_position
        if position is None or self._project is None:
            return False
        index = position[0] - 1 + by
        if not 0 <= index < len(self._project.images):
            return False
        return self.select_image(self._project.images[index].id)

    def annotation_count(self, image_id: int) -> int:
        """How much hand work removing this image would destroy (ADR-0044)."""
        repository = self._app.repository
        return 0 if repository is None else len(repository.annotations_for(image_id))

    # ── The preprocessing preview (M6-T01) ───────────────────────────────────

    @property
    def preview(self) -> PreprocessingResult | None:
        return self._preview

    @property
    def stage(self) -> Stage:
        """Which array the viewer is showing. `RAW` is the file itself."""
        return self._stage

    def show_stage(self, stage: Stage) -> None:
        """Choose the array to look at, and announce it.

        Refused silently for a stage there is no preview for: the panels offer
        one only while a preview exists, and a viewer switched to an array that
        was never computed would draw the last one under a new name — which is
        the one thing ADR-0056 forbids.
        """
        if stage is not Stage.RAW and self._preview is None:
            return
        self._stage = stage
        self.preview_changed.emit(self._preview)

    @property
    def preprocessing(self) -> PreprocessingParams:
        return self._preprocessing

    def set_preprocessing(self, params: PreprocessingParams) -> None:
        """What the preprocessing panel currently says. One place, two readers."""
        self._preprocessing = params

    def preprocess(self) -> Job | None:
        """Level the selected scan and estimate its substrate, in the background.

        A job because preprocessing a 4096² scan is seconds of NumPy
        (Architecture §4.5), and **asked for** rather than live, because a
        pipeline that re-runs on every keystroke is a UI that fights the
        operator (ADR-0061).

        Returns:
            The handle, or `None` when nothing is selected or something is
            already running.
        """
        repository = self._app.repository
        image_id = self._image_id
        if repository is None or image_id is None or self.is_busy:
            return None

        def work(context: JobContext) -> PreprocessingResult:
            context.report(0, 0, "levelling and estimating the substrate")
            return use_cases.preprocess_image(repository, image_id, self._preprocessing)

        name = f"Preprocessing {self.image_record(image_id).display_name}"  # type: ignore[union-attr]
        self._job = self._app.jobs.submit(name, work, listener=self.job_changed.emit)
        return self._job

    @property
    def run(self) -> AnalysisRun | None:
        """The analysis whose detections are being shown, if there is one."""
        return self._run

    @property
    def annotations(self) -> tuple[Annotation, ...]:
        """What a person judged about the selected image."""
        return self._annotations

    def reload_annotations(self) -> None:
        """Re-read them from the project — on selection, and after an edit.

        `annotations_for` has had one caller since M4-T07: M5-T04's confirmation
        dialog, which *counts* them without ever showing one.
        """
        repository = self._app.repository
        self._annotations = (
            ()
            if repository is None or self._image_id is None
            else tuple(repository.annotations_for(self._image_id))
        )
        self.annotations_changed.emit(self._annotations)

    def runs(self) -> list[AnalysisRun]:
        """Every stored analysis of the selected image, oldest first.

        Three analyses of one scan leave three rows, and until M6-T09 the window
        could reach exactly one of them.
        """
        repository = self._app.repository
        if repository is None or self._image_id is None:
            return []
        return repository.runs_for(self._image_id)

    def select_run(self, run_id: int) -> bool:
        """Show a stored run of the selected image.

        Its **masks are not there** and cannot be: nothing persists them
        (ADR-0042, ADR-0064). Its detections and its measurement table are.
        """
        chosen = next((run for run in self.runs() if run.id == run_id), None)
        if chosen is None:
            return False
        self._run = chosen
        self._particle = None
        self.particle_selected.emit(None)
        self.run_changed.emit(chosen)
        return True

    def _show_newest_run(self) -> None:
        """The newest stored run for the selected image, or none.

        `runs_for` has existed since M4-T05 and nothing had read it: a scan
        analysed yesterday, selected today, showed nothing. M6-T09 owns proving
        that survives a restart; showing it at all is M6-T03's job.
        """
        repository = self._app.repository
        runs = (
            []
            if repository is None or self._image_id is None
            else repository.runs_for(self._image_id)
        )
        self._run = runs[-1] if runs else None
        self._particle = None
        self.particle_selected.emit(None)
        self.run_changed.emit(self._run)

    @property
    def particle(self) -> int | None:
        """The selected detection's index in the current run, if any."""
        return self._particle

    def select_particle(self, index: int | None) -> None:
        """Select one particle, from wherever the operator pointed at it.

        The viewmodel holds it because both the table and the canvas can ask and
        both have to be told; two widgets telling each other is what ADR-0057
        removed, and *"and vice versa"* is where it would come back.
        """
        run = self._run
        if index is not None and (run is None or not 0 <= index < len(run.detections)):
            index = None
        if index == self._particle:
            return
        self._particle = index
        self.particle_selected.emit(index)

    def measurements(self) -> pd.DataFrame | None:
        """The current run's stored table, or `None` when it measured nothing.

        `detect` mode writes no table at all (ADR-0042), and an empty grid with
        the right columns would claim it did.
        """
        repository = self._app.repository
        run = self._run
        if repository is None or run is None or run.measurements_path is None:
            return None
        try:
            return repository.measurements_for(run)
        except NanoscopeError as refusal:
            self._refuse(str(refusal))
            return None

    def particle_at(self, x_px: float, y_px: float, *, tolerance: float = 1.0) -> int | None:
        """Which detection sits at these coordinates, if one does.

        **Coordinates, not indices.** The measurement table is a *subset* of the
        detections — a height that is not a number is discarded (ADR-0033) — so
        row *n* is not detection *n*, and `x_px`/`y_px` are the one link both
        sides carry (ADR-0031's core columns).
        """
        run = self._run
        if run is None:
            return None
        for index, detection in enumerate(run.detections):
            if abs(detection.x_px - x_px) <= tolerance and abs(detection.y_px - y_px) <= tolerance:
                return index
        return None

    def detector_options(self) -> tuple[DetectorOption, ...]:
        """What the selected image's modality allows, and why the rest does not.

        The panel renders this and decides nothing: PROJECT_RULES §2.5 keeps
        detector names out of `gui/`, and M6's exit criterion asks for a UI that
        cannot express an invalid request rather than one that apologises after
        the fact (ADR-0062).
        """
        repository = self._app.repository
        record = None if self._image_id is None else self.image_record(self._image_id)
        if repository is None or record is None:
            return ()

        models = repository.list_models()
        frameworks = {model.framework for model in models if model.task is ModelTask.DETECT}
        #: A **registered** model, not a constructed predictor: ADR-0050 made
        #: the registry cheap so that asking "can this project segment?" reads
        #: nothing off a disk. The weights are loaded inside the job (ADR-0064).
        can_segment = any(model.task is ModelTask.SEGMENT for model in models)
        return capabilities.detector_options(
            record.modality.value, frameworks=frameworks, has_predictor=can_segment
        )

    def detect(self, config: PipelineConfig) -> Job | None:
        """Run the pipeline over the selected scan, and **store what it found**.

        Not a preview: `run_analysis` writes the run, its detections and its
        measurement table (ADR-0042). The preprocessing parameters travel with
        it, so the arrays this analyses are the arrays the preview showed.

        Returns:
            The handle, or `None` when nothing is selected or something is
            already running.
        """
        repository = self._app.repository
        image_id = self._image_id
        if repository is None or image_id is None or self.is_busy:
            return None

        params = self._preprocessing
        needs_predictor = any(
            row.requires_predictor
            for row in capabilities.CAPABILITIES
            if (row.detector, row.mode) == (config.detector, config.mode)
        )

        def work(context: JobContext) -> AnalysisRun:
            context.report(0, 0, f"{config.mode} with {config.detector}")
            #: Built here, on the worker thread: constructing it reads weights
            #: off a disk, and the main thread is the one drawing (ADR-0064).
            predictor = self._app.segmentation_predictor() if needs_predictor else None
            return use_cases.run_analysis(
                repository, image_id, config, predictor=predictor, preprocessing=params
            )

        name = f"Analysing {self.image_record(image_id).display_name}"  # type: ignore[union-attr]
        self._job = self._app.jobs.submit(name, work, listener=self.job_changed.emit)
        return self._job

    def export(self, *, everything: bool) -> Job | None:
        """Write measurements to a CSV under the project's `exports/` (M6-T07).

        Args:
            everything: every run of every image, which is ADR-0048's reason for
                taking a collection — *"statistics across a dataset is why the
                measurements exist"*. `False` exports the run on screen.

        Returns:
            The handle, or `None` when there is no project, nothing selected, or
            something already running. A run that measured nothing is **not**
            refused here: `export_measurements` says so in a sentence, and
            pre-empting it with a silent no-op would say less (ADR-0048).
        """
        repository = self._app.repository
        run = self._run
        if repository is None or self.is_busy or (not everything and run is None):
            return None

        #: `None` is "every run of every image"; a one-element list is this one.
        #: Narrowed here rather than in the closure, so the type says what the
        #: guard above already proved.
        runs: list[AnalysisRun] | None = None if everything else [_not_none(run)]

        def work(context: JobContext) -> str:
            context.report(0, 0, "collecting measurements")
            return use_cases.export_measurements(repository, runs)

        name = "Exporting every run" if everything else f"Exporting run {_not_none(run).id}"
        self._job = self._app.jobs.submit(name, work, listener=self.job_changed.emit)
        return self._job

    def _clear_preview(self) -> None:
        """A preview belongs to the scan it was computed from.

        Dropped when the selection changes, because a substrate map from another
        scan drawn over this one would be the worst possible version of this
        feature.
        """
        if self._preview is None and self._stage is Stage.RAW:
            return
        self._preview = None
        self._stage = Stage.RAW
        self.preview_changed.emit(None)

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
        self._clear_preview()
        self._show_newest_run()
        self.reload_annotations()
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

        if job.state is JobState.SUCCEEDED and isinstance(job.result, str):
            #: The path the export went to, relative to the project root — the
            #: one thing an operator needs after asking for a file they never
            #: chose a name for (ADR-0067).
            logger.info("%s finished: %s", job.name, job.result)
            self.reported.emit(f"Exported to {job.result}")
            return

        if job.state is JobState.SUCCEEDED and isinstance(job.result, AnalysisRun):
            logger.info(
                "%s finished: %d detection(s), run %d",
                job.name,
                len(job.result.detections),
                job.result.id,
                extra={"image_id": self._image_id},
            )
            self._run = job.result
            self._particle = None
            self.particle_selected.emit(None)
            self.run_changed.emit(self._run)
            self.run_stored.emit(job.result)
            self.reported.emit(
                f"{job.result.mode} with {job.result.detector}: "
                f"{len(job.result.detections)} detection(s)"
            )
            return

        if job.state is JobState.SUCCEEDED and isinstance(job.result, PreprocessingResult):
            self._preview = job.result
            self._stage = Stage.RESULT
            logger.info(
                "%s finished: opening radius %d px",
                job.name,
                job.result.opening_radius,
                extra={"image_id": self._image_id},
            )
            self.preview_changed.emit(self._preview)
            return

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
        self._clear_preview()
        self._run = None
        self._particle = None
        self._annotations = ()
        self.particle_selected.emit(None)
        self.run_changed.emit(None)
        self.annotations_changed.emit(())
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


def _not_none(run: AnalysisRun | None) -> AnalysisRun:
    """The run the caller's own guard already proved is there."""
    assert run is not None
    return run
