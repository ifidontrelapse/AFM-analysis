"""Detection and segmentation, offering only what can actually run (M6-T02, ADR-0062).

M6's third exit criterion is written against this panel:

> *Invalid combinations are disabled in the UI **because the capability matrix
> says so** — not by a duplicated rule.*

So this widget **enumerates nothing**. It asks the session for the options for
the selected image's modality and renders them; the detector names, the modes,
the parameters each is tuned by, and the sentences explaining what is missing
all come from `application.capabilities`. PROJECT_RULES §2.5 is the reason — no
model or detector name may be written in `gui/` — and the deeper one is D-19:
the deleted React client kept its own copy of this matrix, and the copy had
drifted. **A test greps this package for those names**, so the rule and its
enforcement ship together.

**What cannot run is not on the list.** Until now it was listed and disabled,
which is what ADR-0062 asked for and what an operator read as a broken
application: a row that cannot be clicked, with its explanation in a tooltip
nobody hovers. The explanation is now a sentence under the two combos, on
screen, saying what is missing and which menu registers it — the criterion's
actual demand, met without offering a choice that is not one.

**The mode is asked first**, because it is the question — *am I counting
particles or measuring them?* — and the detector is how it gets answered. Only
detectors that can do the chosen mode are then offered.

Unlike M6-T01's preview, what this produces is **kept**: `run_analysis` writes
the run, its detections and its measurement table (ADR-0042).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nanoscope.application.capabilities import DetectorOption, ModeOption, Parameter
from nanoscope.core.entities import AnalysisRun, PipelineConfig
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: Where an option is kept on its combo entry.
_OPTION = Qt.ItemDataRole.UserRole

#: Said when this detector needs registered weights and the project has chosen
#: none. Names the menu that fixes it, because "greyed out with no explanation"
#: is the failure M6's third exit criterion exists to prevent, and *"choose a
#: model"* without saying where is the same failure with more words.
_NO_ACTIVE_MODEL = (
    "No model is in use in this project. Pick one under File ▸ Models…, "
    "or train one under File ▸ Train a Model…"
)

#: Where the answer to every "…is registered in this project" sentence is.
_WHERE = "File ▸ Models… registers weights you have; File ▸ Train a Model… makes one."


class DetectionPanel(QWidget):
    """Pick a mode and a detector the matrix allows, tune it, then run it."""

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        #: Every parameter an operator has touched, by `PipelineConfig` field, so
        #: switching detector and back does not silently reset a number they set.
        #: Survives the rebuild; the *defaults* come from `PipelineConfig` itself.
        self._values: dict[str, float] = {}
        self._spins: dict[str, QDoubleSpinBox] = {}

        self.mode = QComboBox(self)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.detector = QComboBox(self)
        self.detector.currentIndexChanged.connect(self._detector_changed)

        self.reason = QLabel("", self)
        self.reason.setWordWrap(True)
        self.reason.setStyleSheet(f"color: {tokens.WARNING};")

        #: What is *not* on the two lists above, and what would put it there. The
        #: half of "disabled entries say why" worth keeping (ADR-0062).
        self.missing = QLabel("", self)
        self.missing.setWordWrap(True)
        self.missing.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        self.run = QPushButton("Run", self)
        self.run.clicked.connect(self.start)

        self.report = QLabel("No run yet.", self)
        self.report.setWordWrap(True)
        self.report.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        form = QFormLayout()
        form.addRow("Mode:", self.mode)
        form.addRow("Detector:", self.detector)

        #: Rebuilt on every change of either combo: the blob parameters were on
        #: screen for every detector until this, and a number that does nothing
        #: is a number an operator will spend an afternoon on.
        self.parameters = QFormLayout()

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(self.parameters)
        layout.addWidget(self.reason)
        layout.addWidget(self.missing)
        layout.addWidget(self.run)
        layout.addWidget(self.report)
        layout.addStretch(1)

        session.image_changed.connect(lambda _image: self.reload())
        session.job_changed.connect(lambda _job: self._update_run())
        #: The active model is a stored preference, so choosing one in the
        #: Models dialog has to re-enable Run here without a restart (M8-T06) —
        #: and **registering** one changes the options themselves, not merely
        #: whether Run is pressable: a detector whose framework had no model is
        #: not on the list at all, and until this rebuilt it, the way to reach it
        #: was to select a different scan and come back.
        session.settings_changed.connect(self.reload)
        session.run_stored.connect(self._run_stored)
        self.reload()

    # ── What the matrix allows ────────────────────────────────────────────────

    def reload(self) -> None:
        """Rebuild the choices for whatever image is selected now.

        **What was chosen survives the rebuild** when it is still on offer. This
        runs on every settings change — a colormap included — and a panel that
        answered *"the operator picked a colormap"* by resetting their detector
        would be a worse bug than the one it fixes.
        """
        self._options = self._session.detector_options()
        chosen = self.mode.currentText()

        self.mode.blockSignals(True)
        self.mode.clear()
        for mode in _modes(self._options):
            self.mode.addItem(mode.mode, mode)
            self.mode.setItemData(self.mode.count() - 1, mode.reason, Qt.ItemDataRole.ToolTipRole)
        self.mode.blockSignals(False)
        self.mode.setCurrentIndex(_wanted(self.mode, chosen))
        self._mode_changed()

    def _mode_changed(self) -> None:
        """Offer the detectors that can do the chosen mode, and nothing else."""
        wanted = self.mode.currentText()
        chosen = self.detector.currentText()

        self.detector.blockSignals(True)
        self.detector.clear()
        for option in self._options:
            if option.available and any(m.mode == wanted and m.available for m in option.modes):
                self.detector.addItem(option.detector, option)
        self.detector.blockSignals(False)
        self.detector.setCurrentIndex(_wanted(self.detector, chosen))
        self._detector_changed()

    def _detector_changed(self) -> None:
        self._build_parameters()
        self._update_run()

    # ── The knobs this combination actually has ───────────────────────────────

    def _build_parameters(self) -> None:
        """One spin box per parameter the chosen detector and mode declare."""
        for spin in self._spins.values():
            #: Read before the widget goes, so a number an operator typed is
            #: still theirs when the same parameter comes back.
            self._values[spin.objectName()] = spin.value()
        while self.parameters.rowCount():
            self.parameters.removeRow(0)
        self._spins = {}

        defaults = PipelineConfig()
        for parameter in self._parameters():
            spin = QDoubleSpinBox(self)
            spin.setObjectName(parameter.field)
            spin.setDecimals(parameter.decimals)
            spin.setRange(parameter.minimum, parameter.maximum)
            spin.setSingleStep(parameter.step)
            spin.setValue(
                self._values.get(parameter.field, float(getattr(defaults, parameter.field)))
            )
            spin.setToolTip(parameter.help)
            self.parameters.addRow(f"{parameter.label}:", spin)
            self._spins[parameter.field] = spin

    def _parameters(self) -> tuple[Parameter, ...]:
        """The detector's, then the mode's — how it is found, then how it is measured."""
        option: DetectorOption | None = self.detector.currentData(_OPTION)
        mode = self._mode_of(option)
        return (() if option is None else option.parameters) + (
            () if mode is None else mode.parameters
        )

    def _mode_of(self, option: DetectorOption | None) -> ModeOption | None:
        """The chosen mode **as this detector offers it**, which is where its
        parameters and its availability live."""
        wanted = self.mode.currentText()
        if option is None:
            return None
        return next((m for m in option.modes if m.mode == wanted), None)

    # ── Whether Run may be pressed, and what is missing if not ────────────────

    def _update_run(self) -> None:
        option: DetectorOption | None = self.detector.currentData(_OPTION)
        mode = self._mode_of(option)
        runnable = bool(option and option.available and mode and mode.available)
        #: **Registered is not chosen.** The matrix offers a detector whose
        #: framework has a model in this project; a project can have three and
        #: none in use, and without this that run preprocesses a scan and *then*
        #: refuses — the late failure M8-T06 exists to remove (ADR-0086).
        unchosen = bool(option and runnable and self._session.needs_active_model(option.detector))
        self.run.setEnabled(runnable and not unchosen and not self._session.is_busy)
        self.reason.setText(_NO_ACTIVE_MODEL if unchosen else ("" if runnable else _why_not(self)))
        self.missing.setText(_what_is_missing(self._options))

    # ── Running it ────────────────────────────────────────────────────────────

    def config(self) -> PipelineConfig | None:
        """What the panel is asking for, or `None` when it may not ask.

        Built here and validated below: `run_pipeline` calls `validate_request`
        before it reads a file, so an impossible request refuses in milliseconds
        rather than after a GPU pass — which is D-14, fixed in M2-T10 and worth
        not undoing from a widget.
        """
        option: DetectorOption | None = self.detector.currentData(_OPTION)
        mode = self._mode_of(option)
        if option is None or mode is None or not (option.available and mode.available):
            return None
        config = PipelineConfig(
            detector=option.detector,  # type: ignore[arg-type]  # the matrix's own value
            mode=mode.mode,  # type: ignore[arg-type]
        )
        #: Set by name rather than passed as `**kwargs`: the fields are `float`
        #: and `int` both, and a test asserts every `Parameter.field` is one of
        #: them — which is the check that makes this safe (`test_capabilities`).
        for field, spin in self._spins.items():
            setattr(config, field, _number(spin))
        return config

    def start(self) -> None:
        config = self.config()
        if config is not None:
            self._session.detect(config)
            self._update_run()

    def _run_stored(self, run: AnalysisRun) -> None:
        self.report.setText(
            f"Run {run.id}: {len(run.detections)} detection(s) from {run.detector} "
            f"in {run.mode} mode.\nMeasurements: {run.measurements_path or 'none for this mode'}"
        )


def _number(spin: QDoubleSpinBox) -> float | int:
    """A whole number where the parameter said so — `PipelineConfig` has `int`
    fields, and a float in one of them is a value nothing else in the pipeline
    would have produced."""
    return spin.value() if spin.decimals() else int(spin.value())


def _modes(options: tuple[DetectorOption, ...]) -> list[ModeOption]:
    """Every mode some available detector can run, in the matrix's own order.

    One entry per mode rather than per (detector, mode): the mode is the
    question, and which detectors answer it is the next combo down.
    """
    seen: dict[str, ModeOption] = {}
    for option in options:
        if not option.available:
            continue
        for mode in option.modes:
            if mode.available and mode.mode not in seen:
                seen[mode.mode] = mode
    return list(seen.values())


def _what_is_missing(options: tuple[DetectorOption, ...]) -> str:
    """Why the lists above are as short as they are, in the matrix's words.

    Each sentence once, however many entries it kept off the list, and the menu
    that answers them all appended — *"register a model"* without saying where
    is the greyed-out tooltip again with more words.
    """
    reasons: list[str] = []
    withheld = (
        (option.reason if not option.available else None, *_refused(option)) for option in options
    )
    for reason in (one for group in withheld for one in group):
        if reason and reason not in reasons:
            reasons.append(reason)
    if not reasons:
        return ""
    return f"Not offered here: {' '.join(_sentence(one) for one in reasons)} {_WHERE}"


def _sentence(reason: str) -> str:
    """The matrix writes its reasons lower-case, to be pasted into one; several
    of them in a row read as one run-on sentence unless each starts like one."""
    return f"{reason[:1].upper()}{reason[1:]}."


def _refused(option: DetectorOption) -> tuple[str | None, ...]:
    """The reasons this detector's modes are not on the list."""
    return tuple(mode.reason for mode in option.modes if not mode.available)


def _why_not(panel: DetectionPanel) -> str:
    """The sentence for an empty panel — there is nothing on the lists to explain."""
    if panel.mode.count() == 0:
        return "Select an image to analyse."
    return "Nothing here can run this mode."


def _wanted(combo: QComboBox, text: str) -> int:
    """Where `text` is now, if it is still on offer; else the first entry."""
    index = combo.findText(text)
    return index if index >= 0 else (0 if combo.count() else -1)
