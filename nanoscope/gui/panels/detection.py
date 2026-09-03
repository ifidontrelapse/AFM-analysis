"""Detection, offering only what can actually run (M6-T02, ADR-0062).

M6's third exit criterion is written against this panel:

> *Invalid combinations are disabled in the UI **because the capability matrix
> says so** — not by a duplicated rule.*

So this widget **enumerates nothing**. It asks the session for the options for
the selected image's modality and renders them; the detector names, the modes,
and the sentences explaining why an entry cannot run all come from
`application.capabilities`. PROJECT_RULES §2.5 is the reason — no model or
detector name may be written in `gui/` — and the deeper one is D-19: the deleted
React client kept its own copy of this matrix, and the copy had drifted. **A test
greps this package for those names**, so the rule and its enforcement ship
together.

**A disabled entry says why.** "Greyed out with no explanation" is the failure
this criterion exists to prevent, not a milder version of it.

Unlike M6-T01's preview, what this produces is **kept**: `run_analysis` writes
the run, its detections and its measurement table (ADR-0042).
"""

from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nanoscope.application.capabilities import DetectorOption, ModeOption
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


class DetectionPanel(QWidget):
    """Pick a detector and a mode the matrix allows, then run it."""

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session

        self.detector = QComboBox(self)
        self.detector.currentIndexChanged.connect(self._detector_changed)
        self.mode = QComboBox(self)
        self.mode.currentIndexChanged.connect(self._update_run)

        #: The blob-detector parameters, labelled by what they *do* rather than
        #: by whose they are — the name is the application's business, not a
        #: widget's. `PipelineConfig` carries the defaults, so none of the
        #: numbers below is invented in this file (M6's rule).
        defaults = PipelineConfig()
        self.overlap = QDoubleSpinBox(self)
        self.overlap.setRange(0.0, 1.0)
        self.overlap.setSingleStep(0.05)
        self.overlap.setValue(defaults.log_overlap)
        self.overlap.setToolTip("How much two blobs may overlap before they count as one.")

        self.percentile = QDoubleSpinBox(self)
        self.percentile.setRange(0.0, 100.0)
        self.percentile.setValue(defaults.log_percentile)
        self.percentile.setToolTip(
            "The percentile of the blob response used to pick a threshold when none is given."
        )

        self.reason = QLabel("", self)
        self.reason.setWordWrap(True)
        self.reason.setStyleSheet(f"color: {tokens.WARNING};")

        self.run = QPushButton("Run", self)
        self.run.clicked.connect(self.start)

        self.report = QLabel("No run yet.", self)
        self.report.setWordWrap(True)
        self.report.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        form = QFormLayout()
        form.addRow("Detector:", self.detector)
        form.addRow("Mode:", self.mode)
        form.addRow("Blob overlap:", self.overlap)
        form.addRow("Threshold percentile:", self.percentile)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.reason)
        layout.addWidget(self.run)
        layout.addWidget(self.report)
        layout.addStretch(1)

        session.image_changed.connect(lambda _image: self.reload())
        session.job_changed.connect(lambda _job: self._update_run())
        #: The active model is a stored preference, so choosing one in the
        #: Models dialog has to re-enable Run here without a restart (M8-T06).
        session.settings_changed.connect(self._update_run)
        session.run_stored.connect(self._run_stored)
        self.reload()

    # ── What the matrix allows ────────────────────────────────────────────────

    def reload(self) -> None:
        """Rebuild the choices for whatever image is selected now."""
        self.detector.blockSignals(True)
        self.detector.clear()
        for option in self._session.detector_options():
            self.detector.addItem(option.detector, option)
            index = self.detector.count() - 1
            if not option.available:
                #: Disabled *and* explained: the item keeps its reason as a
                #: tooltip, and selecting it is impossible rather than merely
                #: discouraged.
                _item(self.detector, index).setEnabled(False)
                self.detector.setItemData(index, option.reason, Qt.ItemDataRole.ToolTipRole)
        self.detector.blockSignals(False)
        self.detector.setCurrentIndex(_first_available(self.detector))
        self._detector_changed()

    def _detector_changed(self) -> None:
        option: DetectorOption | None = self.detector.currentData(_OPTION)
        self.mode.blockSignals(True)
        self.mode.clear()
        for mode in option.modes if option else ():
            self.mode.addItem(mode.mode, mode)
            index = self.mode.count() - 1
            if not mode.available:
                _item(self.mode, index).setEnabled(False)
                self.mode.setItemData(index, mode.reason, Qt.ItemDataRole.ToolTipRole)
        self.mode.blockSignals(False)
        self.mode.setCurrentIndex(_first_available(self.mode))
        self._update_run()

    def _update_run(self) -> None:
        option: DetectorOption | None = self.detector.currentData(_OPTION)
        mode: ModeOption | None = self.mode.currentData(_OPTION)
        runnable = bool(option and option.available and mode and mode.available)
        #: **Registered is not chosen.** The matrix refuses a detector whose
        #: framework has no model in this project; a project can have three and
        #: none in use, and without this that run preprocesses a scan and *then*
        #: refuses — the late failure M8-T06 exists to remove (ADR-0086).
        unchosen = bool(option and runnable and self._session.needs_active_model(option.detector))
        self.run.setEnabled(runnable and not unchosen and not self._session.is_busy)
        self.reason.setText(
            _NO_ACTIVE_MODEL if unchosen else ("" if runnable else _why_not(option, mode))
        )

    # ── Running it ────────────────────────────────────────────────────────────

    def config(self) -> PipelineConfig | None:
        """What the panel is asking for, or `None` when it may not ask.

        Built here and validated below: `run_pipeline` calls `validate_request`
        before it reads a file, so an impossible request refuses in milliseconds
        rather than after a GPU pass — which is D-14, fixed in M2-T10 and worth
        not undoing from a widget.
        """
        option: DetectorOption | None = self.detector.currentData(_OPTION)
        mode: ModeOption | None = self.mode.currentData(_OPTION)
        if option is None or mode is None or not (option.available and mode.available):
            return None
        return PipelineConfig(
            detector=option.detector,  # type: ignore[arg-type]  # the matrix's own value
            mode=mode.mode,  # type: ignore[arg-type]
            log_overlap=self.overlap.value(),
            log_percentile=self.percentile.value(),
        )

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


def _item(combo: QComboBox, index: int) -> QStandardItem:
    """One entry of a combo, as the thing that can be disabled.

    `QComboBox.model()` is typed as the abstract base; a default combo's model
    *is* a `QStandardItemModel`, and disabling an entry through it is Qt's own
    way. The alternative — writing a magic value into `UserRole - 1` — is the
    same thing with the type information thrown away.
    """
    return cast(QStandardItemModel, combo.model()).item(index)


def _why_not(option: DetectorOption | None, mode: ModeOption | None) -> str:
    """The sentence the application gave, or a plain statement of the situation."""
    if option is None:
        return "Select an image to analyse."
    if not option.available:
        return option.reason or "This detector cannot run here."
    if mode is not None and not mode.available:
        return mode.reason or "This mode cannot run here."
    return ""


def _first_available(combo: QComboBox) -> int:
    """The first entry that can be chosen, or the first one at all.

    A combo opening on a disabled entry is a combo whose Run button is dead for
    a reason nobody asked for.
    """
    for index in range(combo.count()):
        if _item(combo, index).isEnabled():
            return index
    return 0 if combo.count() else -1
