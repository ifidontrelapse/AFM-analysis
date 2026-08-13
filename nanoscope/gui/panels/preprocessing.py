"""Levelling and the substrate, with the numbers the run actually used (M6-T01).

The first analysis step reachable from a window, and the step every later one
stands on: `run_analysis` preprocesses before it detects, and the substrate this
builds decides what counts as a particle at all.

Three parameters, and **not one of them has a default that lives here**
(`docs/Roadmap.md`, M6: *the UI must not introduce its own defaults*). The
spin boxes start on the values `run_preprocessing` would have used on its own, so
a panel nobody touches produces the byte-identical result the function produced
before this panel existed.

**The preview is asked for, not live.** Preprocessing a 4096² scan is seconds of
NumPy, and a pipeline that re-runs on every keystroke is a UI that fights the
operator (ADR-0061). It runs as a job, which is M5-T07's machinery meeting its
second consumer.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nanoscope.application.use_cases.preprocessing import (
    DEFAULT_MIN_SIZE_NM,
    DEFAULT_OPENING_SCALE,
    PreprocessingParams,
)
from nanoscope.core.entities import PreprocessingResult
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: What the manual radius box means at its minimum: *do not use one*. ADR-0014
#: made a manual radius **the** radius when it is given, so "off" has to be a
#: value the operator can choose rather than a blank they have to interpret.
AUTOMATIC = "estimate it"


class PreprocessingPanel(QWidget):
    """The parameters, the button, and what came back."""

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session

        self.min_size = QDoubleSpinBox(self)
        self.min_size.setRange(0.0, 1_000.0)
        self.min_size.setDecimals(2)
        self.min_size.setSuffix(" nm")
        self.min_size.setValue(DEFAULT_MIN_SIZE_NM)
        self.min_size.setToolTip(
            "The smallest particle radius that counts. A physical size at both of its "
            "sites since ADR-0024 — nothing converts it to pixels with int()."
        )

        self.opening_scale = QDoubleSpinBox(self)
        self.opening_scale.setRange(0.5, 10.0)
        self.opening_scale.setSingleStep(0.1)
        self.opening_scale.setValue(DEFAULT_OPENING_SCALE)
        self.opening_scale.setToolTip(
            "Multiplier on the Otsu typical radius, measured in ADR-0037:\n"
            "smaller finds more particles in a dense field, larger measures radii better."
        )

        self.manual_radius = QDoubleSpinBox(self)
        self.manual_radius.setRange(0.0, 500.0)
        self.manual_radius.setDecimals(1)
        self.manual_radius.setSuffix(" px")
        self.manual_radius.setSpecialValueText(AUTOMATIC)
        self.manual_radius.setToolTip(
            "When given, this is the opening radius — the estimate is not consulted "
            "at all (ADR-0014)."
        )

        self.flatten_note = QCheckBox("Flatten and level (always)", self)
        self.flatten_note.setChecked(True)
        self.flatten_note.setEnabled(False)
        self.flatten_note.setToolTip(
            "A substrate is estimated from a levelled map; there is no version of this "
            "step that skips it. Shown so the pipeline reads in order."
        )

        self.run = QPushButton("Preview", self)
        self.run.clicked.connect(self.preview)
        self.run.setEnabled(False)

        self.report = QLabel("", self)
        self.report.setWordWrap(True)
        self.report.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        form = QFormLayout()
        form.addRow("Minimum radius:", self.min_size)
        form.addRow("Opening scale:", self.opening_scale)
        form.addRow("Opening radius:", self.manual_radius)

        layout = QVBoxLayout(self)
        layout.addWidget(self.flatten_note)
        layout.addLayout(form)
        layout.addWidget(self.run)
        layout.addWidget(self.report)
        layout.addStretch(1)

        for box in (self.min_size, self.opening_scale, self.manual_radius):
            #: The session holds what these say, so a detection run uses the
            #: same numbers as the last preview (M6-T02, ADR-0062).
            box.valueChanged.connect(lambda _value: self._publish())
        self._publish()

        session.image_changed.connect(self._image_changed)
        session.preview_changed.connect(self._preview_changed)
        session.job_changed.connect(lambda _job: self._update_button())
        self._preview_changed(session.preview)

    @property
    def params(self) -> PreprocessingParams:
        """What the boxes currently say. Read by the session, and by a *detection*
        run through it — a scan previewed at one opening scale and analysed at
        another, with nothing saying so, is what one shared value prevents."""
        radius = self.manual_radius.value()
        return PreprocessingParams(
            min_size_nm=self.min_size.value(),
            manual_radius_px=radius if radius > 0 else None,
            opening_scale=self.opening_scale.value(),
        )

    def _publish(self) -> None:
        self._session.set_preprocessing(self.params)

    def preview(self) -> None:
        """Ask for it. The session runs it; this hands over the numbers."""
        self._publish()
        self._session.preprocess()
        self._update_button()

    def _image_changed(self, _image: object) -> None:
        self._update_button()

    def _preview_changed(self, preview: PreprocessingResult | None) -> None:
        self.report.setText(_describe(preview))
        self._update_button()

    def _update_button(self) -> None:
        self.run.setEnabled(self._session.image_id is not None and not self._session.is_busy)


def _describe(preview: PreprocessingResult | None) -> str:
    """What the run used, not what it was asked for.

    ADR-0014 and ADR-0017 both end on that distinction: a manual radius is the
    radius, and the Otsu estimator counts what it *kept*. Both numbers are in
    the result already; nothing here recomputes them.
    """
    if preview is None:
        return "No preview yet."

    sizes = preview.sizes
    radius_nm = sizes.get("typical_radius_nm")
    typical = (
        f"{sizes['typical_radius_px']:.2f} px"
        if radius_nm is None
        else f"{sizes['typical_radius_px']:.2f} px ({radius_nm:.1f} nm)"
    )
    return (
        f"Opening radius used: {preview.opening_radius} px\n"
        f"Typical particle radius: {typical}\n"
        f"Objects kept by the estimate: {sizes.get('n_objects', '—')}"
    )
