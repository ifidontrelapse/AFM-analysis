"""What is running, and the button that asks it to stop (M5-T07, ADR-0058).

M5's third exit criterion is *"a long-running job shows progress and can be
cancelled **without freezing the UI**"*, and that phrase rules out the obvious
implementation: a modal progress dialog *is* the frozen window it is reporting
on. This lives in the status bar instead, so an operator can look at their data
while an import runs.

Two honesty rules, both from ADR-0043 and both easy to break by accident:

- **`total == 0` means "cannot say"**, so the bar goes into Qt's busy mode. A
  determinate bar sitting at 0 % that never moves is a lie about the same fact.
- **Cancel asks.** A queued job is dropped, a running one stops at its next
  checkpoint, and one with no checkpoint runs to the end — so once it is pressed
  the strip says *stopping*, not *stopped*.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from nanoscope.application.jobs import Job
from nanoscope.gui.viewmodels import SessionViewModel

#: What the button says after it has been pressed. ADR-0043 §3 in four words:
#: the request is recorded, and the work stops where the work can stop.
STOPPING = "Stopping…"


class JobStatus(QWidget):
    """One strip: what is running, how far along, and a way to ask it to stop.

    Hidden whenever nothing is running — a progress bar with no job is furniture
    that teaches an operator to stop reading the status bar.
    """

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session

        self.label = QLabel("", self)
        self.bar = QProgressBar(self)
        self.bar.setMaximumWidth(160)
        #: Counts, not a percentage — `%v of %m` is "2 of 40", which is what the
        #: job actually knows (ADR-0043 §4). Looked at in a real window first:
        #: with the text off, a bar at 0 of 6 is an empty box saying nothing.
        #: Qt draws no text at all in busy mode, which is the right answer there.
        self.bar.setFormat("%v of %m")
        self.cancel = QPushButton("Cancel", self)
        self.cancel.setToolTip(
            "Asks the job to stop at its next checkpoint.\n"
            "An import stops between files; work already done is kept."
        )
        self.cancel.clicked.connect(self._cancel_pressed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.label, self.bar, self.cancel):
            layout.addWidget(widget)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        session.job_changed.connect(self.show_job)
        self.show_job(session.job)

    def show_job(self, job: Job | None) -> None:
        """Describe the job, or disappear.

        Called on the **main** thread even though the runner reports from a
        worker one: the session's signal is queued, which is the whole of the
        marshalling (ADR-0058 §1).
        """
        if job is None or job.is_finished:
            self.setVisible(False)
            return

        progress = job.progress
        self.setVisible(True)
        self.label.setText(f"{job.name} — {progress.message}" if progress.message else job.name)
        #: `(0, 0)` is Qt's busy indicator, and `total == 0` is a job saying it
        #: cannot count (ADR-0043 §4).
        self.bar.setRange(0, progress.total)
        self.bar.setValue(progress.done)

        #: Reset for a new job, rather than leaving the last one's "Stopping…"
        #: on a button that now controls something else.
        asked = job.cancellation_requested
        self.cancel.setEnabled(not asked)
        self.cancel.setText(STOPPING if asked else "Cancel")

    def _cancel_pressed(self) -> None:
        self.cancel.setEnabled(False)
        self.cancel.setText(STOPPING)
        self._session.cancel_job()
