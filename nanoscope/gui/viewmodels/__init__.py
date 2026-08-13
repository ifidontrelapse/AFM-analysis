"""Per-session state, between the widgets and the container (M5-T06).

One class, not one per view: what more than one panel needs is the *session* —
the open project, the selected image, the array that was loaded. State only one
widget can want stays in that widget (ADR-0057).
"""

from nanoscope.gui.viewmodels.session import SessionViewModel

__all__ = ["SessionViewModel"]
