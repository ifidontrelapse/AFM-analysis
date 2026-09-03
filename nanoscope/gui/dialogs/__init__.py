"""Modal questions the window has to ask (M5-T07).

A dialog is here when it asks for something the application cannot know and must
not guess. `ImportOptions` is the first: which instrument produced these files,
and at what scale. `LabelSource` is the second, and the same shape one milestone
on: a label file does not say whether a person or a model drew the box
(M7-T09, ADR-0044).

`ImageChooser` is not one of those: it asks *which files*, which the operator
always knew — what it adds is a look at them first, because a name off a Bruker
is an acquisition number and says nothing about what was scanned.

Neither is `TrainingDialog`, and it is the first one here that is **modeless**:
it configures a run, watches it and stops it, and closing it does not stop the
run — a modal window over six hours of training is the frozen application M5's
third exit criterion rules out (M8-T05).
"""

from nanoscope.gui.dialogs.choose_images import ImageChooser
from nanoscope.gui.dialogs.import_options import ImportOptions
from nanoscope.gui.dialogs.label_source import LabelSource
from nanoscope.gui.dialogs.models import ModelsDialog
from nanoscope.gui.dialogs.settings import SettingsDialog
from nanoscope.gui.dialogs.training import TrainingDialog

__all__ = [
    "ImageChooser",
    "ImportOptions",
    "LabelSource",
    "ModelsDialog",
    "SettingsDialog",
    "TrainingDialog",
]
