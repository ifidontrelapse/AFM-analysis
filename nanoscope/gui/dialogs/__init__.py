"""Modal questions the window has to ask (M5-T07).

A dialog is here when it asks for something the application cannot know and must
not guess. `ImportOptions` is the first: which instrument produced these files,
and at what scale. `LabelSource` is the second, and the same shape one milestone
on: a label file does not say whether a person or a model drew the box
(M7-T09, ADR-0044).

`ImageChooser` is not one of those: it asks *which files*, which the operator
always knew — what it adds is a look at them first, because a name off a Bruker
is an acquisition number and says nothing about what was scanned.
"""

from nanoscope.gui.dialogs.choose_images import ImageChooser
from nanoscope.gui.dialogs.import_options import ImportOptions
from nanoscope.gui.dialogs.label_source import LabelSource
from nanoscope.gui.dialogs.settings import SettingsDialog

__all__ = ["ImageChooser", "ImportOptions", "LabelSource", "SettingsDialog"]
