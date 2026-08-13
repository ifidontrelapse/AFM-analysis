"""Modal questions the window has to ask (M5-T07).

A dialog is here when it asks for something the application cannot know and must
not guess. `ImportOptions` is the first: which instrument produced these files,
and at what scale.
"""

from nanoscope.gui.dialogs.import_options import ImportOptions
from nanoscope.gui.dialogs.settings import SettingsDialog

__all__ = ["ImportOptions", "SettingsDialog"]
