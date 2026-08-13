"""The dockable panels (M5-T04 onward).

Each one takes the **session viewmodel** and nothing else — not the composition
root, and never another panel. Intent goes up as a method call, state comes back
as a signal, and no panel is wired to a panel (ADR-0057, Architecture §2.3).
"""

from nanoscope.gui.panels.job_status import JobStatus
from nanoscope.gui.panels.log_panel import LogPanel
from nanoscope.gui.panels.project_explorer import ProjectExplorer
from nanoscope.gui.panels.properties import PropertiesPanel
from nanoscope.gui.panels.viewer import ImageViewer

__all__ = ["ImageViewer", "JobStatus", "LogPanel", "ProjectExplorer", "PropertiesPanel"]
