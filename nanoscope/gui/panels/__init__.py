"""The dockable panels (M5-T04 onward).

Each one takes the composition root and calls it. A panel holds no adapter of
its own — Architecture §2.3, and the reason `Nanoscope.repository` is typed as a
port rather than as the SQLite class.
"""

from nanoscope.gui.panels.project_explorer import ProjectExplorer

__all__ = ["ProjectExplorer"]
