"""An array, as something Qt can draw (M5-T05, moved here 2026-09-02).

`_to_qimage` lived in `panels/viewer.py` while the canvas was the only surface
that drew a scan. It is not, since the import preview and the explorer's
thumbnails: three widgets wrapping numpy buffers in `QImage` is three chances to
get the copy wrong, and the copy is the part that matters.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def to_qimage(rgb: np.ndarray) -> QImage:
    """An `(h, w, 3)` uint8 array as a `QImage`, copied.

    Copied on purpose: `QImage` does not own the buffer it is handed, and a view
    onto a numpy array that Python then frees is a crash that happens later,
    somewhere else.

    Args:
        rgb: `(height, width, 3)` of `uint8` — what `render` and `thumbnail`
            return.

    Returns:
        A `QImage` owning its own pixels.
    """
    height, width, _ = rgb.shape
    contiguous = np.ascontiguousarray(rgb)
    image = QImage(contiguous.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return image.copy()


def to_pixmap(rgb: np.ndarray) -> QPixmap:
    """The same array as a `QPixmap`, for a label or an icon."""
    return QPixmap.fromImage(to_qimage(rgb))
