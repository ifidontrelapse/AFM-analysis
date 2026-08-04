"""Reading images off a disk.

The first adapter in the project: everything here takes a path and opens it.
M2-T08 puts an `ImageLoader` port in front of these functions; until then they are
called directly, exactly as they were from `src/afm_io.py`.
"""

from nanoscope.infrastructure.storage.loaders import (
    load_afm,
    load_microscopy_image,
)

__all__ = ["load_afm", "load_microscopy_image"]
