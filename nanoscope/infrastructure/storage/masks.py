"""A painted mask, on disk (M7-T04, ADR-0073).

PROJECT_RULES §5: *"no mask bitmaps in the database — masks are files, the
database stores paths"*. This is the pair of functions that makes that true, and
they live in `infrastructure` because writing a PNG is `cv2` and `application`
may import neither it nor the filesystem (Architecture §3.2).

**PNG rather than `.npy`:** a mask an operator painted is a picture of their
judgement, and a format every image viewer on their machine can open is worth
more than a few bytes. It is written as 0/255 and read back as a boolean, so
nothing downstream has to remember which convention was used.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from nanoscope.core.errors import InvalidParameterError, MissingFileError


def write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a boolean mask as a 0/255 PNG, creating its directory if needed.

    Raises:
        InvalidParameterError: the file could not be written — a full disk, or a
            path the operator cannot write to. Loud, because a row that points
            at a file nobody wrote is the dangling half of `check_integrity`
            created deliberately.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    painted = (np.asarray(mask, dtype=bool) * 255).astype(np.uint8)
    if not cv2.imwrite(str(path), painted):
        raise InvalidParameterError(f"could not write the painted mask to {path}")


def read_mask(path: Path) -> np.ndarray:
    """Read one back as a boolean array.

    Raises:
        MissingFileError: the file is gone. The row is kept — a missing file is
            as likely to be an unmounted drive as a deletion (ADR-0040) — so the
            caller is told rather than handed an empty mask, which would read as
            *"the operator painted nothing"*.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise MissingFileError(f"no painted mask at {path}")
    return image > 0
