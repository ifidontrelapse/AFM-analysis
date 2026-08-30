"""The picture a network is shown, made the same way for training and inference.

A height map is not an image. The scans in a project are 2-D `float32` arrays in
nanometres, often with negative values and a range that depends on the sample;
a network reads 8-bit pixels. Something has to make a picture, and **whatever
makes it decides what the model learns.**

This is that one thing, and it is one thing on purpose. A model trained on
pictures made one way and used on pictures made another is measured on a
question nobody asked, and the failure is silent: no exception, no wrong shape,
just a detector that is worse than it should be for a reason nobody can see. The
copy is what makes it happen — as `display.py`'s second copy of a four-entry
extension map did on 2026-08-30, one milestone before this file existed.

Three decisions are already made, and this is where they are kept:

- **ADR-0015 — normalise in floating point, then cast.** Casting a map in
  nanometres to `uint8` first keeps only the integers inside its range and wraps
  the rest; on one characterization phantom the result was *anti-correlated*
  (r = -0.499) with the correct image.
- **ADR-0023 — invert for `BRIGHT_ON_DARK`.** The weights want dark particles,
  which is what an inverted AFM height map has and what a TEM image already is.
- **ADR-0016 — letterboxing is not here.** It is geometry, and the two callers
  need different geometry: the detector pads to its own square input, while a
  trainer letterboxes to `imgsz` itself and transforms the labels with it. Doing
  it in both places would letterbox twice.

`infrastructure`, because it is `cv2`; `imaging` rather than `models`, because it
belongs to neither side of ADR-0006's separation and both sides import it.
"""

from __future__ import annotations

import numpy as np

from nanoscope.core.values import Polarity


def as_network_input(z: np.ndarray, *, polarity: Polarity) -> np.ndarray:
    """One height map or greyscale image as the `uint8` picture a network sees.

    Args:
        z: a 2-D array. For AFM this is `z_above` — `z_flat - substrate`, what
            `detect` is handed and what every detection in this project was made
            from, not the raw file. For SEM/TEM it is the image as loaded.
        polarity: whether particles are bright on dark, and therefore whether
            the picture is inverted (ADR-0023).

    Returns:
        A `uint8` array of the **same shape**: min-max stretched over the whole
        of `z`, then inverted if the particles are bright. No resizing and no
        padding — see the module docstring.
    """
    import cv2

    picture: np.ndarray = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if polarity is Polarity.BRIGHT_ON_DARK:
        picture = cv2.bitwise_not(picture)
    return picture
