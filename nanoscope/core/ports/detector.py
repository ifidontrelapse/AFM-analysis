"""The contract every particle detector satisfies.

`Detector` is what `application` will depend on so that it never has to know
whether a detection came from a Laplacian-of-Gaussian filter or from a neural
network with 40 MB of weights behind it.

**This is not `BaseDetector`.** They look alike and do different jobs:

- `BaseDetector` (`core.science.detection.base`) is an ABC that the two detectors
  *inherit* — it exists to share `_blobs_to_detections`, which carries the
  `radius_px = sigma * sqrt(2)` relation.
- `Detector` is a `Protocol`, structural: a class satisfies it by having the right
  method, without importing anything from here. That is what lets
  `infrastructure.models.YoloDetector` conform without `core` ever naming it.

Both current implementations already satisfy it, which is asserted — by mypy, at
gate time — in `tests/unit/test_ports.py`. A port nothing implements is a guess;
this one is a description.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from nanoscope.core.entities import Detection


@runtime_checkable
class Detector(Protocol):
    """Finds particles in a prepared image.

    `runtime_checkable` so a composition root can fail loudly at wiring time
    rather than at the first inference; note that it only checks the method
    exists, never its signature — the signature is mypy's job.
    """

    def detect(self, z_above: np.ndarray, pixel_size_nm: float | None) -> list[Detection]:
        """Find particles.

        Args:
            z_above: the image to search — for AFM, `z_flat - substrate`.
            pixel_size_nm: nm per pixel, used to report `radius_nm`. `None` when
                the physical scale is unknown, and then `radius_nm` is `None`
                too — the scale is never invented (D-07, ADR-0019).

        Returns:
            One `Detection` per particle, in the coordinate space of `z_above`.
        """
        ...
