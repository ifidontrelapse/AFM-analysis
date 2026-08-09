"""An absent argument is `None`, and the annotation now says so (M3-T19).

`detect_particles(threshold: float = None)` and
`build_substrate_map(manual_radius_px: float = None)` both branched on `None` in
their bodies while their annotations promised a number — mypy's implicit-Optional
error, and six of the twelve it reported.

Nothing here is executed differently. What these tests pin is the *meaning* the
annotation was corrected to state: `None` is the supported way to say "not
supplied, work it out", so passing it explicitly must equal omitting it. Without
that, `float | None` would be a claim nobody checks.
"""

from __future__ import annotations

import numpy as np

from nanoscope.core.science.detection.log import detect_particles
from nanoscope.core.science.preprocessing import build_substrate_map, flatten_plane

SIZES = {"radii_px": np.array([3.0, 5.0])}


def _particles(size: int = 128, radius: float = 6.0, spacing: int = 32) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size].astype(float)
    z = np.zeros((size, size), dtype=np.float32)
    for cy in range(spacing // 2, size, spacing):
        for cx in range(spacing // 2, size, spacing):
            z += 10.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * (radius / 1.5) ** 2))
    return z


class TestExplicitNoneEqualsTheOmittedArgument:
    def test_the_log_threshold_is_derived_either_way(self) -> None:
        """`threshold=None` means "estimate it adaptively" — the path the golden
        harness has always taken, and the one the annotation used to deny."""
        z_above = flatten_plane(_particles())

        omitted = detect_particles(z_above, 2.0, SIZES)
        explicit = detect_particles(z_above, 2.0, SIZES, threshold=None)

        np.testing.assert_array_equal(omitted, explicit)

    def test_the_opening_radius_is_estimated_either_way(self) -> None:
        """`manual_radius_px=None` is what 100 % of real callers pass — the
        automatic path D-01 crashed on before ADR-0014."""
        z = flatten_plane(_particles())

        omitted = build_substrate_map(z, pixel_size_nm=2.0)
        explicit = build_substrate_map(z, pixel_size_nm=2.0, manual_radius_px=None)

        assert omitted[2] == explicit[2]
        np.testing.assert_array_equal(omitted[0], explicit[0])
